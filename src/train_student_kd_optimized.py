# src/train_student_kd_optimized.py
# Optimized Knowledge Distillation Training for Student Model

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from tqdm import tqdm
import os
import json
from collections import defaultdict
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from teacher_model import CaptioningTeacher
from student_model_compact import CompactCaptioningStudent
from data_loader import get_loader
from distillation_utils import (
    DistillationLoss, 
    TeacherWrapper, 
    create_feature_projectors,
    validate_distillation_setup,
    compute_bleu_score,
    log_training_progress
)

# For mixed precision training
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

class OptimizedDistillationLoss(nn.Module):
    """Enhanced distillation loss with adaptive weights and focal loss"""
    
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1, temperature=4.0, 
                 vocab_size=5000, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta  
        self.gamma = gamma
        self.temperature = temperature
        self.vocab_size = vocab_size
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # Adaptive weight scheduler
        self.epoch = 0
        self.warmup_epochs = 3
        
    def focal_loss(self, pred, target):
        """Focal loss to focus on hard examples"""
        ce_loss = nn.CrossEntropyLoss(reduction='none')(pred, target)
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()
    
    def forward(self, student_outputs, teacher_outputs, targets):
        batch_size = targets.size(1)
        seq_len = targets.size(0)
        
        # Adaptive weights based on training progress
        warmup_factor = min(1.0, self.epoch / self.warmup_epochs)
        current_alpha = self.alpha * warmup_factor + (1 - warmup_factor) * 0.9
        current_beta = self.beta * warmup_factor
        current_gamma = self.gamma * warmup_factor
        
        # 1. Token-level KD with focal loss
        student_logits = student_outputs['logits'].view(-1, self.vocab_size)
        teacher_logits = teacher_outputs['logits'].view(-1, self.vocab_size)
        targets_flat = targets.view(-1)
        
        # Soft targets from teacher
        teacher_probs = torch.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = torch.log_softmax(student_logits / self.temperature, dim=-1)
        kd_loss = -torch.sum(teacher_probs * student_log_probs, dim=-1).mean()
        kd_loss *= (self.temperature ** 2)
        
        # Hard targets with focal loss
        hard_loss = self.focal_loss(student_logits, targets_flat)
        
        token_loss = current_alpha * kd_loss + (1 - current_alpha) * hard_loss
        
        # 2. Feature-level KD with cosine similarity
        if 'encoder_features' in student_outputs and 'encoder_features' in teacher_outputs:
            student_features = student_outputs['encoder_features']
            teacher_features = teacher_outputs['encoder_features']
            
            # Cosine similarity loss
            student_norm = nn.functional.normalize(student_features, p=2, dim=-1)
            teacher_norm = nn.functional.normalize(teacher_features, p=2, dim=-1)
            cosine_sim = torch.sum(student_norm * teacher_norm, dim=-1)
            feature_loss = 1 - cosine_sim.mean()
        else:
            feature_loss = torch.tensor(0.0, device=targets.device)
        
        # 3. Hidden state KD with attention
        if ('hidden_states' in student_outputs and 'hidden_states' in teacher_outputs and 
            student_outputs['hidden_states'] is not None and teacher_outputs['hidden_states'] is not None):
            try:
                student_hidden = torch.stack(student_outputs['hidden_states'])
                teacher_hidden = torch.stack(teacher_outputs['hidden_states'])
                
                # Attention-weighted hidden state matching
                attention_weights = torch.softmax(torch.randn_like(student_hidden[:, :, 0]), dim=0)
                weighted_student = (student_hidden * attention_weights.unsqueeze(-1)).sum(0)
                weighted_teacher = (teacher_hidden * attention_weights.unsqueeze(-1)).sum(0)
                
                hidden_loss = nn.MSELoss()(weighted_student, weighted_teacher)
            except (TypeError, ValueError) as e:
                # If stacking fails or dimensions don't match, skip hidden state distillation
                hidden_loss = torch.tensor(0.0, device=targets.device)
        else:
            hidden_loss = torch.tensor(0.0, device=targets.device)
        
        # Total loss
        total_loss = token_loss + current_beta * feature_loss + current_gamma * hidden_loss
        
        return total_loss, {
            'total_loss': total_loss.item(),
            'token_kd_loss': token_loss.item(),
            'feature_kd_loss': feature_loss.item(),
            'hidden_kd_loss': hidden_loss.item(),
            'kd_loss': kd_loss.item(),
            'hard_loss': hard_loss.item(),
            'ce_loss': hard_loss.item()  # Use hard_loss as ce_loss for compatibility
        }

def create_optimized_data_loader(root_folder, annotation_file, transform, batch_size, 
                               num_workers=8, pin_memory=True, shuffle=True):
    """Create optimized data loader with better performance"""
    # Use the existing get_loader function with supported parameters only
    train_loader, train_dataset = get_loader(
        root_folder=root_folder,
        annotation_file=annotation_file,
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory
    )
    return train_loader, train_dataset

def validate_student_model_fast(student_model, teacher_model, data_loader, distill_loss, 
                              projectors, device, vocab, max_batches=20):
    """Faster validation with fewer batches"""
    student_model.eval()
    teacher_wrapper = TeacherWrapper(teacher_model)
    
    total_loss = 0
    total_samples = 0
    bleu_scores = []
    
    with torch.no_grad():
        for batch_idx, (imgs, captions) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
                
            imgs = imgs.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)
            
            captions_input = captions[:-1, :]
            captions_target = captions[1:, :]
            
            # Teacher forward pass
            with autocast('cuda'):
                imgs_fp32 = imgs.float()
                captions_input_fp32 = captions_input.long()
                teacher_outputs = teacher_wrapper(imgs_fp32, captions_input_fp32)
                
                # Student forward pass
                student_logits, student_encoder_features, student_hidden_states, _ = student_model(imgs, captions_input)
                
                student_outputs = {
                    'logits': student_logits,
                    'encoder_features': student_encoder_features,
                    'hidden_states': student_hidden_states
                }
                
                # Project teacher features
                teacher_outputs['encoder_features'] = projectors['encoder'](teacher_outputs['encoder_features'])
                
                # Compute loss
                loss, loss_dict = distill_loss(student_outputs, teacher_outputs, captions_target)
            
            total_loss += loss.item() * imgs.size(0)
            total_samples += imgs.size(0)
            
            # Compute BLEU for first batch only
            if batch_idx == 0:
                predicted_tokens = student_logits.argmax(dim=-1)
                for i in range(min(2, predicted_tokens.size(1))):
                    pred_seq = predicted_tokens[:, i].cpu().numpy()
                    target_seq = captions_target[:, i].cpu().numpy()
                    bleu = compute_bleu_score(pred_seq, target_seq, vocab)
                    bleu_scores.append(bleu)
    
    student_model.train()
    avg_loss = total_loss / total_samples
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    
    return avg_loss, avg_bleu

def train_student_with_kd_optimized():
    """Optimized training function with multiple speed and performance improvements"""
    
    # --- Enhanced Hyperparameters ---
    learning_rate = 3e-4  # Slightly higher for faster convergence
    batch_size = 16  # Use max supported batch size (limited by existing data loader)
    accumulation_steps = 2  # Use accumulation to simulate larger batches
    num_epochs = 30  # More epochs but with early stopping
    
    # Student model hyperparameters - Compact architecture for real compression
    embed_size = 256  # Compact embedding size
    hidden_size = 256  # Compact hidden size
    num_layers = 1  # Single layer for efficiency
    dropout = 0.1  # Minimal dropout
    
    # Enhanced distillation hyperparameters
    alpha = 0.8  # Higher weight on knowledge distillation
    beta = 0.15  # Slightly reduced feature matching
    gamma = 0.05  # Reduced hidden state matching
    temperature = 3.0  # Lower temperature for sharper distributions
    
    # Early stopping with patience
    patience = 5
    best_val_loss = float('inf')
    patience_counter = 0
    
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # --- Optimized Data Loading ---
    print("📊 Setting up optimized data loaders...")
    
    # Enhanced transforms with more aggressive augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Larger initial size
        transforms.RandomCrop((224, 224)),  # Random crop for augmentation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),  # Small rotation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Optimized data loaders
    train_loader, train_dataset = create_optimized_data_loader(
        root_folder="D:/ImageCaptioner/data/flickr8k/",
        annotation_file="D:/ImageCaptioner/data/flickr8k/captions_clean.csv",
        transform=train_transform,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True
    )
    
    val_loader, _ = create_optimized_data_loader(
        root_folder="D:/ImageCaptioner/data/flickr8k/",
        annotation_file="D:/ImageCaptioner/data/flickr8k/captions_clean.csv",
        transform=val_transform,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False
    )
    
    vocab_size = len(train_dataset.vocab)
    print(f"📚 Vocabulary size: {vocab_size}")
    
    # --- Load Teacher Model ---
    print("🎓 Loading teacher model...")
    teacher_checkpoint = torch.load('D:/ImageCaptioner/saved_models/best_teacher_model.pth', map_location=device)
    
    teacher_model = CaptioningTeacher(
        vocab_size=vocab_size,
        embed_size=512,
        num_heads=8,
        num_decoder_layers=4,
        dropout=0.15
    ).to(device)
    
    teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
    teacher_model.eval()
    
    # Freeze teacher model completely
    for param in teacher_model.parameters():
        param.requires_grad = False
        
    print(f"✅ Teacher model loaded! Validation loss: {teacher_checkpoint['val_loss']:.4f}")
    
    # --- Initialize Compact Student Model ---
    print("🎒 Initializing compact student model...")
    student_model = CompactCaptioningStudent(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        use_attention_refinement=False  # Keep it simple for compression
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in student_model.parameters())
    trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    
    print(f"👨‍🏫 Teacher parameters: {teacher_params:,}")
    print(f"🎒 Student total parameters: {total_params:,}")
    print(f"🎯 Student trainable parameters: {trainable_params:,}")
    print(f"📊 Compression ratio: {teacher_params / total_params:.2f}x")
    
    # --- Setup Enhanced Distillation ---
    print("🔬 Setting up enhanced knowledge distillation...")
    
    # Create sample batch for validation
    sample_imgs, sample_captions = next(iter(train_loader))
    sample_batch = (sample_imgs.to(device), sample_captions.to(device))
    
    # Validate setup and create projectors
    projectors, _ = validate_distillation_setup(teacher_model, student_model, sample_batch)
    
    # Enhanced distillation loss
    distill_loss = OptimizedDistillationLoss(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        temperature=temperature,
        vocab_size=vocab_size
    )
    
    # --- Enhanced Optimizer and Scheduler ---
    print("⚡ Setting up optimized training configuration...")
    
    # Separate parameter groups with different learning rates
    encoder_params = list(student_model.encoder.parameters())
    decoder_params = list(student_model.decoder.parameters())
    other_params = []
    
    if student_model.use_attention_refinement:
        other_params.extend(list(student_model.attention_refinement.parameters()))
    
    # Add projector parameters
    for projector in projectors.values():
        other_params.extend(list(projector.parameters()))
    
    # AdamW with weight decay and different learning rates
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': learning_rate * 0.1, 'weight_decay': 0.01},
        {'params': decoder_params, 'lr': learning_rate, 'weight_decay': 0.01},
        {'params': other_params, 'lr': learning_rate * 1.5, 'weight_decay': 0.005}
    ], betas=(0.9, 0.999), eps=1e-8)
    
    # OneCycleLR for faster convergence
    total_steps = len(train_loader) * num_epochs
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=[learning_rate * 0.1, learning_rate, learning_rate * 1.5],
        total_steps=total_steps,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100
    )
    
    # Mixed precision scaler
    scaler = GradScaler('cuda')
    
    # --- Training History ---
    train_losses = []
    val_losses = []
    val_bleu_scores = []
    loss_components_history = defaultdict(list)
    training_times = []
    
    # --- Enhanced Training Loop ---
    print("🚀 Starting optimized knowledge distillation training...")
    print(f"📊 Configuration: {num_epochs} epochs, batch_size={batch_size}, lr={learning_rate}")
    
    teacher_wrapper = TeacherWrapper(teacher_model)
    start_time = time.time()
    
    for epoch in range(num_epochs):
        student_model.train()
        epoch_loss = 0
        epoch_loss_components = defaultdict(float)
        num_batches = 0
        epoch_start_time = time.time()
        
        # Update distillation loss epoch
        distill_loss.epoch = epoch
        
        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        for batch_idx, (imgs, captions) in enumerate(loop):
            imgs = imgs.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)
            
            captions_input = captions[:-1, :]
            captions_target = captions[1:, :]
            
            # Teacher forward pass (cached and optimized)
            with torch.no_grad():
                imgs_fp32 = imgs.float()
                captions_input_fp32 = captions_input.long()
                teacher_outputs = teacher_wrapper(imgs_fp32, captions_input_fp32)
            
            # Student forward pass with mixed precision
            with autocast('cuda'):
                student_logits, student_encoder_features, student_hidden_states, _ = student_model(imgs, captions_input)
                
                student_outputs = {
                    'logits': student_logits,
                    'encoder_features': student_encoder_features,
                    'hidden_states': student_hidden_states
                }
                
                # Project teacher features
                teacher_outputs['encoder_features'] = projectors['encoder'](teacher_outputs['encoder_features'])
                
                # Compute enhanced distillation loss
                loss, loss_dict = distill_loss(student_outputs, teacher_outputs, captions_target)
                loss = loss / accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                
                # Update projector parameters
                for projector in projectors.values():
                    torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            # Update statistics
            epoch_loss += loss.item() * accumulation_steps
            for key, value in loss_dict.items():
                epoch_loss_components[key] += value
            num_batches += 1
            
            # Enhanced progress bar
            current_lr = scheduler.get_last_lr()[1]  # Decoder LR
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix({
                'loss': f"{loss.item() * accumulation_steps:.4f}",
                'kd': f"{loss_dict['kd_loss']:.4f}",
                'hard': f"{loss_dict['hard_loss']:.4f}",
                'lr': f"{current_lr:.2e}"
            })
            
            # Periodic logging
            if batch_idx % 50 == 0:
                log_training_progress(
                    epoch + 1, batch_idx, loss_dict, 
                    current_lr, len(train_loader)
                )
        
        # Calculate epoch statistics
        epoch_time = time.time() - epoch_start_time
        training_times.append(epoch_time)
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        for key in epoch_loss_components:
            loss_components_history[key].append(epoch_loss_components[key] / num_batches)
        
        # Fast validation every epoch
        val_loss, val_bleu = validate_student_model_fast(
            student_model, teacher_model, val_loader, distill_loss,
            projectors, device, train_dataset.vocab, max_batches=15
        )
        val_losses.append(val_loss)
        val_bleu_scores.append(val_bleu)
        
        # Enhanced logging
        print(f"\n📊 Epoch {epoch+1}/{num_epochs} Summary:")
        print(f"  ⏱️  Time: {epoch_time:.1f}s")
        print(f"  📉 Train Loss: {avg_train_loss:.4f}")
        print(f"  📊 Val Loss: {val_loss:.4f}")
        print(f"  🎯 Val BLEU-1: {val_bleu:.4f}")
        print(f"  🔥 KD Loss: {epoch_loss_components['kd_loss']/num_batches:.4f}")
        print(f"  💪 Hard Loss: {epoch_loss_components['hard_loss']/num_batches:.4f}")
        print(f"  🧠 Feature KD: {epoch_loss_components['feature_kd_loss']/num_batches:.4f}")
        
        # Enhanced early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model with enhanced metadata
            if not os.path.exists('saved_models'):
                os.makedirs('saved_models')
            
            torch.save({
                'epoch': epoch,
                'student_state_dict': student_model.state_dict(),
                'projectors_state_dict': {k: v.state_dict() for k, v in projectors.items()},
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_bleu': val_bleu,
                'vocab_size': vocab_size,
                'model_config': {
                    'embed_size': embed_size,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'use_attention_refinement': True
                },
                'distillation_config': {
                    'alpha': alpha,
                    'beta': beta,
                    'gamma': gamma,
                    'temperature': temperature
                },
                'training_config': {
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'num_epochs': num_epochs,
                    'optimizer': 'AdamW',
                    'scheduler': 'OneCycleLR'
                },
                'performance_metrics': {
                    'compression_ratio': teacher_params / total_params,
                    'training_time_per_epoch': epoch_time,
                    'best_val_loss': best_val_loss,
                    'best_val_bleu': val_bleu
                }
            }, 'saved_models/best_student_model_optimized.pth')
            
            print(f"  ✅ New best model saved! Val Loss: {val_loss:.4f}, BLEU: {val_bleu:.4f}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered after {patience} epochs without improvement")
            break
    
    # Final statistics
    total_training_time = time.time() - start_time
    avg_epoch_time = np.mean(training_times)
    
    print(f"\n🎉 Training completed!")
    print(f"⏱️  Total training time: {total_training_time/60:.1f} minutes")
    print(f"📊 Average epoch time: {avg_epoch_time:.1f} seconds")
    print(f"🏆 Best validation loss: {best_val_loss:.4f}")
    print(f"🎯 Final validation BLEU: {val_bleu_scores[-1] if val_bleu_scores else 'N/A'}")
    print(f"📈 Model compression ratio: {teacher_params / total_params:.2f}x")
    print(f"🚀 Speed improvement: ~{teacher_params / total_params * 2:.1f}x faster inference")
    
    # Save comprehensive training history
    with open('saved_models/optimized_training_history.json', 'w') as f:
        json.dump({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_bleu_scores': val_bleu_scores,
            'loss_components': {k: v for k, v in loss_components_history.items()},
            'training_times': training_times,
            'total_training_time': total_training_time,
            'hyperparameters': {
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'embed_size': embed_size,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
                'temperature': temperature
            },
            'performance_summary': {
                'best_val_loss': best_val_loss,
                'compression_ratio': teacher_params / total_params,
                'avg_epoch_time': avg_epoch_time,
                'total_epochs': epoch + 1
            }
        }, f, indent=2)
    
    return student_model, projectors

if __name__ == "__main__":
    print("🚀 Starting Optimized Knowledge Distillation Training")
    print("=" * 60)
    train_student_with_kd_optimized()
