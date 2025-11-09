# src/train_student_kd.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from tqdm import tqdm
import os
import json
from collections import defaultdict
import numpy as np

from teacher_model import CaptioningTeacher
from student_model import CaptioningStudent
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

def validate_student_model(student_model, teacher_model, data_loader, distill_loss, 
                          projectors, device, vocab, max_batches=50):
    """Validation function for student model"""
    student_model.eval()
    teacher_wrapper = TeacherWrapper(teacher_model)
    
    total_loss = 0
    total_samples = 0
    bleu_scores = []
    
    with torch.no_grad():
        for batch_idx, (imgs, captions) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
                
            imgs = imgs.to(device)
            captions = captions.to(device)
            
            captions_input = captions[:-1, :]
            captions_target = captions[1:, :]
            
            # Teacher forward pass - ensure correct precision
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
            
            # Project teacher features to match student dimensions
            teacher_outputs['encoder_features'] = projectors['encoder'](teacher_outputs['encoder_features'])
            
            # Compute loss
            with autocast('cuda'):
                loss, loss_dict = distill_loss(student_outputs, teacher_outputs, captions_target)
            
            total_loss += loss.item() * imgs.size(0)
            total_samples += imgs.size(0)
            
            # Compute BLEU scores for a few samples
            if batch_idx < 5:  # Only compute for first few batches to save time
                predicted_tokens = student_logits.argmax(dim=-1)  # (seq_len, batch_size)
                for i in range(min(2, predicted_tokens.size(1))):  # First 2 samples in batch
                    pred_seq = predicted_tokens[:, i].cpu().numpy()
                    target_seq = captions_target[:, i].cpu().numpy()
                    bleu = compute_bleu_score(pred_seq, target_seq, vocab)
                    bleu_scores.append(bleu)
    
    student_model.train()
    avg_loss = total_loss / total_samples
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    
    return avg_loss, avg_bleu

def train_student_with_kd():
    """Main training function for student model with knowledge distillation"""
    
    # --- Hyperparameters ---
    learning_rate = 2e-4  # Higher than teacher since student is smaller
    batch_size = 16  # Can use larger batch size for smaller model
    accumulation_steps = 2
    num_epochs = 1
    
    # Student model hyperparameters
    embed_size = 256  # Smaller than teacher (512)
    hidden_size = 512
    num_layers = 2
    dropout = 0.3
    
    # Distillation hyperparameters
    alpha = 0.7  # Token-level KD weight
    beta = 0.2   # Encoder feature KD weight
    gamma = 0.1  # Hidden state KD weight
    temperature = 4.0
    
    # Early stopping
    patience = 7
    best_val_loss = float('inf')
    patience_counter = 0
    
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # --- Data Loading ---
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_loader, train_dataset = get_loader(
        root_folder="D:/ImageCaptioner/data/flickr8k/",
        annotation_file="D:/ImageCaptioner/data/flickr8k/captions_clean.csv",
        transform=train_transform,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True
    )
    
    val_loader, _ = get_loader(
        root_folder="D:/ImageCaptioner/data/flickr8k/",
        annotation_file="D:/ImageCaptioner/data/flickr8k/captions_clean.csv",
        transform=val_transform,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False
    )
    
    vocab_size = len(train_dataset.vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # --- Load Teacher Model ---
    print("Loading teacher model...")
    teacher_checkpoint = torch.load('D:/ImageCaptioner/saved_models/best_teacher_model.pth', map_location=device)
    
    teacher_model = CaptioningTeacher(
        vocab_size=vocab_size,
        embed_size=512,  # Teacher's embedding size
        num_heads=8,
        num_decoder_layers=4,
        dropout=0.15
    ).to(device)
    
    teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
    teacher_model.eval()
    print(f"Teacher model loaded! Validation loss was: {teacher_checkpoint['val_loss']:.4f}")
    
    # --- Initialize Student Model ---
    print("Initializing student model...")
    student_model = CaptioningStudent(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        use_attention_refinement=True
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in student_model.parameters())
    trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    
    print(f"Teacher parameters: {teacher_params:,}")
    print(f"Student total parameters: {total_params:,}")
    print(f"Student trainable parameters: {trainable_params:,}")
    print(f"Compression ratio: {teacher_params / total_params:.2f}x")
    
    # --- Setup Distillation ---
    print("Setting up knowledge distillation...")
    
    # Create sample batch for validation
    sample_imgs, sample_captions = next(iter(train_loader))
    sample_batch = (sample_imgs.to(device), sample_captions.to(device))
    
    # Validate setup and create projectors
    projectors, distill_loss = validate_distillation_setup(teacher_model, student_model, sample_batch)
    
    # Update distillation loss parameters
    distill_loss = DistillationLoss(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        temperature=temperature,
        vocab_size=vocab_size
    )
    
    # Move projectors to device (already done in validate_distillation_setup)
    # for key in projectors:
    #     projectors[key] = projectors[key].to(device)
    
    # --- Optimizer and Scheduler ---
    # Separate learning rates for different components
    encoder_params = list(student_model.encoder.parameters())
    decoder_params = list(student_model.decoder.parameters())
    other_params = []
    
    if student_model.use_attention_refinement:
        other_params.extend(list(student_model.attention_refinement.parameters()))
    
    # Add projector parameters
    for projector in projectors.values():
        other_params.extend(list(projector.parameters()))
    
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': learning_rate * 0.1},  # Lower LR for pre-trained encoder
        {'params': decoder_params, 'lr': learning_rate},
        {'params': other_params, 'lr': learning_rate}
    ], weight_decay=0.01)
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    
    # Mixed precision scaler
    scaler = GradScaler('cuda')
    
    # --- Training History ---
    train_losses = []
    val_losses = []
    val_bleu_scores = []
    loss_components_history = defaultdict(list)
    
    # --- Training Loop ---
    print("Starting knowledge distillation training...")
    teacher_wrapper = TeacherWrapper(teacher_model)
    
    for epoch in range(num_epochs):
        student_model.train()
        epoch_loss = 0
        epoch_loss_components = defaultdict(float)
        num_batches = 0
        
        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        for batch_idx, (imgs, captions) in enumerate(loop):
            imgs = imgs.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)
            
            captions_input = captions[:-1, :]
            captions_target = captions[1:, :]
            
            # Teacher forward pass (no gradients) - ensure FP32
            imgs_fp32 = imgs.float()
            captions_input_fp32 = captions_input.long()
            teacher_outputs = teacher_wrapper(imgs_fp32, captions_input_fp32)
            
            # Student forward pass
            with autocast('cuda'):
                student_logits, student_encoder_features, student_hidden_states, _ = student_model(imgs, captions_input)
                
                student_outputs = {
                    'logits': student_logits,
                    'encoder_features': student_encoder_features,
                    'hidden_states': student_hidden_states
                }
                
                # Project teacher features to match student dimensions
                teacher_outputs['encoder_features'] = projectors['encoder'](teacher_outputs['encoder_features'])
                
                # Compute distillation loss
                loss, loss_dict = distill_loss(student_outputs, teacher_outputs, captions_target)
                loss = loss / accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                
                # Update projector parameters too
                for projector in projectors.values():
                    torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                scheduler.step(epoch + batch_idx / len(train_loader))
            
            # Update statistics
            epoch_loss += loss.item() * accumulation_steps
            for key, value in loss_dict.items():
                epoch_loss_components[key] += value
            num_batches += 1
            
            # Update progress bar
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix({
                'loss': loss.item() * accumulation_steps,
                'token_kd': loss_dict['token_kd_loss'],
                'lr': optimizer.param_groups[0]['lr']
            })
            
            # Log progress
            if batch_idx % 100 == 0:
                log_training_progress(
                    epoch + 1, batch_idx, loss_dict, 
                    optimizer.param_groups[0]['lr'], len(train_loader)
                )
        
        # Calculate average losses
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        for key in epoch_loss_components:
            loss_components_history[key].append(epoch_loss_components[key] / num_batches)
        
        # Validation
        if epoch % 2 == 0:  # Validate every 2 epochs
            val_loss, val_bleu = validate_student_model(
                student_model, teacher_model, val_loader, distill_loss,
                projectors, device, train_dataset.vocab
            )
            val_losses.append(val_loss)
            val_bleu_scores.append(val_bleu)
            
            print(f"\nEpoch {epoch+1}:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val BLEU-1: {val_bleu:.4f}")
            print(f"  Token KD: {epoch_loss_components['token_kd_loss']/num_batches:.4f}")
            print(f"  Feature KD: {epoch_loss_components['feature_kd_loss']/num_batches:.4f}")
            print(f"  Hidden KD: {epoch_loss_components['hidden_kd_loss']/num_batches:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
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
                        'dropout': dropout
                    },
                    'distillation_config': {
                        'alpha': alpha,
                        'beta': beta,
                        'gamma': gamma,
                        'temperature': temperature
                    }
                }, 'saved_models/best_student_model.pth')
                print(f"  New best model saved! Val Loss: {val_loss:.4f}, BLEU: {val_bleu:.4f}")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break
        else:
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}")
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'student_state_dict': student_model.state_dict(),
        'projectors_state_dict': {k: v.state_dict() for k, v in projectors.items()},
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_bleu_scores': val_bleu_scores,
        'loss_components': dict(loss_components_history),
        'vocab_size': vocab_size,
        'model_config': {
            'embed_size': embed_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout
        }
    }, 'saved_models/final_student_model.pth')
    
    # Save training history
    with open('saved_models/student_training_history.json', 'w') as f:
        json.dump({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_bleu_scores': val_bleu_scores,
            'loss_components': {k: v for k, v in loss_components_history.items()},
            'hyperparameters': {
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'embed_size': embed_size,
                'hidden_size': hidden_size,
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
                'temperature': temperature
            }
        }, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final validation BLEU: {val_bleu_scores[-1] if val_bleu_scores else 'N/A'}")
    print(f"Model compression ratio: {teacher_params / total_params:.2f}x")
    
    return student_model, projectors

if __name__ == "__main__":
    train_student_with_kd()
