# src/train_kd.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os

from models import CaptioningTeacher
from student_model import CaptioningStudent
from data_loader import get_loader
from torch.amp import GradScaler, autocast

class KnowledgeDistillationLoss(nn.Module):
    """Combined loss for knowledge distillation"""
    
    def __init__(self, alpha=0.7, temperature=4.0, vocab_size=None, pad_idx=None):
        super().__init__()
        self.alpha = alpha  # Weight for distillation loss
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=pad_idx)
        
    def forward(self, student_logits, teacher_logits, targets):
        # Standard cross-entropy loss
        ce_loss = self.ce_loss(student_logits.view(-1, student_logits.size(-1)), targets.view(-1))
        
        # Knowledge distillation loss (soft targets)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KL divergence loss
        kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
        
        return total_loss, ce_loss.item(), kd_loss.item()

def load_teacher_model(model_path, vocab_size, device):
    """Load the pre-trained teacher model"""
    checkpoint = torch.load(model_path, map_location=device)
    
    teacher = CaptioningTeacher(
        vocab_size=vocab_size,
        embed_size=512,  
        num_heads=8,
        num_decoder_layers=4,
        dropout=0.15
    ).to(device)
    
    teacher.load_state_dict(checkpoint['model_state_dict'])
    teacher.eval()  # Set to evaluation mode
    
    # Freeze teacher parameters
    for param in teacher.parameters():
        param.requires_grad = False
        
    print("Teacher model loaded successfully!")
    return teacher

def train_student_with_kd():
    # Hyperparameters
    learning_rate = 2e-4
    batch_size = 16  # Can use larger batch for student
    num_epochs = 20
    temperature = 4.0
    alpha = 0.8  # Weight for KD loss
    beta = 0.3   # Weight for feature-based distillation loss
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Data loading
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader, dataset = get_loader(
        root_folder="data/flickr8k/Images",
        annotation_file="data/flickr8k/captions_clean.csv",
        transform=transform,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True
    )

    vocab_size = len(dataset.vocab)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    
    # Load teacher model
    teacher = load_teacher_model('saved_models/best_teacher_model.pth', vocab_size, device)
    
    # Initialize student model
    student = CaptioningStudent(
        vocab_size=vocab_size,
        embed_size=256,  # Much smaller than teacher
        num_heads=4,
        num_decoder_layers=2,
        dropout=0.1
    ).to(device)
    
    # Count parameters
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    print(f"Teacher parameters: {teacher_params:,}")
    print(f"Student parameters: {student_params:,}")
    print(f"Compression ratio: {teacher_params/student_params:.1f}x")

    student_feature_dim = 256 
    teacher_feature_dim = 512 
    feature_projection = nn.Linear(student_feature_dim, teacher_feature_dim).to(device)
    
    # Loss and optimizer
    kd_criterion = KnowledgeDistillationLoss(alpha=alpha, temperature=temperature, 
                                           vocab_size=vocab_size, pad_idx=pad_idx)
    # Add a simple MSE loss for comparing feature vectors
    feature_loss_criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        list(student.parameters()) + list(feature_projection.parameters()), 
        lr=learning_rate, weight_decay=0.01
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    scaler = GradScaler('cuda')
    
    # Training loop
    print("Starting Knowledge Distillation training...")
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        student.train()
        feature_projection.train()
        epoch_loss = 0
        epoch_ce_loss = 0
        epoch_kd_loss = 0
        epoch_feat_loss = 0
        
        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        for imgs, captions in loop:
            imgs = imgs.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)
            
            captions_input = captions[:-1, :]
            captions_target = captions[1:, :]
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                # Get teacher predictions (no gradients)
                with torch.no_grad():
                    teacher_logits = teacher(imgs, captions_input)
                    teacher_features = teacher.encoder.forward_features(imgs)
                    teacher_features = teacher.encoder_projection(teacher_features) # Shape: (batch, 197, 512)
                
                # Get student predictions
                student_logits = student(imgs, captions_input)
                student_raw_features = student.encoder.forward_features(imgs)
                if len(student_raw_features.shape) == 4:
                    B, C, H, W = student_raw_features.shape
                    student_raw_features = student_raw_features.view(B, C, H*W).permute(0, 2, 1)
                else:
                    student_raw_features = student_raw_features
                
                student_features = student.encoder_projection(student_raw_features)
                
                # --- Loss Calculation ---
                # 1. Calculate original distillation loss (logits + cross-entropy)
                logit_kd_loss, ce_loss, kd_loss = kd_criterion(student_logits, teacher_logits, captions_target)

                # 2. NEW: Calculate feature distillation loss
                # Project student features to match teacher's dimension for comparison
                student_features_for_comparison = feature_projection(student_features)

                 # We need to ensure features are comparable.
                # The ViT encoder produces a class token + patch tokens. Let's use all of them.
                # To handle potentially different sequence lengths (e.g., 197 vs 49),
                # we can use average pooling across the sequence dimension.
                teacher_features_pooled = torch.mean(teacher_features, dim=1)
                student_features_pooled = torch.mean(student_features_for_comparison, dim=1)
                
                loss_features = feature_loss_criterion(student_features_pooled, teacher_features_pooled)
                
                # 3. NEW: Combine all losses into the final loss
                # The original kd_loss is weighted by alpha, the new feature loss by beta.
                # The remaining weight (1-beta) is applied to the logit-based loss.
                loss = (1 - beta) * logit_kd_loss + beta * loss_features
            
            # Backward pass (uses the final combined 'loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # We need to clip gradients for both the student and the new projection layer
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(feature_projection.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # --- Accumulate all loss components ---
            epoch_loss += loss.item()
            epoch_ce_loss += ce_loss
            epoch_kd_loss += kd_loss
            epoch_feat_loss += loss_features.item()
            
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix({
                'total_loss': loss.item(),
                'ce_loss': ce_loss,
                'kd_loss': kd_loss,
                'feat_loss': loss_features.item(),
                'lr': optimizer.param_groups[0]['lr']
            })
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        avg_ce_loss = epoch_ce_loss / len(train_loader)
        avg_kd_loss = epoch_kd_loss / len(train_loader)
        avg_feat_loss = epoch_feat_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}: Loss: {avg_loss:.4f}, CE: {avg_ce_loss:.4f}, KD: {avg_kd_loss:.4f}, Feat: {avg_feat_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            if not os.path.exists('saved_models'):
                os.makedirs('saved_models')
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'vocab_size': vocab_size
            }, 'saved_models/best_student_model.pth')
            print(f"Best student model saved with loss: {avg_loss:.4f}")
    
    print(f"Knowledge Distillation completed! Best loss: {best_loss:.4f}")
    
    # Compare model sizes
    teacher_size = sum(p.numel() * 4 for p in teacher.parameters()) / (1024**2)  # MB
    student_size = sum(p.numel() * 4 for p in student.parameters()) / (1024**2)  # MB
    print(f"Teacher model size: {teacher_size:.1f} MB")
    print(f"Student model size: {student_size:.1f} MB")
    print(f"Size reduction: {teacher_size/student_size:.1f}x smaller")

if __name__ == "__main__":
    train_student_with_kd()