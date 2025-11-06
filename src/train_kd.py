# src/train_simple_kd.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os

from models import CaptioningTeacher
from student_model import LightweightCaptioningStudent
from data_loader import get_loader
from torch.amp import GradScaler, autocast

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
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print("Teacher model loaded successfully!")
    return teacher

def train_lightweight_student():
    """Train lightweight student with simplified distillation"""
    
    # --- Hyperparameters ---
    learning_rate = 3e-4
    batch_size = 32
    num_epochs = 30
    beta = 0.5   # Weight for feature distillation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Data Loading ---
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
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training batches: {len(train_loader)}")
    
    # --- Model Setup ---
    teacher = load_teacher_model('saved_models/best_teacher_model.pth', vocab_size, device)
    
    student = LightweightCaptioningStudent(
        vocab_size=vocab_size,
        embed_size=128,
        hidden_size=256,
        num_layers=1,
        dropout=0.2
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"\nLightweight Student Model:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Feature projection for distillation
    student_feature_dim = student.encoder_dim  # 576 for MobileNetV3
    teacher_feature_dim = 512
    feature_projection = nn.Linear(student_feature_dim, teacher_feature_dim).to(device)
    
    # --- Loss and Optimizer ---
    ce_criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    feature_loss_criterion = nn.MSELoss()
    
    optimizer = optim.AdamW(
        list(student.parameters()) + list(feature_projection.parameters()), 
        lr=learning_rate, 
        weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler = GradScaler('cuda')
    
    # --- Training Loop ---
    print(f"\nStarting Lightweight Student Training...")
    print(f"Feature distillation weight: {beta}")
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        student.train()
        feature_projection.train()
        epoch_loss, epoch_ce_loss, epoch_feat_loss = 0, 0, 0
        
        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")

        for batch_idx, (imgs, captions) in enumerate(loop):
            imgs = imgs.to(device)
            captions = captions.to(device)
            
            # Debug: Check batch sizes (only for first batch)
            if batch_idx == 0:
                print(f"[DEBUG] Batch {batch_idx}: imgs.shape={imgs.shape}, captions.shape={captions.shape}")
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                # --- Teacher Feature Extraction ---
                with torch.no_grad():
                    teacher_features_seq = teacher.encoder.forward_features(imgs)
                    teacher_features_seq = teacher.encoder_projection(teacher_features_seq)
                    teacher_features_pooled = torch.mean(teacher_features_seq, dim=1)
                    
                # --- Student Forward Pass ---
                student_logits = student(imgs, captions)
                student_features = student.get_image_features(imgs)

                # --- LOSS CALCULATION ---
                # 1. Standard Cross-Entropy Loss (learning from data)
                # Ensure sequence lengths match by taking minimum length
                seq_len_target = captions.size(0) - 1  # Remove START token
                seq_len_logits = student_logits.size(0)
                min_seq_len = min(seq_len_target, seq_len_logits)
                
                # Truncate both to same sequence length
                targets = captions[1:min_seq_len+1, :].permute(1, 0).reshape(-1)  # Skip START, take min_seq_len tokens
                logits = student_logits[:min_seq_len, :, :].reshape(-1, student_logits.size(-1))
                
                # Final safety check - if shapes still don't match, force alignment
                if targets.size(0) != logits.size(0):
                    min_size = min(targets.size(0), logits.size(0))
                    targets = targets[:min_size]
                    logits = logits[:min_size]
                    if min_size == 0:  # Skip if no valid data
                        continue
                
                loss_ce = ce_criterion(logits, targets)

                # 2. Feature Distillation Loss (learning to "see" like the teacher)
                student_features_projected = feature_projection(student_features)
                loss_features = feature_loss_criterion(student_features_projected, teacher_features_pooled)
                
                # 3. Combine the two losses
                loss = (1 - beta) * loss_ce + beta * loss_features
            
            # --- Backward Pass ---
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(feature_projection.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Update logs
            epoch_loss += loss.item()
            epoch_ce_loss += loss_ce.item()
            epoch_feat_loss += loss_features.item()
            
            loop.set_postfix({
                'loss': loss.item(), 
                'ce_loss': loss_ce.item(), 
                'feat_loss': loss_features.item()
            })
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        avg_ce_loss = epoch_ce_loss / len(train_loader)
        avg_feat_loss = epoch_feat_loss / len(train_loader)
        
        print(f"Epoch {epoch+1} Summary -> Total Loss: {avg_loss:.4f}, CE Loss: {avg_ce_loss:.4f}, Feature Loss: {avg_feat_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': student.state_dict(),
                'feature_projection_state_dict': feature_projection.state_dict(),
                'loss': avg_loss,
                'epoch': epoch,
                'vocab_size': vocab_size
            }, 'saved_models/best_lightweight_student_model.pth')
            print(f"✅ Best lightweight student model saved with loss: {avg_loss:.4f}")

    print(f"\n🎉 Training completed! Best loss: {best_loss:.4f}")
    print("Model saved as: saved_models/best_lightweight_student_model.pth")

if __name__ == "__main__":
    train_lightweight_student()
