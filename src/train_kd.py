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
from student_model import CaptioningStudent # Your CNN-LSTM student
from data_loader import get_loader
from torch.cuda.amp import GradScaler, autocast

# load_teacher_model function remains the same
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

def train_student_with_kd():
    # --- Hyperparameters ---
    learning_rate = 3e-4
    batch_size = 32
    num_epochs = 30
    # We now only need one weight: beta, for the feature distillation.
    # The rest of the loss will be standard cross-entropy.
    beta = 0.5   # Weight for feature-based distillation loss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ... (Data loading remains the same) ...
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_loader, dataset = get_loader(
        root_folder="data/flickr8k/Images", annotation_file="data/flickr8k/captions_clean.csv",
        transform=transform, batch_size=batch_size, num_workers=4, shuffle=True
    )
    vocab_size = len(dataset.vocab)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    
    # --- MODEL LOADING & SETUP ---
    teacher = load_teacher_model('saved_models/best_teacher_model.pth', vocab_size, device)
    student = CaptioningStudent(vocab_size=vocab_size, embed_size=256, hidden_size=512, num_layers=2, dropout=0.5).to(device)
    
    student_feature_dim = 2048
    teacher_feature_dim = 512
    feature_projection = nn.Linear(student_feature_dim, teacher_feature_dim).to(device)
    
    # --- LOSS AND OPTIMIZER (SIMPLIFIED) ---
    # We only need the standard CrossEntropyLoss and MSELoss now
    ce_criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    feature_loss_criterion = nn.MSELoss()
    
    # The optimizer no longer needs to worry about the KnowledgeDistillationLoss class
    optimizer = optim.AdamW(
        list(student.parameters()) + list(feature_projection.parameters()), 
        lr=learning_rate, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler = GradScaler()

    # --- TRAINING LOOP (SIMPLIFIED) ---
    print("Starting Simplified Distillation (CE + Feature Loss) for CNN-LSTM Student...")
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        student.train()
        feature_projection.train()
        epoch_loss, epoch_ce_loss, epoch_feat_loss = 0, 0, 0
        
        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")

        for imgs, captions in loop:
            imgs = imgs.to(device)
            captions = captions.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                # --- Teacher Feature Extraction ---
                with torch.no_grad():
                    teacher_features_seq = teacher.encoder.forward_features(imgs)
                    teacher_features_seq = teacher.encoder_projection(teacher_features_seq)
                    
                # --- Student Forward Pass ---
                student_logits = student(imgs, captions)
                student_features = student.get_image_features(imgs)

                # --- LOSS CALCULATION (SIMPLIFIED) ---
                # 1. Standard Cross-Entropy Loss (learning from data)
                targets_flat = captions[1:, :].permute(1, 0).reshape(-1)
                loss_ce = ce_criterion(student_logits.reshape(-1, student_logits.size(-1)), targets_flat)

                # 2. Feature Distillation Loss (learning to "see" like the teacher)
                teacher_features_pooled = torch.mean(teacher_features_seq, dim=1)
                student_features_for_comparison = feature_projection(student_features)
                loss_features = feature_loss_criterion(student_features_for_comparison, teacher_features_pooled)
                
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
            
            loop.set_postfix({'loss': loss.item(), 'ce_loss': loss_ce.item(), 'feat_loss': loss_features.item()})
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        avg_ce_loss = epoch_ce_loss / len(train_loader)
        avg_feat_loss = epoch_feat_loss / len(train_loader)
        
        print(f"Epoch {epoch+1} Summary -> Total Loss: {avg_loss:.4f}, CE Loss: {avg_ce_loss:.4f}, Feature Loss: {avg_feat_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'model_state_dict': student.state_dict(), 'loss': avg_loss}, 'saved_models/best_student_model_lstm.pth')
            print(f"Best LSTM student model saved with loss: {avg_loss:.4f}")

    print(f"Simplified Distillation training completed! Best loss: {best_loss:.4f}")

if __name__ == "__main__":
    train_student_with_kd()