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

# KnowledgeDistillationLoss class remains the same
class KnowledgeDistillationLoss(nn.Module):
    # ... (no changes needed here) ...
    """Combined loss for knowledge distillation"""
    
    def __init__(self, alpha=0.7, temperature=4.0, vocab_size=None, pad_idx=None):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=pad_idx)
        
    def forward(self, student_logits, teacher_logits, targets):
        # --- FIX IS HERE ---
        # Use .reshape() instead of .view() for safety against non-contiguous tensors
        ce_loss = self.ce_loss(student_logits.reshape(-1, student_logits.size(-1)), targets)
        
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # Also use .reshape() here for consistency and safety
        student_log_probs_flat = student_soft.reshape(-1, student_soft.size(-1))
        teacher_probs_flat = teacher_soft.reshape(-1, teacher_soft.size(-1))
        
        kd_loss = F.kl_div(student_log_probs_flat, teacher_probs_flat, reduction='batchmean') * (self.temperature ** 2)
        
        total_loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
        
        return total_loss, ce_loss.item(), kd_loss.item()


# load_teacher_model function remains the same
def load_teacher_model(model_path, vocab_size, device):
    # ... (no changes needed here) ...
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
    temperature = 4.0
    # Note: alpha and beta will now be set inside the epoch loop

    # ... (device setup, data loading, model loading remains the same) ...
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
    
    teacher = load_teacher_model('saved_models/best_teacher_model.pth', vocab_size, device)
    student = CaptioningStudent(vocab_size=vocab_size, embed_size=256, hidden_size=512, num_layers=2, dropout=0.5).to(device)
    
    student_feature_dim = 2048
    teacher_feature_dim = 512
    feature_projection = nn.Linear(student_feature_dim, teacher_feature_dim).to(device)
    
    # --- Loss and Optimizer ---
    # We instantiate the KD loss here, but its alpha value will be ignored inside the loop
    kd_criterion = KnowledgeDistillationLoss(temperature=temperature, vocab_size=vocab_size, pad_idx=pad_idx)
    feature_loss_criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        list(student.parameters()) + list(feature_projection.parameters()), 
        lr=learning_rate, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler = GradScaler()

    # --- Training Loop with Phased Learning ---
    print("Starting Staged Knowledge Distillation for CNN-LSTM Student...")
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        student.train()
        feature_projection.train()
        
        # --- NEW: SET ALPHA AND BETA BASED ON THE CURRENT EPOCH ---
        if epoch < 10:
            # Phase 1: Focus on CE loss (Grounding)
            alpha = 0.3  # Low weight for logit distillation
            beta = 0.1   # Low weight for feature distillation
            phase = 1
        elif epoch < 20:
            # Phase 2: Balanced distillation
            alpha = 0.7
            beta = 0.3
            phase = 2
        else:
            # Phase 3: Heavy distillation (Fine-tuning)
            alpha = 0.9
            beta = 0.5
            phase = 3
        
        # We need to update the alpha inside the criterion object for the logit loss
        kd_criterion.alpha = alpha

        epoch_loss = 0
        epoch_ce_loss = 0
        epoch_kd_loss = 0
        epoch_feat_loss = 0
        
        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        # Add phase info to the tqdm description
        loop.set_description(f"Phase {phase} | Epoch [{epoch+1}/{num_epochs}]")

        for imgs, captions in loop:
            imgs = imgs.to(device)
            captions = captions.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                with torch.no_grad():
                    teacher_logits = teacher(imgs, captions[:-1, :])
                    teacher_features_seq = teacher.encoder.forward_features(imgs)
                    teacher_features_seq = teacher.encoder_projection(teacher_features_seq)
                    
                student_logits = student(imgs, captions)
                student_features = student.get_image_features(imgs)
                targets_flat = captions[1:, :].permute(1, 0).reshape(-1)

                # 1. Logit Distillation Loss (uses the updated alpha)
                # The first returned value is the combined (alpha * kd + (1-alpha) * ce)
                logit_kd_loss, ce_loss, kd_loss = kd_criterion(student_logits, teacher_logits, targets_flat)

                # 2. Feature Distillation Loss
                teacher_features_pooled = torch.mean(teacher_features_seq, dim=1)
                student_features_for_comparison = feature_projection(student_features)
                loss_features = feature_loss_criterion(student_features_for_comparison, teacher_features_pooled)
                
                # 3. Combine all losses using the current phase's beta
                loss = (1 - beta) * logit_kd_loss + beta * loss_features
            
            # --- Backward Pass ---
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(feature_projection.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Update logs
            epoch_loss += loss.item()
            epoch_ce_loss += ce_loss
            epoch_kd_loss += kd_loss
            epoch_feat_loss += loss_features.item()
            
            # Add alpha and beta to the postfix to see them change
            loop.set_postfix({'loss': loss.item(), 'α': alpha, 'β': beta})
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        avg_ce_loss = epoch_ce_loss / len(train_loader)
        avg_kd_loss = epoch_kd_loss / len(train_loader)
        avg_feat_loss = epoch_feat_loss / len(train_loader)
        
        print(f"Epoch {epoch+1} (Phase {phase}) Summary -> Loss: {avg_loss:.4f}, CE: {avg_ce_loss:.4f}, KD: {avg_kd_loss:.4f}, Feat: {avg_feat_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            # It's good practice to include epoch and phase info in the saved file
            torch.save({
                'epoch': epoch,
                'phase': phase,
                'model_state_dict': student.state_dict(),
                'loss': avg_loss,
            }, 'saved_models/best_student_model_lstm_staged.pth')
            print(f"Best LSTM student model saved with loss: {avg_loss:.4f}")

    print(f"Staged Knowledge Distillation completed! Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    train_student_with_kd()