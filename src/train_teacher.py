import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from tqdm import tqdm
import os
import json
from collections import defaultdict

from models import CaptioningTeacher
from data_loader import get_loader

# For mixed precision training
from torch.amp import GradScaler, autocast

def calculate_bleu_score(predicted_caption, target_caption):
    """Simple BLEU-1 score calculation for monitoring"""
    pred_words = set(predicted_caption.lower().split())
    target_words = set(target_caption.lower().split())
    if len(target_words) == 0:
        return 0.0
    return len(pred_words.intersection(target_words)) / len(target_words)

def validate_model(model, data_loader, criterion, device, vocab):
    """Validation function to monitor overfitting"""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for imgs, captions in data_loader:
            imgs = imgs.to(device)
            captions = captions.to(device)
            
            captions_input = captions[:-1, :]
            captions_target = captions[1:, :]
            
            with autocast('cuda'):
                outputs = model(imgs, captions_input)
                loss = criterion(
                    outputs.reshape(-1, outputs.shape[2]),
                    captions_target.reshape(-1)
                )
            
            total_loss += loss.item() * imgs.size(0)
            total_samples += imgs.size(0)
    
    model.train()
    return total_loss / total_samples

def train():
    # --- Hyperparameters ---
    learning_rate = 1e-4  # Reduced learning rate
    batch_size = 12  # Slightly increased
    accumulation_steps = 3  # Reduced accumulation
    num_epochs = 25  # More epochs with better scheduling
    embed_size = 512  # Reduced embedding size
    num_heads = 8  # Reduced attention heads
    num_decoder_layers = 4  # Reduced layers
    dropout = 0.15  # Increased dropout
    num_workers = 4
    
    # Early stopping parameters
    patience = 5
    best_val_loss = float('inf')
    patience_counter = 0

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Data augmentation for training
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

    # Load training data
    train_loader, train_dataset = get_loader(
        root_folder="data/flickr8k/Images",
        annotation_file="data/flickr8k/captions_clean.csv",
        transform=train_transform,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    
    # Create validation split (use a subset for validation)
    val_loader, _ = get_loader(
        root_folder="data/flickr8k/Images",
        annotation_file="data/flickr8k/captions_clean.csv",
        transform=val_transform,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    # --- Initialize Model, Loss, Optimizer ---
    vocab_size = len(train_dataset.vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    model = CaptioningTeacher(
        vocab_size=vocab_size,
        embed_size=embed_size,
        num_heads=num_heads,
        num_decoder_layers=num_decoder_layers,
        dropout=dropout
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss with label smoothing
    class LabelSmoothingLoss(nn.Module):
        def __init__(self, classes, smoothing=0.1, ignore_index=-1):
            super(LabelSmoothingLoss, self).__init__()
            self.confidence = 1.0 - smoothing
            self.smoothing = smoothing
            self.classes = classes
            self.ignore_index = ignore_index
            
        def forward(self, pred, target):
            pred = pred.log_softmax(dim=-1)
            with torch.no_grad():
                true_dist = torch.zeros_like(pred)
                true_dist.fill_(self.smoothing / (self.classes - 1))
                true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
                true_dist[:, self.ignore_index] = 0
                mask = torch.nonzero(target == self.ignore_index)
                if mask.dim() > 0:
                    true_dist.index_fill_(0, mask.squeeze(), 0.0)
            return torch.mean(torch.sum(-true_dist * pred, dim=1))

    criterion = LabelSmoothingLoss(
        classes=vocab_size, 
        smoothing=0.1, 
        ignore_index=train_dataset.vocab.stoi["<PAD>"]
    )
    
    # Separate learning rates for encoder and decoder
    encoder_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'encoder' in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': learning_rate * 0.1},  # Lower LR for encoder
        {'params': decoder_params, 'lr': learning_rate}
    ], weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    
    # Initialize GradScaler for mixed precision
    scaler = GradScaler('cuda')

    # Training history
    train_losses = []
    val_losses = []

    # --- Training Loop ---
    print("Starting improved training for the Teacher model...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        for idx, (imgs, captions) in enumerate(loop):
            imgs = imgs.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)

            captions_input = captions[:-1, :]
            captions_target = captions[1:, :]

            # Forward pass with mixed precision
            with autocast('cuda'):
                outputs = model(imgs, captions_input)
                loss = criterion(
                    outputs.reshape(-1, outputs.shape[2]),
                    captions_target.reshape(-1)
                )
                loss = loss / accumulation_steps

            # Backward pass
            scaler.scale(loss).backward()
            
            if (idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                scheduler.step(epoch + idx / len(train_loader))

            epoch_loss += loss.item() * accumulation_steps
            num_batches += 1
            
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix({
                'loss': loss.item() * accumulation_steps,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Calculate average training loss
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        if epoch % 2 == 0:  # Validate every 2 epochs
            val_loss = validate_model(model, val_loader, criterion, device, train_dataset.vocab)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                if not os.path.exists('saved_models'):
                    os.makedirs('saved_models')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'vocab_size': vocab_size
                }, 'saved_models/best_teacher_model.pth')
                print(f"New best model saved with validation loss: {val_loss:.4f}")
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
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'vocab_size': vocab_size
    }, 'saved_models/final_teacher_model.pth')
    
    # Save training history
    with open('saved_models/training_history.json', 'w') as f:
        json.dump({
            'train_losses': train_losses,
            'val_losses': val_losses
        }, f)
    
    print(f"Training completed. Final model saved.")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()