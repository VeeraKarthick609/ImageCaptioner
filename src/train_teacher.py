# src/train_teacher.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
import os

from models import CaptioningTeacher
from data_loader import get_loader

# NEW: For mixed precision training
from torch.cuda.amp import GradScaler, autocast

def train():
    # --- Hyperparameters ---
    # Adjusted for 4GB VRAM
    learning_rate = 3e-4
    batch_size = 8  # REDUCED: Drastically lower the batch size
    accumulation_steps = 4 # NEW: Accumulate gradients to simulate a larger batch size (8 * 4 = 32)
    num_epochs = 5
    embed_size = 768
    num_heads = 12
    num_decoder_layers = 6
    num_workers = 2 # REDUCED: Lower num_workers if you see memory-related issues during data loading

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_loader, dataset = get_loader(
        root_folder="data/flickr8k/Images",
        annotation_file="data/flickr8k/captions_clean.csv",
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers, # Using the reduced value
        shuffle=True
    )

    # --- Initialize Model, Loss, Optimizer ---
    vocab_size = len(dataset.vocab)
    model = CaptioningTeacher(
        vocab_size=vocab_size,
        embed_size=embed_size,
        num_heads=num_heads,
        num_decoder_layers=num_decoder_layers,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # NEW: Initialize GradScaler for mixed precision
    scaler = GradScaler()

    # --- Training Loop ---
    model.train()

    print("Starting memory-optimized training for the Teacher model...")
    for epoch in range(num_epochs):
        loop = tqdm(data_loader, total=len(data_loader), leave=True)
        for idx, (imgs, captions) in enumerate(loop):
            imgs = imgs.to(device)
            captions = captions.to(device)

            captions_input = captions[:-1, :]
            captions_target = captions[1:, :]

            # NEW: Use autocast for the forward pass (mixed precision)
            with autocast():
                outputs = model(imgs, captions_input)
                loss = criterion(
                    outputs.reshape(-1, outputs.shape[2]),
                    captions_target.reshape(-1)
                )
                # Normalize loss for accumulation
                loss = loss / accumulation_steps

            # NEW: Scaler scales loss. Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()
            
            # NEW: Gradient Accumulation
            # Step the optimizer only after 'accumulation_steps' batches
            if (idx + 1) % accumulation_steps == 0:
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer)
                
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called.
                # Otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)
                
                # Updates the scale for next iteration.
                scaler.update()
                
                # Zero the gradients for the next accumulation cycle
                optimizer.zero_grad()

            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            # We multiply by accumulation_steps to show the "actual" loss, not the scaled down one
            loop.set_postfix(loss=loss.item() * accumulation_steps)
    
    # --- Save the Model ---
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    
    model_path = 'saved_models/teacher_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\nTeacher model saved to {model_path}")


if __name__ == "__main__":
    train()