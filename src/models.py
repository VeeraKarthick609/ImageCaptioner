# src/models.py

import torch
import torch.nn as nn
import timm
import math

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal Positional Encoding.
    From PyTorch tutorial: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CaptioningTeacher(nn.Module):
    def __init__(self, vocab_size, embed_size=384, num_heads=12, num_decoder_layers=6, dropout=0.1):
        super(CaptioningTeacher, self).__init__()
        
        # --- ENCODER ---
        # Use a smaller, more efficient ViT model
        self.encoder = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
        
        # Get the actual feature dimension from the model
        encoder_dim = self.encoder.num_features  # Should be 384 for vit_small
        
        # Unfreeze some encoder layers for fine-tuning
        # Keep early layers frozen, allow later layers to adapt
        for i, (name, param) in enumerate(self.encoder.named_parameters()):
            if 'blocks.8' in name or 'blocks.9' in name or 'blocks.10' in name or 'blocks.11' in name or 'norm' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Project encoder features to decoder dimension if they don't match
        self.encoder_projection = nn.Linear(encoder_dim, embed_size) if encoder_dim != embed_size else nn.Identity()
        
        # --- DECODER ---
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Initialize embeddings with smaller variance
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        self.pos_encoder = PositionalEncoding(d_model=embed_size, dropout=dropout)
        
        # Use fewer decoder layers to reduce overfitting
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size, 
            nhead=num_heads, 
            dim_feedforward=embed_size * 2,  # Reduced from default 4x
            dropout=dropout,
            batch_first=False
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Add layer normalization before final projection
        self.pre_output_norm = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        
        # Initialize output layer with smaller weights
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, images, captions):
        # --- ENCODER FORWARD ---
        with torch.amp.autocast("cuda"):
            image_features = self.encoder.forward_features(images)  # Shape: (batch_size, 197, encoder_dim)
            image_features = self.encoder_projection(image_features)  # Shape: (batch_size, 197, embed_size)
            image_features = image_features.permute(1, 0, 2)  # Shape: (197, batch_size, embed_size)

        # --- DECODER FORWARD ---
        embed_captions = self.embedding(captions)
        embed_captions = self.pos_encoder(embed_captions)
        
        # Create causal mask
        seq_len = captions.size(0)
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=captions.device), diagonal=1).bool()
        
        # Decoder forward pass
        decoder_output = self.decoder(
            tgt=embed_captions, 
            memory=image_features, 
            tgt_mask=tgt_mask
        )
        
        # Apply normalization and dropout before final projection
        decoder_output = self.pre_output_norm(decoder_output)
        decoder_output = self.dropout(decoder_output)
        output = self.fc_out(decoder_output)
        
        return output
    
    def caption_image(self, image, vocabulary, max_length=20):  # Reduced max length
        """
        Generate caption using beam search for better quality
        """
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            # Get image features
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(device)
            
            image_features = self.encoder.forward_features(image)
            image_features = self.encoder_projection(image_features)
            image_features = image_features.permute(1, 0, 2)

            # Simple greedy decoding
            result_caption = []
            inputs = torch.tensor([[vocabulary.stoi.get("<START>", vocabulary.stoi["<UNK>"])]]).to(device).T
            
            for _ in range(max_length):
                embed_captions = self.embedding(inputs)
                embed_captions = self.pos_encoder(embed_captions)
                
                seq_len = inputs.size(0)
                tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
                
                decoder_output = self.decoder(embed_captions, image_features, tgt_mask=tgt_mask)
                decoder_output = self.pre_output_norm(decoder_output)
                output = self.fc_out(decoder_output)
                
                # Get the most likely next word
                predicted_idx = output[-1, 0, :].argmax().item()
                
                # Stop if we predict <END>
                if vocabulary.itos[predicted_idx] == "<END>":
                    break
                
                result_caption.append(predicted_idx)
                
                # Add predicted token to input sequence
                new_token = torch.tensor([[predicted_idx]]).to(device)
                inputs = torch.cat([inputs, new_token], dim=0)
        
        return [vocabulary.itos[idx] for idx in result_caption]