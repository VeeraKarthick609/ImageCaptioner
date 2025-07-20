# src/student_model.py

import torch
import torch.nn as nn
import timm

class CaptioningStudent(nn.Module):
    """Smaller student model for knowledge distillation"""
    
    def __init__(self, vocab_size, embed_size=256, num_heads=4, num_decoder_layers=2, dropout=0.1):
        super(CaptioningStudent, self).__init__()
        
        # Much smaller encoder - use a lightweight model
        self.encoder = timm.create_model('mobilevit_xxs', pretrained=True, num_classes=0)
        # Alternative: 'efficientnet_b0', 'resnet18', 'mobilenetv3_small_100'
        
        encoder_dim = self.encoder.num_features
        self.encoder_projection = nn.Linear(encoder_dim, embed_size)
        
        # Freeze encoder for faster training (optional)
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Lightweight decoder
        self.embedding = nn.Embedding(vocab_size, embed_size)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        from models import PositionalEncoding
        self.pos_encoder = PositionalEncoding(d_model=embed_size, dropout=dropout)
        
        # Fewer layers and smaller dimensions
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size, 
            nhead=num_heads, 
            dim_feedforward=embed_size,  # Much smaller
            dropout=dropout,
            batch_first=False
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.pre_output_norm = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, images, captions):
        # Encoder forward
        image_features = self.encoder.forward_features(images)
        
        # Handle different output shapes from different encoders
        if len(image_features.shape) == 4:  # CNN features (B, C, H, W)
            B, C, H, W = image_features.shape
            image_features = image_features.view(B, C, H*W).permute(2, 0, 1)  # (H*W, B, C)
        else:  # ViT-like features (B, N, C)
            image_features = image_features.permute(1, 0, 2)  # (N, B, C)
        
        image_features = self.encoder_projection(image_features)
        
        # Decoder forward
        embed_captions = self.embedding(captions)
        embed_captions = self.pos_encoder(embed_captions)
        
        seq_len = captions.size(0)
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=captions.device), diagonal=1).bool()
        
        decoder_output = self.decoder(
            tgt=embed_captions, 
            memory=image_features, 
            tgt_mask=tgt_mask
        )
        
        decoder_output = self.pre_output_norm(decoder_output)
        decoder_output = self.dropout(decoder_output)
        output = self.fc_out(decoder_output)
        
        return output