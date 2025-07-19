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
    def __init__(self, vocab_size, embed_size=768, num_heads=12, num_decoder_layers=6, dropout=0.1):
        super(CaptioningTeacher, self).__init__()
        
        # --- ENCODER ---
        # We use a pre-trained Vision Transformer (ViT)
        # We'll use the 'vit_base_patch16_224_in21k' model, which has a hidden size of 768 to match GPT-2's base.
        self.encoder = timm.create_model('vit_base_patch16_224_in21k', pretrained=True, num_classes=0) # num_classes=0 removes the classifier head
        
        # Freeze the encoder parameters initially. We are feature-extracting.
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # --- DECODER ---
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(d_model=embed_size, dropout=dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, images, captions):
        # --- ENCODER FORWARD ---
        # Encoder output needs to be in shape (sequence_length, batch_size, embed_size) for the decoder
        # ViT output is (batch_size, num_patches, embed_size), so we permute it
        image_features = self.encoder.forward_features(images) # Shape: (batch_size, 197, 768)
        image_features = image_features.permute(1, 0, 2)       # Shape: (197, batch_size, 768)

        # --- DECODER FORWARD ---
        # captions shape: (caption_len, batch_size)
        embed_captions = self.embedding(captions)
        embed_captions = self.pos_encoder(embed_captions)
        
        # Create a target mask to prevent the decoder from "cheating" by looking at future words
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(captions.size(0)).to(images.device)
        
        # The decoder takes the image features as "memory"
        decoder_output = self.decoder(
            tgt=embed_captions, 
            memory=image_features, 
            tgt_mask=tgt_mask
        )
        
        output = self.fc_out(decoder_output) # Shape: (caption_len, batch_size, vocab_size)
        return output
    
    def caption_image(self, image, vocabulary, max_length=50):
        """
        Greedy search to generate a caption for an image.
        """
        self.eval() # Set model to evaluation mode
        result_caption = []

        with torch.no_grad():
            # Add batch dimension and get features
            image = image.unsqueeze(0)
            image_features = self.encoder.forward_features(image)
            image_features = image_features.permute(1, 0, 2)

            # Start with the <START> token
            states = (image_features)
            inputs = torch.tensor([vocabulary.stoi["<START>"]]).unsqueeze(1).to(image.device)

            for _ in range(max_length):
                embed_captions = self.embedding(inputs)
                embed_captions = self.pos_encoder(embed_captions)
                
                output = self.decoder(embed_captions, states)
                output = self.fc_out(output) # Shape: (seq_len, 1, vocab_size)
                
                predicted_idx = output.argmax(2)[-1, :] # Get the last predicted word
                
                result_caption.append(predicted_idx.item())
                
                # Check if the predicted token is <END>
                if vocabulary.itos[predicted_idx.item()] == "<END>":
                    break
                
                # The predicted word becomes the input for the next time step
                inputs = torch.cat([inputs, predicted_idx.unsqueeze(0)], dim=0)

        self.train() # Set model back to training mode
        return [vocabulary.itos[idx] for idx in result_caption]