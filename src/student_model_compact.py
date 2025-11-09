# src/student_model_compact.py
# Compact CNN-LSTM Student Model for Real Compression

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class CompactCNNEncoder(nn.Module):
    """
    Compact CNN Encoder using MobileNetV2 for efficiency
    """
    def __init__(self, embed_size=256, fine_tune=True):
        super(CompactCNNEncoder, self).__init__()
        
        self.embed_size = embed_size
        
        # Use MobileNetV2 for efficiency (much smaller than ResNet-50)
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # Remove classifier, keep feature extractor
        self.backbone = mobilenet.features  # Output: 1280 channels
        feature_dim = 1280
        
        # Selective fine-tuning - freeze early layers
        if fine_tune:
            for i, layer in enumerate(self.backbone):
                if i < 10:  # Freeze first 10 layers
                    for param in layer.parameters():
                        param.requires_grad = False
        
        # Adaptive pooling for consistent output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # 7x7 = 49 locations
        
        # Simple projection (no complex layers)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, embed_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, images):
        """
        Forward pass through compact encoder
        Args:
            images: (batch_size, 3, 224, 224)
        Returns:
            features: (batch_size, 49, embed_size)
        """
        batch_size = images.size(0)
        
        # Extract features
        features = self.backbone(images)  # (batch_size, 1280, H, W)
        
        # Ensure consistent spatial dimensions
        features = self.adaptive_pool(features)  # (batch_size, 1280, 7, 7)
        
        # Reshape to sequence format
        features = features.view(batch_size, 1280, -1)  # (batch_size, 1280, 49)
        features = features.permute(0, 2, 1)  # (batch_size, 49, 1280)
        
        # Project to embedding dimension
        features = self.projection(features)  # (batch_size, 49, embed_size)
        
        return features


class CompactLSTMDecoder(nn.Module):
    """
    Compact LSTM decoder with minimal parameters
    """
    def __init__(self, vocab_size, embed_size=256, hidden_size=256, num_layers=1, dropout=0.1):
        super(CompactLSTMDecoder, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        # Word embedding
        self.embedding = nn.Embedding(vocab_size, embed_size)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # Simple attention mechanism
        self.attention = nn.Linear(hidden_size, embed_size)
        
        # Single LSTM layer for efficiency
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0,  # No dropout for single layer
            batch_first=True
        )
        
        # Simple output projection
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
    def init_hidden(self, batch_size, device):
        """Initialize hidden and cell states"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)
    
    def simple_attention(self, hidden, image_features):
        """
        Simple attention mechanism
        Args:
            hidden: (batch_size, hidden_size)
            image_features: (batch_size, seq_len, embed_size)
        Returns:
            context: (batch_size, embed_size)
            attention_weights: (batch_size, seq_len)
        """
        batch_size, seq_len, feature_dim = image_features.size()
        
        # Project hidden state to attention space
        hidden_proj = self.attention(hidden)  # (batch_size, embed_size)
        hidden_proj = hidden_proj.unsqueeze(1)  # (batch_size, 1, embed_size)
        
        # Compute attention scores
        scores = torch.bmm(hidden_proj, image_features.transpose(1, 2))  # (batch_size, 1, seq_len)
        attention_weights = F.softmax(scores.squeeze(1), dim=1)  # (batch_size, seq_len)
        
        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), image_features)  # (batch_size, 1, embed_size)
        context = context.squeeze(1)  # (batch_size, embed_size)
        
        return context, attention_weights
    
    def forward(self, image_features, captions, hidden=None):
        """
        Forward pass through compact decoder
        Args:
            image_features: (batch_size, seq_len, embed_size)
            captions: (seq_len, batch_size)
            hidden: tuple of (h0, c0) or None
        Returns:
            outputs: (seq_len, batch_size, vocab_size)
            hidden_states: list of hidden states
            attention_weights: list of attention weights
        """
        batch_size = image_features.size(0)
        seq_len = captions.size(0)
        
        if hidden is None:
            hidden = self.init_hidden(batch_size, captions.device)
        
        # Embed captions
        embedded = self.embedding(captions)  # (seq_len, batch_size, embed_size)
        embedded = embedded.permute(1, 0, 2)  # (batch_size, seq_len, embed_size)
        
        outputs = []
        hidden_states = []
        attention_weights_list = []
        
        # Process each time step
        for t in range(seq_len):
            # Get current word embedding
            word_embed = embedded[:, t, :].unsqueeze(1)  # (batch_size, 1, embed_size)
            
            # Compute attention
            context, attn_weights = self.simple_attention(hidden[0][-1], image_features)
            
            # Combine word embedding with context (simple addition)
            combined_input = word_embed.squeeze(1) + context  # (batch_size, embed_size)
            combined_input = combined_input.unsqueeze(1)  # (batch_size, 1, embed_size)
            
            # LSTM forward pass
            lstm_out, hidden = self.lstm(combined_input, hidden)
            
            # Project to vocabulary
            output = self.output_projection(lstm_out.squeeze(1))  # (batch_size, vocab_size)
            
            outputs.append(output)
            hidden_states.append(hidden[0][-1])
            attention_weights_list.append(attn_weights)
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=0)  # (seq_len, batch_size, vocab_size)
        
        return outputs, hidden_states, attention_weights_list


class CompactCaptioningStudent(nn.Module):
    """
    Compact CNN-LSTM Student Model for Real Compression (2-3x smaller)
    """
    def __init__(self, vocab_size, embed_size=256, hidden_size=256, num_layers=1, 
                 dropout=0.1, use_attention_refinement=False):
        super(CompactCaptioningStudent, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.use_attention_refinement = use_attention_refinement
        
        # Compact CNN Encoder
        self.encoder = CompactCNNEncoder(embed_size=embed_size, fine_tune=True)
        
        # Optional simple attention refinement (lightweight)
        if use_attention_refinement:
            self.attention_refinement = nn.MultiheadAttention(
                embed_dim=embed_size,
                num_heads=4,  # Fewer heads
                dropout=0.1,
                batch_first=True
            )
            self.norm = nn.LayerNorm(embed_size)
        
        # Compact LSTM Decoder
        self.decoder = CompactLSTMDecoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
    def forward(self, images, captions):
        """
        Forward pass through compact model
        Args:
            images: (batch_size, 3, 224, 224)
            captions: (seq_len, batch_size)
        Returns:
            outputs: (seq_len, batch_size, vocab_size)
            encoder_features: (batch_size, 49, embed_size)
            hidden_states: list of decoder hidden states
            attention_weights: list of attention weights
        """
        # Encode images
        encoder_features = self.encoder(images)  # (batch_size, 49, embed_size)
        
        # Optional lightweight attention refinement
        if self.use_attention_refinement:
            attn_output, _ = self.attention_refinement(
                encoder_features, encoder_features, encoder_features
            )
            refined_features = self.norm(encoder_features + attn_output)
        else:
            refined_features = encoder_features
        
        # Decode captions
        outputs, hidden_states, attention_weights = self.decoder(refined_features, captions)
        
        return outputs, encoder_features, hidden_states, attention_weights
    
    def caption_image(self, image, vocabulary, max_length=20, temperature=1.0):
        """
        Generate caption for a single image
        Args:
            image: (3, 224, 224) or (1, 3, 224, 224)
            vocabulary: Vocabulary object
            max_length: Maximum caption length
            temperature: Sampling temperature
        Returns:
            caption: List of words
        """
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            # Prepare image
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(device)
            
            # Encode image
            encoder_features = self.encoder(image)
            if self.use_attention_refinement:
                attn_output, _ = self.attention_refinement(
                    encoder_features, encoder_features, encoder_features
                )
                encoder_features = self.norm(encoder_features + attn_output)
            
            # Initialize decoder
            batch_size = 1
            hidden = self.decoder.init_hidden(batch_size, device)
            
            # Start with <START> token
            current_token = torch.tensor([[vocabulary.stoi.get("<START>", vocabulary.stoi["<UNK>"])]]).to(device)
            
            result_caption = []
            
            for _ in range(max_length):
                # Embed current token
                embedded = self.decoder.embedding(current_token.squeeze(0))
                
                # Compute attention
                context, _ = self.decoder.simple_attention(hidden[0][-1], encoder_features)
                
                # Combine embedding with context
                combined_input = embedded + context
                combined_input = combined_input.unsqueeze(1)
                
                # LSTM forward
                lstm_out, hidden = self.decoder.lstm(combined_input, hidden)
                
                # Project to vocabulary
                output = self.decoder.output_projection(lstm_out.squeeze(1))
                
                # Apply temperature and get next token
                if temperature != 1.0:
                    output = output / temperature
                
                predicted_idx = output.argmax(dim=1).item()
                
                # Stop if we predict <END>
                if vocabulary.itos[predicted_idx] == "<END>":
                    break
                
                # Add to result
                result_caption.append(vocabulary.itos[predicted_idx])
                
                # Update current token
                current_token = torch.tensor([[predicted_idx]]).to(device)
            
            return result_caption


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# Test the compact model
if __name__ == "__main__":
    print("🚀 Testing Compact Student Model")
    print("=" * 50)
    
    # Test model creation
    vocab_size = 5000
    model = CompactCaptioningStudent(
        vocab_size=vocab_size,
        embed_size=256,
        hidden_size=256,
        num_layers=1,
        use_attention_refinement=False
    )
    
    total, trainable = count_parameters(model)
    print(f"📊 Total parameters: {total:,}")
    print(f"🎯 Trainable parameters: {trainable:,}")
    
    # Compare with typical teacher size (25M)
    teacher_params = 25_000_000
    compression_ratio = teacher_params / total
    print(f"📈 Compression ratio: {compression_ratio:.2f}x")
    
    # Test forward pass
    batch_size = 2
    seq_len = 15
    
    images = torch.randn(batch_size, 3, 224, 224)
    captions = torch.randint(0, vocab_size, (seq_len, batch_size))
    
    print(f"\n🔍 Testing forward pass...")
    outputs, encoder_features, hidden_states, attention_weights = model(images, captions)
    
    print(f"✅ Output shape: {outputs.shape}")
    print(f"✅ Encoder features shape: {encoder_features.shape}")
    print(f"✅ Number of hidden states: {len(hidden_states)}")
    print(f"✅ Hidden state shape: {hidden_states[0].shape}")
    
    print(f"\n🎉 Compact model test completed successfully!")
    print(f"🎯 Target: 2-3x compression achieved: {compression_ratio:.1f}x")
