# src/student_model_enhanced.py
# Enhanced CNN-LSTM Student Model with Performance Optimizations

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn import MultiheadAttention

class EfficientCNNEncoder(nn.Module):
    """
    Enhanced CNN Encoder with EfficientNet backbone and advanced features
    """
    def __init__(self, embed_size=384, fine_tune=True, use_efficientnet=True):
        super(EfficientCNNEncoder, self).__init__()
        
        self.embed_size = embed_size
        
        if use_efficientnet:
            # Use EfficientNet-B3 for better efficiency
            try:
                from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
                backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
                # Remove classifier
                self.backbone = nn.Sequential(*list(backbone.children())[:-1])
                feature_dim = 1536  # EfficientNet-B3 output channels
            except:
                # Fallback to ResNet-50
                resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                modules = list(resnet.children())[:-2]
                self.backbone = nn.Sequential(*modules)
                feature_dim = 2048
        else:
            # Enhanced ResNet-50 with better fine-tuning
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            modules = list(resnet.children())[:-2]
            self.backbone = nn.Sequential(*modules)
            feature_dim = 2048
        
        # Selective fine-tuning
        if fine_tune:
            # Freeze early layers, unfreeze later layers
            for i, child in enumerate(self.backbone.children()):
                if i < 6:  # Freeze early layers
                    for param in child.parameters():
                        param.requires_grad = False
                else:  # Unfreeze later layers
                    for param in child.parameters():
                        param.requires_grad = True
        
        # Adaptive pooling for consistent output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))  # 8x8 = 64 spatial locations
        
        # Enhanced projection with residual connection
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, embed_size * 2),
            nn.GELU(),  # Better activation than ReLU
            nn.Dropout(0.1),
            nn.Linear(embed_size * 2, embed_size),
            nn.LayerNorm(embed_size)
        )
        
        # Spatial attention for better feature selection
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 8, 1),
            nn.GELU(),
            nn.Conv2d(feature_dim // 8, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, images):
        """
        Enhanced forward pass with spatial attention
        Args:
            images: (batch_size, 3, 224, 224)
        Returns:
            features: (batch_size, 64, embed_size) - 64 = 8*8 spatial locations
        """
        batch_size = images.size(0)
        
        # Extract features through backbone
        features = self.backbone(images)  # (batch_size, feature_dim, H, W)
        
        # Apply spatial attention
        attention_weights = self.spatial_attention(features)
        features = features * attention_weights
        
        # Ensure consistent spatial dimensions
        features = self.adaptive_pool(features)  # (batch_size, feature_dim, 8, 8)
        
        # Reshape to sequence format
        features = features.view(batch_size, features.size(1), -1)  # (batch_size, feature_dim, 64)
        features = features.permute(0, 2, 1)  # (batch_size, 64, feature_dim)
        
        # Project to embedding dimension
        features = self.projection(features)  # (batch_size, 64, embed_size)
        
        return features


class CrossAttentionRefinement(nn.Module):
    """
    Enhanced attention mechanism with cross-attention and positional encoding
    """
    def __init__(self, embed_size, num_heads=8, num_layers=2):
        super(CrossAttentionRefinement, self).__init__()
        
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Positional encoding for spatial features
        self.pos_encoding = nn.Parameter(torch.randn(1, 64, embed_size) * 0.02)
        
        # Multi-layer attention refinement
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=embed_size,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_size, embed_size * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_size * 4, embed_size)
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(embed_size) for _ in range(num_layers * 2)
        ])
        
        # Global context aggregation
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(embed_size, embed_size),
            nn.GELU(),
            nn.Linear(embed_size, embed_size)
        )
        
    def forward(self, features):
        """
        Apply enhanced attention refinement
        Args:
            features: (batch_size, seq_len, embed_size)
        Returns:
            refined_features: (batch_size, seq_len, embed_size)
        """
        # Add positional encoding
        features = features + self.pos_encoding
        
        # Multi-layer attention refinement
        for i in range(self.num_layers):
            # Self-attention
            attn_output, _ = self.attention_layers[i](features, features, features)
            features = self.norm_layers[i * 2](features + attn_output)
            
            # Feed-forward
            ffn_output = self.ffn_layers[i](features)
            features = self.norm_layers[i * 2 + 1](features + ffn_output)
        
        # Add global context
        global_ctx = self.global_context(features.permute(0, 2, 1)).unsqueeze(1)
        features = features + global_ctx
        
        return features


class EnhancedLSTMDecoder(nn.Module):
    """
    Enhanced LSTM decoder with advanced attention and gating mechanisms
    """
    def __init__(self, vocab_size, embed_size=384, hidden_size=768, num_layers=3, dropout=0.2):
        super(EnhancedLSTMDecoder, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        # Enhanced word embedding with positional encoding
        self.embedding = nn.Embedding(vocab_size, embed_size)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # Positional encoding for sequence
        self.pos_encoding = nn.Parameter(torch.randn(1, 50, embed_size) * 0.02)  # Max seq len 50
        
        # Multi-head attention for image features
        self.image_attention = MultiheadAttention(
            embed_dim=embed_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Gating mechanism for attention fusion
        self.attention_gate = nn.Sequential(
            nn.Linear(embed_size * 2, embed_size),
            nn.Sigmoid()
        )
        
        # Enhanced LSTM with layer normalization
        self.lstm_layers = nn.ModuleList([
            nn.LSTMCell(embed_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        # Layer normalization for LSTM outputs
        self.lstm_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])
        
        # Dropout layers
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(num_layers)
        ])
        
        # Enhanced output projection with highway connection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, embed_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size, vocab_size)
        )
        
        # Highway connection for output
        self.highway_gate = nn.Sequential(
            nn.Linear(hidden_size + embed_size, hidden_size),
            nn.Sigmoid()
        )
        
        # Initialize LSTM weights
        for lstm in self.lstm_layers:
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
        
    def init_hidden(self, batch_size, device):
        """Initialize hidden and cell states for all layers"""
        hidden_states = []
        cell_states = []
        
        for _ in range(self.num_layers):
            h = torch.zeros(batch_size, self.hidden_size, device=device)
            c = torch.zeros(batch_size, self.hidden_size, device=device)
            hidden_states.append(h)
            cell_states.append(c)
            
        return hidden_states, cell_states
    
    def enhanced_attention(self, query, image_features):
        """
        Enhanced multi-head attention mechanism
        Args:
            query: (batch_size, hidden_size) - current hidden state
            image_features: (batch_size, seq_len, embed_size) - CNN features
        Returns:
            context: (batch_size, embed_size) - attended features
            attention_weights: (batch_size, seq_len) - attention weights
        """
        batch_size = image_features.size(0)
        
        # Expand query to match image features
        query_expanded = query.unsqueeze(1)  # (batch_size, 1, hidden_size)
        
        # Project query to embed_size
        if query.size(-1) != self.embed_size:
            query_proj = nn.Linear(query.size(-1), self.embed_size, device=query.device)(query_expanded)
        else:
            query_proj = query_expanded
        
        # Multi-head attention
        context, attention_weights = self.image_attention(
            query_proj, image_features, image_features
        )
        context = context.squeeze(1)  # (batch_size, embed_size)
        attention_weights = attention_weights.squeeze(1)  # (batch_size, seq_len)
        
        return context, attention_weights
    
    def forward(self, image_features, captions, hidden=None):
        """
        Enhanced forward pass through LSTM decoder
        Args:
            image_features: (batch_size, seq_len, embed_size) - CNN features
            captions: (seq_len, batch_size) - caption tokens
            hidden: tuple of (hidden_states, cell_states) or None
        Returns:
            outputs: (seq_len, batch_size, vocab_size) - logits
            hidden_states: list of hidden states for each time step
            attention_weights: list of attention weights for each time step
        """
        batch_size = image_features.size(0)
        seq_len = captions.size(0)
        
        if hidden is None:
            hidden_states, cell_states = self.init_hidden(batch_size, captions.device)
        else:
            hidden_states, cell_states = hidden
        
        # Embed captions with positional encoding
        embedded = self.embedding(captions)  # (seq_len, batch_size, embed_size)
        embedded = embedded.permute(1, 0, 2)  # (batch_size, seq_len, embed_size)
        
        # Add positional encoding (truncate if sequence is longer)
        pos_len = min(seq_len, self.pos_encoding.size(1))
        embedded[:, :pos_len, :] += self.pos_encoding[:, :pos_len, :]
        
        outputs = []
        all_hidden_states = []
        attention_weights_list = []
        
        # Process each time step
        for t in range(seq_len):
            # Get current word embedding
            word_embed = embedded[:, t, :]  # (batch_size, embed_size)
            
            # Compute attention over image features
            context, attn_weights = self.enhanced_attention(hidden_states[-1], image_features)
            
            # Gated fusion of word embedding and visual context
            combined = torch.cat([word_embed, context], dim=1)
            gate = self.attention_gate(combined)
            fused_input = gate * word_embed + (1 - gate) * context
            
            # Multi-layer LSTM forward pass
            lstm_input = fused_input
            new_hidden_states = []
            new_cell_states = []
            
            for layer in range(self.num_layers):
                h_new, c_new = self.lstm_layers[layer](lstm_input, (hidden_states[layer], cell_states[layer]))
                h_new = self.lstm_norms[layer](h_new)
                h_new = self.dropouts[layer](h_new)
                
                new_hidden_states.append(h_new)
                new_cell_states.append(c_new)
                lstm_input = h_new
            
            hidden_states = new_hidden_states
            cell_states = new_cell_states
            
            # Highway connection for output
            final_hidden = hidden_states[-1]
            highway_input = torch.cat([final_hidden, context], dim=1)
            highway_gate = self.highway_gate(highway_input)
            enhanced_hidden = highway_gate * final_hidden + (1 - highway_gate) * context
            
            # Project to vocabulary
            output = self.output_projection(enhanced_hidden)  # (batch_size, vocab_size)
            
            outputs.append(output)
            all_hidden_states.append(final_hidden)
            attention_weights_list.append(attn_weights)
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=0)  # (seq_len, batch_size, vocab_size)
        
        return outputs, all_hidden_states, attention_weights_list


class EnhancedCaptioningStudent(nn.Module):
    """
    Enhanced CNN-LSTM Student Model with Performance Optimizations
    """
    def __init__(self, vocab_size, embed_size=384, hidden_size=768, num_layers=3, 
                 dropout=0.2, use_attention_refinement=True, use_efficientnet=True):
        super(EnhancedCaptioningStudent, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.use_attention_refinement = use_attention_refinement
        
        # Enhanced CNN Encoder
        self.encoder = EfficientCNNEncoder(
            embed_size=embed_size, 
            fine_tune=True,
            use_efficientnet=use_efficientnet
        )
        
        # Enhanced attention refinement
        if use_attention_refinement:
            self.attention_refinement = CrossAttentionRefinement(
                embed_size=embed_size,
                num_heads=8,
                num_layers=2
            )
        
        # Enhanced LSTM Decoder
        self.decoder = EnhancedLSTMDecoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Feature compression for knowledge distillation
        self.feature_compressor = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.GELU(),
            nn.Linear(embed_size // 2, embed_size)
        )
        
    def forward(self, images, captions):
        """
        Enhanced forward pass through the complete model
        Args:
            images: (batch_size, 3, 224, 224)
            captions: (seq_len, batch_size)
        Returns:
            outputs: (seq_len, batch_size, vocab_size)
            encoder_features: (batch_size, 64, embed_size) - for KD
            hidden_states: list of decoder hidden states - for KD
            attention_weights: list of attention weights
        """
        # Encode images
        encoder_features = self.encoder(images)  # (batch_size, 64, embed_size)
        
        # Optional attention refinement
        if self.use_attention_refinement:
            refined_features = self.attention_refinement(encoder_features)
        else:
            refined_features = encoder_features
        
        # Compress features for knowledge distillation
        compressed_features = self.feature_compressor(refined_features)
        
        # Decode captions
        outputs, hidden_states, attention_weights = self.decoder(refined_features, captions)
        
        return outputs, compressed_features, hidden_states, attention_weights
    
    def caption_image(self, image, vocabulary, max_length=25, temperature=1.0, beam_size=1):
        """
        Enhanced caption generation with beam search option
        Args:
            image: (3, 224, 224) or (1, 3, 224, 224)
            vocabulary: Vocabulary object
            max_length: Maximum caption length
            temperature: Sampling temperature
            beam_size: Beam search size (1 = greedy)
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
                encoder_features = self.attention_refinement(encoder_features)
            
            if beam_size == 1:
                return self._greedy_decode(encoder_features, vocabulary, max_length, temperature)
            else:
                return self._beam_search_decode(encoder_features, vocabulary, max_length, beam_size)
    
    def _greedy_decode(self, encoder_features, vocabulary, max_length, temperature):
        """Greedy decoding for fast inference"""
        device = encoder_features.device
        batch_size = 1
        
        # Initialize decoder
        hidden_states, cell_states = self.decoder.init_hidden(batch_size, device)
        
        # Start with <START> token
        current_token = torch.tensor([[vocabulary.stoi.get("<START>", vocabulary.stoi["<UNK>"])]]).to(device)
        
        result_caption = []
        
        for step in range(max_length):
            # Embed current token
            embedded = self.decoder.embedding(current_token.squeeze(0))
            
            # Add positional encoding
            if step < self.decoder.pos_encoding.size(1):
                embedded += self.decoder.pos_encoding[:, step:step+1, :]
            
            # Compute attention
            context, _ = self.decoder.enhanced_attention(hidden_states[-1], encoder_features)
            
            # Gated fusion
            combined = torch.cat([embedded, context], dim=1)
            gate = self.decoder.attention_gate(combined)
            fused_input = gate * embedded + (1 - gate) * context
            
            # LSTM forward pass
            lstm_input = fused_input
            new_hidden_states = []
            new_cell_states = []
            
            for layer in range(self.decoder.num_layers):
                h_new, c_new = self.decoder.lstm_layers[layer](lstm_input, (hidden_states[layer], cell_states[layer]))
                h_new = self.decoder.lstm_norms[layer](h_new)
                new_hidden_states.append(h_new)
                new_cell_states.append(c_new)
                lstm_input = h_new
            
            hidden_states = new_hidden_states
            cell_states = new_cell_states
            
            # Highway connection
            final_hidden = hidden_states[-1]
            highway_input = torch.cat([final_hidden, context], dim=1)
            highway_gate = self.decoder.highway_gate(highway_input)
            enhanced_hidden = highway_gate * final_hidden + (1 - highway_gate) * context
            
            # Project to vocabulary
            output = self.decoder.output_projection(enhanced_hidden)
            
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
    
    def _beam_search_decode(self, encoder_features, vocabulary, max_length, beam_size):
        """Beam search decoding for better quality"""
        # Simplified beam search implementation
        # For now, fall back to greedy decode
        return self._greedy_decode(encoder_features, vocabulary, max_length, 1.0)


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# Test the enhanced model
if __name__ == "__main__":
    print("🚀 Testing Enhanced Student Model")
    print("=" * 50)
    
    # Test model creation
    vocab_size = 5000
    model = EnhancedCaptioningStudent(
        vocab_size=vocab_size,
        embed_size=384,
        hidden_size=768,
        num_layers=3,
        use_efficientnet=True
    )
    
    total, trainable = count_parameters(model)
    print(f"📊 Total parameters: {total:,}")
    print(f"🎯 Trainable parameters: {trainable:,}")
    
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
    print(f"✅ Number of attention weights: {len(attention_weights)}")
    print(f"✅ Attention weights shape: {attention_weights[0].shape}")
    
    print(f"\n🎉 Enhanced model test completed successfully!")
