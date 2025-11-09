# src/student_model.py

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    """
    CNN Encoder using ResNet backbone for feature extraction
    """
    def __init__(self, embed_size=256, fine_tune=True):
        super(CNNEncoder, self).__init__()
        
        # Use ResNet-50 as backbone (smaller than ViT)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Remove the final classification layers
        modules = list(resnet.children())[:-2]  # Remove avgpool and fc
        self.resnet = nn.Sequential(*modules)
        
        # Fine-tuning: freeze early layers, unfreeze later layers
        if fine_tune:
            for i, child in enumerate(self.resnet.children()):
                if i < 6:  # Freeze first 6 layers (conv1, bn1, relu, maxpool, layer1, layer2)
                    for param in child.parameters():
                        param.requires_grad = False
                else:  # Unfreeze layer3 and layer4
                    for param in child.parameters():
                        param.requires_grad = True
        
        # ResNet-50 outputs 2048 channels, spatial size depends on input
        # For 224x224 input, we get 7x7 spatial dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # Ensure 7x7 output
        
        # Project to embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(2048, embed_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(embed_size)
        )
        
        self.embed_size = embed_size
        
    def forward(self, images):
        """
        Forward pass through CNN encoder
        Args:
            images: (batch_size, 3, 224, 224)
        Returns:
            features: (batch_size, 49, embed_size) - 49 = 7*7 spatial locations
        """
        batch_size = images.size(0)
        
        # Extract features through ResNet
        features = self.resnet(images)  # (batch_size, 2048, 7, 7)
        
        # Ensure consistent spatial dimensions
        features = self.adaptive_pool(features)  # (batch_size, 2048, 7, 7)
        
        # Reshape to sequence format
        features = features.view(batch_size, 2048, -1)  # (batch_size, 2048, 49)
        features = features.permute(0, 2, 1)  # (batch_size, 49, 2048)
        
        # Project to embedding dimension
        features = self.projection(features)  # (batch_size, 49, embed_size)
        
        return features


class AttentionRefinement(nn.Module):
    """
    Pre-projection attention mechanism to refine CNN features
    """
    def __init__(self, embed_size, num_heads=4):
        super(AttentionRefinement, self).__init__()
        
        self.embed_size = embed_size
        self.num_heads = num_heads
        
        # Multi-head attention for feature refinement
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_size * 2, embed_size)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
    def forward(self, features):
        """
        Apply self-attention to refine features
        Args:
            features: (batch_size, seq_len, embed_size)
        Returns:
            refined_features: (batch_size, seq_len, embed_size)
        """
        # Self-attention
        attn_output, _ = self.attention(features, features, features)
        features = self.norm1(features + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(features)
        features = self.norm2(features + ffn_output)
        
        return features


class LSTMDecoder(nn.Module):
    """
    LSTM-based decoder for caption generation
    """
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_layers=2, dropout=0.2):
        super(LSTMDecoder, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        # Word embedding
        self.embedding = nn.Embedding(vocab_size, embed_size)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # Attention mechanism for image features
        self.attention = nn.Linear(hidden_size + embed_size, embed_size)
        self.attention_combine = nn.Linear(embed_size * 2, embed_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, embed_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size, vocab_size)
        )
        
        # Initialize LSTM weights
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
    
    def attention_mechanism(self, hidden, image_features):
        """
        Compute attention over image features
        Args:
            hidden: (batch_size, hidden_size) - current hidden state
            image_features: (batch_size, seq_len, embed_size) - CNN features
        Returns:
            context: (batch_size, embed_size) - attended features
            attention_weights: (batch_size, seq_len) - attention weights
        """
        batch_size, seq_len, feature_dim = image_features.size()
        
        # Expand hidden state to match image features
        hidden_expanded = hidden.unsqueeze(1).expand(batch_size, seq_len, self.hidden_size)
        
        # Concatenate hidden state with image features
        combined = torch.cat([hidden_expanded, image_features], dim=2)
        
        # Compute attention scores
        attention_scores = self.attention(combined)  # (batch_size, seq_len, embed_size)
        attention_scores = torch.tanh(attention_scores)
        attention_scores = torch.sum(attention_scores, dim=2)  # (batch_size, seq_len)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len)
        
        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), image_features)  # (batch_size, 1, embed_size)
        context = context.squeeze(1)  # (batch_size, embed_size)
        
        return context, attention_weights
    
    def forward(self, image_features, captions, hidden=None):
        """
        Forward pass through LSTM decoder
        Args:
            image_features: (batch_size, seq_len, embed_size) - CNN features
            captions: (seq_len, batch_size) - caption tokens
            hidden: tuple of (h0, c0) or None
        Returns:
            outputs: (seq_len, batch_size, vocab_size) - logits
            hidden_states: list of hidden states for each time step
            attention_weights: list of attention weights for each time step
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
            
            # Compute attention over image features using current hidden state
            context, attn_weights = self.attention_mechanism(hidden[0][-1], image_features)
            
            # Combine word embedding with attended image features
            combined_input = torch.cat([word_embed.squeeze(1), context], dim=1)  # (batch_size, embed_size * 2)
            combined_input = self.attention_combine(combined_input).unsqueeze(1)  # (batch_size, 1, embed_size)
            
            # LSTM forward pass
            lstm_out, hidden = self.lstm(combined_input, hidden)  # lstm_out: (batch_size, 1, hidden_size)
            
            # Project to vocabulary
            output = self.output_projection(lstm_out.squeeze(1))  # (batch_size, vocab_size)
            
            outputs.append(output)
            hidden_states.append(hidden[0][-1])  # Store last layer hidden state
            attention_weights_list.append(attn_weights)
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=0)  # (seq_len, batch_size, vocab_size)
        
        return outputs, hidden_states, attention_weights_list


class CaptioningStudent(nn.Module):
    """
    Complete CNN-LSTM Student Model for Image Captioning
    """
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_layers=2, 
                 dropout=0.2, use_attention_refinement=True):
        super(CaptioningStudent, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        
        # CNN Encoder
        self.encoder = CNNEncoder(embed_size=embed_size, fine_tune=True)
        
        # Optional attention refinement
        self.use_attention_refinement = use_attention_refinement
        if use_attention_refinement:
            self.attention_refinement = AttentionRefinement(embed_size=embed_size)
        
        # LSTM Decoder
        self.decoder = LSTMDecoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
    def forward(self, images, captions):
        """
        Forward pass through the complete model
        Args:
            images: (batch_size, 3, 224, 224)
            captions: (seq_len, batch_size)
        Returns:
            outputs: (seq_len, batch_size, vocab_size)
            encoder_features: (batch_size, 49, embed_size) - for KD
            hidden_states: list of decoder hidden states - for KD
            attention_weights: list of attention weights
        """
        # Encode images
        encoder_features = self.encoder(images)  # (batch_size, 49, embed_size)
        
        # Optional attention refinement
        if self.use_attention_refinement:
            refined_features = self.attention_refinement(encoder_features)
        else:
            refined_features = encoder_features
        
        # Decode captions
        outputs, hidden_states, attention_weights = self.decoder(refined_features, captions)
        
        return outputs, encoder_features, hidden_states, attention_weights
    
    def caption_image(self, image, vocabulary, max_length=20, temperature=1.0):
        """
        Generate caption for a single image using greedy decoding
        Args:
            image: (3, 224, 224) or (1, 3, 224, 224)
            vocabulary: Vocabulary object
            max_length: Maximum caption length
            temperature: Sampling temperature (1.0 = no change)
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
            
            # Initialize decoder
            batch_size = 1
            hidden = self.decoder.init_hidden(batch_size, device)
            
            # Start with <START> token
            current_token = torch.tensor([[vocabulary.stoi.get("<START>", vocabulary.stoi["<UNK>"])]]).to(device)
            
            result_caption = []
            
            for _ in range(max_length):
                # Embed current token
                embedded = self.decoder.embedding(current_token.squeeze(0))  # (1, embed_size)
                
                # Compute attention
                context, _ = self.decoder.attention_mechanism(hidden[0][-1], encoder_features)
                
                # Combine embedding with context
                combined_input = torch.cat([embedded, context], dim=1)
                combined_input = self.decoder.attention_combine(combined_input).unsqueeze(1)
                
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


# Test the model
if __name__ == "__main__":
    # Test model creation
    vocab_size = 5000
    model = CaptioningStudent(vocab_size=vocab_size)
    
    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 15
    
    images = torch.randn(batch_size, 3, 224, 224)
    captions = torch.randint(0, vocab_size, (seq_len, batch_size))
    
    outputs, encoder_features, hidden_states, attention_weights = model(images, captions)
    
    print(f"Output shape: {outputs.shape}")
    print(f"Encoder features shape: {encoder_features.shape}")
    print(f"Number of hidden states: {len(hidden_states)}")
    print(f"Hidden state shape: {hidden_states[0].shape}")
    print(f"Number of attention weights: {len(attention_weights)}")
    print(f"Attention weights shape: {attention_weights[0].shape}")
