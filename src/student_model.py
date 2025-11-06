# src/improved_student_model.py

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class CaptioningStundent(nn.Module):
    """
    Improved Student Model with better architecture and attention mechanisms
    """
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_layers=2, dropout=0.3):
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            embed_size (int): Dimension of the word embeddings.
            hidden_size (int): Dimension of the LSTM hidden states.
            num_layers (int): Number of layers in the LSTM.
            dropout (float): Dropout probability.
        """
        super(CaptioningStundent, self).__init__()
        
        # --- IMPROVED CNN ENCODER ---
        # Use EfficientNet-B0 instead of ResNet-50 for better efficiency
        try:
            import timm
            self.encoder = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
            encoder_dim = self.encoder.num_features  # 1280 for EfficientNet-B0
        except ImportError:
            # Fallback to ResNet-34 (lighter than ResNet-50)
            resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            modules = list(resnet.children())[:-1]
            self.encoder = nn.Sequential(*modules)
            encoder_dim = 512  # ResNet-34 feature dimension
        
        # Adaptive pooling to ensure consistent output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature projection with batch normalization
        self.feature_projection = nn.Sequential(
            nn.Linear(encoder_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5)
        )
        
        # Separate projection for cell state initialization
        self.cell_projection = nn.Sequential(
            nn.Linear(encoder_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Freeze early layers, fine-tune later layers
        self._freeze_encoder_layers()
        
        # --- IMPROVED LSTM DECODER ---
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Initialize embeddings with Xavier uniform
        nn.init.xavier_uniform_(self.embedding.weight)
        
        # Bidirectional LSTM for better context (only for the first layer)
        self.lstm = nn.LSTM(
            embed_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # Keep unidirectional for autoregressive generation
        )
        
        # --- ATTENTION MECHANISM ---
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Attention projection
        self.attention_projection = nn.Linear(encoder_dim, hidden_size)
        
        # --- OUTPUT LAYERS ---
        # Add layer normalization and residual connections
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Output projection with intermediate layer
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, vocab_size)
        )
        
        # Initialize output layers
        self._initialize_weights()
        
        # Store dimensions for distillation
        self.encoder_dim = encoder_dim
        self.hidden_size = hidden_size

    def _freeze_encoder_layers(self):
        """Freeze early encoder layers, fine-tune later ones"""
        # Freeze first 60% of encoder parameters
        total_params = len(list(self.encoder.parameters()))
        freeze_until = int(total_params * 0.6)
        
        for i, param in enumerate(self.encoder.parameters()):
            if i < freeze_until:
                param.requires_grad = False
            else:
                param.requires_grad = True
                
    def _initialize_weights(self):
        """Initialize weights with appropriate methods"""
        for module in [self.feature_projection, self.cell_projection, self.output_projection]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def get_image_features(self, images):
        """Extract image features for distillation"""
        features = self.encoder(images)
        if features.dim() > 2:
            features = self.adaptive_pool(features)
            features = features.view(features.size(0), -1)
        return features

    def get_attention_features(self, images):
        """Get spatial features for attention mechanism"""
        # Get features before global pooling for attention
        features = self.encoder.forward_features(images) if hasattr(self.encoder, 'forward_features') else self.encoder(images)
        
        if features.dim() == 4:  # (batch, channels, height, width)
            batch_size, channels, height, width = features.shape
            # Reshape to (batch, spatial_locations, channels)
            features = features.view(batch_size, channels, -1).permute(0, 2, 1)
        elif features.dim() == 3:  # Already in correct format
            pass
        else:
            # Fallback: treat as single spatial location
            features = features.unsqueeze(1)
            
        return self.attention_projection(features)

    def forward(self, images, captions):
        """Forward pass with attention mechanism"""
        batch_size = images.size(0)
        
        # --- ENCODER FORWARD ---
        # Get global image features for LSTM initialization
        global_features = self.get_image_features(images)
        
        # Get spatial features for attention
        spatial_features = self.get_attention_features(images)
        
        # Project features for LSTM initialization
        h_0 = self.feature_projection(global_features)
        c_0 = self.cell_projection(global_features)
        
        # Expand for multiple LSTM layers
        num_layers = self.lstm.num_layers
        h_0 = h_0.unsqueeze(0).repeat(num_layers, 1, 1)
        c_0 = c_0.unsqueeze(0).repeat(num_layers, 1, 1)

        # --- DECODER FORWARD ---
        # Prepare captions - handle input format correctly
        if captions.dim() == 2 and captions.size(0) > captions.size(1):
            # Input is (seq_len, batch) - convert to (batch, seq_len)
            captions = captions.permute(1, 0)
        
        # Ensure captions batch size matches images batch size
        if captions.size(0) != batch_size:
            if captions.size(0) > batch_size:
                captions = captions[:batch_size]
            else:
                raise ValueError(f"Caption batch size {captions.size(0)} < image batch size {batch_size}")
        
        captions_input = captions[:, :-1]  # Remove last token for input
        
        # Embed captions
        embeddings = self.embedding(captions_input)
        
        # LSTM forward pass
        lstm_outputs, _ = self.lstm(embeddings, (h_0, c_0))
        
        # Apply attention mechanism
        attended_outputs, attention_weights = self.attention(
            lstm_outputs, spatial_features, spatial_features
        )
        
        # Residual connection and layer normalization
        outputs = self.layer_norm(lstm_outputs + attended_outputs)
        outputs = self.dropout(outputs)
        
        # Final projection
        outputs = self.output_projection(outputs)
        
        # Reshape for loss computation (seq_len, batch, vocab_size)
        outputs = outputs.permute(1, 0, 2)
        
        return outputs

    def generate_caption(self, image, vocab, max_length=25, device='cuda'):
        """Generate caption with attention visualization"""
        self.eval()
        
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(device)
            
            # Get image features
            global_features = self.get_image_features(image)
            spatial_features = self.get_attention_features(image)
            
            # Initialize LSTM states
            h_0 = self.feature_projection(global_features)
            c_0 = self.cell_projection(global_features)
            
            num_layers = self.lstm.num_layers
            hidden = h_0.unsqueeze(0).repeat(num_layers, 1, 1)
            cell = c_0.unsqueeze(0).repeat(num_layers, 1, 1)
            
            # Start generation
            caption = []
            input_token = torch.tensor([[vocab.stoi["<START>"]]]).to(device)
            
            attention_weights_list = []
            
            for _ in range(max_length):
                # Embed current token
                embedding = self.embedding(input_token)
                
                # LSTM step
                lstm_output, (hidden, cell) = self.lstm(embedding, (hidden, cell))
                
                # Attention step
                attended_output, attention_weights = self.attention(
                    lstm_output, spatial_features, spatial_features
                )
                attention_weights_list.append(attention_weights.cpu())
                
                # Combine and project
                combined = self.layer_norm(lstm_output + attended_output)
                combined = self.dropout(combined)
                output = self.output_projection(combined)
                
                # Get next token
                predicted_idx = output.squeeze(0).argmax(-1).item()
                
                if vocab.itos[predicted_idx] == "<END>":
                    break
                    
                if vocab.itos[predicted_idx] != "<UNK>":
                    caption.append(vocab.itos[predicted_idx])
                
                input_token = torch.tensor([[predicted_idx]]).to(device)
            
            return caption, attention_weights_list


class LightweightCaptioningStudent(nn.Module):
    """
    Ultra-lightweight student model for maximum efficiency
    """
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=1, dropout=0.2):
        super(LightweightCaptioningStudent, self).__init__()
        
        # --- LIGHTWEIGHT ENCODER ---
        # Use MobileNetV3 for maximum efficiency
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.encoder = mobilenet.features
        encoder_dim = 576  # MobileNetV3-Small output channels
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Minimal projection layers
        self.feature_projection = nn.Linear(encoder_dim, hidden_size)
        
        # Freeze encoder completely
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # --- LIGHTWEIGHT DECODER ---
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Store dimensions
        self.encoder_dim = encoder_dim

    def get_image_features(self, images):
        """Extract features for distillation"""
        features = self.encoder(images)
        features = self.adaptive_pool(features)
        return features.view(features.size(0), -1)

    def forward(self, images, captions):
        """Lightweight forward pass"""
        batch_size = images.size(0)
        
        # Extract and project features
        features = self.get_image_features(images)
        features = torch.relu(self.feature_projection(features))
        
        # Initialize LSTM with correct batch size
        num_layers = self.lstm.num_layers
        h_0 = features.unsqueeze(0).repeat(num_layers, 1, 1)
        c_0 = features.unsqueeze(0).repeat(num_layers, 1, 1)
        
        # Process captions - handle input format correctly
        if captions.dim() == 2 and captions.size(0) > captions.size(1):
            # Input is (seq_len, batch) - convert to (batch, seq_len)
            captions = captions.permute(1, 0)
        
        # Ensure captions batch size matches images batch size
        if captions.size(0) != batch_size:
            # Truncate or pad captions to match batch size
            if captions.size(0) > batch_size:
                captions = captions[:batch_size]
            else:
                # Handle case where caption batch is smaller than image batch
                # This can happen with drop_last=False or inconsistent batching
                if batch_idx := getattr(self, '_batch_warning_count', 0) < 3:  # Only show first 3 warnings
                    print(f"[WARNING] Caption batch size {captions.size(0)} < image batch size {batch_size}")
                    print(f"[WARNING] Truncating image batch to match caption batch size")
                    self._batch_warning_count = getattr(self, '_batch_warning_count', 0) + 1
                # Truncate images to match caption batch size
                images = images[:captions.size(0)]
                batch_size = captions.size(0)
                # Re-extract features with the truncated batch
                features = self.get_image_features(images)
                features = torch.relu(self.feature_projection(features))
                # Re-initialize LSTM hidden states with correct batch size
                num_layers = self.lstm.num_layers
                h_0 = features.unsqueeze(0).repeat(num_layers, 1, 1)
                c_0 = features.unsqueeze(0).repeat(num_layers, 1, 1)
        
        captions_input = captions[:, :-1]  # Remove last token for input
        embeddings = self.embedding(captions_input)
        
        # LSTM forward
        outputs, _ = self.lstm(embeddings, (h_0, c_0))
        outputs = self.dropout(outputs)
        outputs = self.fc_out(outputs)
        
        # Return in (seq_len, batch, vocab_size) format to match teacher
        return outputs.permute(1, 0, 2)
