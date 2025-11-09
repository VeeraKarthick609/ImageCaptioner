# src/distillation_utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DistillationLoss(nn.Module):
    """
    Comprehensive Knowledge Distillation Loss combining multiple distillation strategies
    """
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1, temperature=4.0, vocab_size=None):
        super(DistillationLoss, self).__init__()
        
        self.alpha = alpha  # Weight for token-level KD
        self.beta = beta    # Weight for encoder feature KD
        self.gamma = gamma  # Weight for decoder hidden state KD
        self.temperature = temperature
        self.vocab_size = vocab_size
        
        # Standard cross-entropy loss
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is PAD token
        
        # MSE loss for feature matching
        self.mse_loss = nn.MSELoss()
        
        # Cosine similarity loss
        self.cosine_loss = nn.CosineEmbeddingLoss()
        
    def token_level_distillation(self, student_logits, teacher_logits, temperature=None):
        """
        Token-level knowledge distillation using soft targets
        Args:
            student_logits: (seq_len, batch_size, vocab_size)
            teacher_logits: (seq_len, batch_size, vocab_size)
            temperature: Temperature for softmax (default: self.temperature)
        Returns:
            kd_loss: Knowledge distillation loss
        """
        if temperature is None:
            temperature = self.temperature
            
        # Flatten logits for easier computation
        student_flat = student_logits.view(-1, self.vocab_size)  # (seq_len * batch_size, vocab_size)
        teacher_flat = teacher_logits.view(-1, self.vocab_size)  # (seq_len * batch_size, vocab_size)
        
        # Apply temperature scaling and compute soft targets
        student_soft = F.log_softmax(student_flat / temperature, dim=1)
        teacher_soft = F.softmax(teacher_flat / temperature, dim=1)
        
        # KL divergence loss
        kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)
        
        return kd_loss
    
    def encoder_feature_distillation(self, student_features, teacher_features):
        """
        Encoder feature knowledge distillation
        Args:
            student_features: (batch_size, L_s, E_s) - CNN features (e.g., 49 tokens from 7x7)
            teacher_features: (batch_size, L_t, E_t) - ViT features (e.g., 197 tokens)
        Returns:
            feature_loss: Feature matching loss
        """
        batch_size = student_features.size(0)
        
        # Method 1: Global average pooling to match dimensions
        student_global = torch.mean(student_features, dim=1)  # (batch_size, E_s)
        teacher_global = torch.mean(teacher_features, dim=1)  # (batch_size, E_t)
        
        # If dimensions don't match, we need a projection layer (handled in training script)
        if student_global.size(-1) != teacher_global.size(-1):
            # This should be handled by a projection layer in the training script
            raise ValueError(f"Feature dimensions don't match: student {student_global.size(-1)}, teacher {teacher_global.size(-1)}")
        
        # MSE loss on global features
        global_loss = self.mse_loss(student_global, teacher_global)
        
        # Method 2: Attention-based feature matching
        # Compute attention weights for both student and teacher
        student_attn = F.softmax(torch.sum(student_features, dim=-1), dim=1)  # (batch_size, L_s)
        teacher_attn = F.softmax(torch.sum(teacher_features, dim=-1), dim=1)  # (batch_size, L_t)
        
        # Weighted features
        student_weighted = torch.sum(student_features * student_attn.unsqueeze(-1), dim=1)  # (batch_size, E_s)
        teacher_weighted = torch.sum(teacher_features * teacher_attn.unsqueeze(-1), dim=1)  # (batch_size, E_t)
        
        # MSE loss on attention-weighted features
        attention_loss = self.mse_loss(student_weighted, teacher_weighted)
        
        # Combine losses
        feature_loss = 0.6 * global_loss + 0.4 * attention_loss
        
        return feature_loss
    
    def decoder_hidden_state_distillation(self, student_hiddens, teacher_hiddens):
        """
        Decoder hidden state knowledge distillation
        Args:
            student_hiddens: List of (batch_size, hidden_size) tensors
            teacher_hiddens: List of (batch_size, hidden_size) tensors or None
        Returns:
            hidden_loss: Hidden state matching loss
        """
        # If teacher hiddens are not available, return zero loss
        if teacher_hiddens is None or student_hiddens is None:
            return torch.tensor(0.0, device=student_hiddens[0].device if student_hiddens else torch.device('cpu'))
        
        if len(student_hiddens) != len(teacher_hiddens):
            # Interpolate to match sequence lengths
            min_len = min(len(student_hiddens), len(teacher_hiddens))
            student_hiddens = student_hiddens[:min_len]
            teacher_hiddens = teacher_hiddens[:min_len]
        
        hidden_losses = []
        
        for s_hidden, t_hidden in zip(student_hiddens, teacher_hiddens):
            # If dimensions don't match, project (handled in training script)
            if s_hidden.size(-1) != t_hidden.size(-1):
                raise ValueError(f"Hidden dimensions don't match: student {s_hidden.size(-1)}, teacher {t_hidden.size(-1)}")
            
            # Cosine similarity loss
            target = torch.ones(s_hidden.size(0)).to(s_hidden.device)
            cos_loss = self.cosine_loss(s_hidden, t_hidden, target)
            
            # MSE loss
            mse_loss = self.mse_loss(s_hidden, t_hidden)
            
            # Combine losses
            combined_loss = 0.7 * mse_loss + 0.3 * cos_loss
            hidden_losses.append(combined_loss)
        
        # Average over time steps
        hidden_loss = torch.mean(torch.stack(hidden_losses))
        
        return hidden_loss
    
    def forward(self, student_outputs, teacher_outputs, targets):
        """
        Complete distillation loss computation
        Args:
            student_outputs: Dict containing student model outputs
            teacher_outputs: Dict containing teacher model outputs
            targets: Ground truth targets (seq_len, batch_size)
        Returns:
            total_loss: Combined distillation loss
            loss_dict: Dictionary of individual loss components
        """
        # Extract outputs
        student_logits = student_outputs['logits']  # (seq_len, batch_size, vocab_size)
        teacher_logits = teacher_outputs['logits']  # (seq_len, batch_size, vocab_size)
        
        # 1. Standard cross-entropy loss with ground truth
        ce_loss = self.ce_loss(student_logits.view(-1, self.vocab_size), targets.view(-1))
        
        # 2. Token-level knowledge distillation
        token_kd_loss = self.token_level_distillation(student_logits, teacher_logits)
        
        # 3. Encoder feature distillation (if available)
        feature_kd_loss = 0.0
        if 'encoder_features' in student_outputs and 'encoder_features' in teacher_outputs:
            feature_kd_loss = self.encoder_feature_distillation(
                student_outputs['encoder_features'],
                teacher_outputs['encoder_features']
            )
        
        # Ensure feature_kd_loss is a tensor
        if not isinstance(feature_kd_loss, torch.Tensor):
            feature_kd_loss = torch.tensor(0.0, device=student_logits.device)
        
        # 4. Decoder hidden state distillation (if available)
        hidden_kd_loss = 0.0
        if 'hidden_states' in student_outputs and 'hidden_states' in teacher_outputs:
            hidden_kd_loss = self.decoder_hidden_state_distillation(
                student_outputs['hidden_states'],
                teacher_outputs['hidden_states']
            )
        
        # Ensure hidden_kd_loss is a tensor
        if not isinstance(hidden_kd_loss, torch.Tensor):
            hidden_kd_loss = torch.tensor(0.0, device=student_logits.device)
        
        # Combine all losses
        total_loss = (
            (1 - self.alpha - self.beta - self.gamma) * ce_loss +
            self.alpha * token_kd_loss +
            self.beta * feature_kd_loss +
            self.gamma * hidden_kd_loss
        )
        
        # Loss dictionary for monitoring
        loss_dict = {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'token_kd_loss': token_kd_loss.item(),
            'feature_kd_loss': feature_kd_loss.item() if isinstance(feature_kd_loss, torch.Tensor) else feature_kd_loss,
            'hidden_kd_loss': hidden_kd_loss.item() if isinstance(hidden_kd_loss, torch.Tensor) else hidden_kd_loss
        }
        
        return total_loss, loss_dict


class FeatureProjector(nn.Module):
    """
    Projects teacher features to student feature space
    Handles both feature dimension and sequence length mismatches
    """
    def __init__(self, teacher_dim, student_dim, teacher_seq_len=197, student_seq_len=64):
        super(FeatureProjector, self).__init__()
        self.teacher_dim = teacher_dim
        self.student_dim = student_dim
        self.teacher_seq_len = teacher_seq_len
        self.student_seq_len = student_seq_len
        
        # Feature dimension projection
        if teacher_dim != student_dim:
            self.feature_projection = nn.Sequential(
                nn.Linear(teacher_dim, student_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.LayerNorm(student_dim)
            )
        else:
            self.feature_projection = nn.Identity()
        
        # Sequence length alignment
        if teacher_seq_len != student_seq_len:
            # Use adaptive pooling to handle sequence length mismatch
            self.seq_projection = nn.AdaptiveAvgPool1d(student_seq_len)
        else:
            self.seq_projection = nn.Identity()
    
    def forward(self, features):
        """
        Args:
            features: (batch_size, teacher_seq_len, teacher_dim)
        Returns:
            projected_features: (batch_size, student_seq_len, student_dim)
        """
        # First project feature dimensions
        projected = self.feature_projection(features)  # (batch_size, teacher_seq_len, student_dim)
        
        # Then handle sequence length mismatch
        if self.teacher_seq_len != self.student_seq_len:
            # Transpose for pooling: (batch_size, student_dim, teacher_seq_len)
            projected = projected.transpose(1, 2)
            # Pool to target sequence length: (batch_size, student_dim, student_seq_len)
            projected = self.seq_projection(projected)
            # Transpose back: (batch_size, student_seq_len, student_dim)
            projected = projected.transpose(1, 2)
        
        return projected


class TeacherWrapper(nn.Module):
    """
    Wrapper for teacher model to extract intermediate features for distillation
    """
    def __init__(self, teacher_model):
        super(TeacherWrapper, self).__init__()
        self.teacher = teacher_model
        self.teacher.eval()  # Always in eval mode
        
        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def forward(self, images, captions):
        """
        Forward pass through teacher model with feature extraction
        """
        with torch.no_grad():
            # Ensure inputs are in the right precision (FP32)
            images = images.float()
            captions = captions.long()
            
            # Get teacher outputs without autocast to avoid precision issues
            teacher_logits = self.teacher(images, captions)
            
            # Extract encoder features (ViT features)
            teacher_encoder_features = self.teacher.encoder.forward_features(images)  # (batch_size, 197, 384)
            teacher_encoder_features = self.teacher.encoder_projection(teacher_encoder_features)
            
            # Ensure outputs are in FP32
            teacher_logits = teacher_logits.float()
            teacher_encoder_features = teacher_encoder_features.float()
            
            return {
                'logits': teacher_logits,
                'encoder_features': teacher_encoder_features,
                'hidden_states': None  # Would need teacher model modification to extract these
            }


def create_feature_projectors(teacher_model, student_model):
    """
    Create feature projection layers to match dimensions between teacher and student
    """
    projectors = {}
    
    # Encoder feature projector
    # Get teacher encoder output dimension
    if hasattr(teacher_model.encoder_projection, 'out_features'):
        teacher_encoder_dim = teacher_model.encoder_projection.out_features
    elif hasattr(teacher_model.encoder_projection, 'in_features'):
        teacher_encoder_dim = teacher_model.encoder_projection.in_features
    else:
        teacher_encoder_dim = teacher_model.encoder.num_features
    
    student_encoder_dim = student_model.embed_size
    
    # Dynamically detect student sequence length from encoder architecture
    if hasattr(student_model.encoder, 'adaptive_pool'):
        # Get the output size from adaptive pooling layer
        pool_output_size = student_model.encoder.adaptive_pool.output_size
        if isinstance(pool_output_size, tuple):
            student_seq_len = pool_output_size[0] * pool_output_size[1]
        else:
            student_seq_len = pool_output_size * pool_output_size
    else:
        # Fallback to default
        student_seq_len = 64
    
    print(f"Creating encoder projector: {teacher_encoder_dim} -> {student_encoder_dim}, seq_len: 197 -> {student_seq_len}")
    projectors['encoder'] = FeatureProjector(
        teacher_encoder_dim, 
        student_encoder_dim,
        teacher_seq_len=197,  # ViT tokens
        student_seq_len=student_seq_len  # Dynamically detected CNN spatial locations
    )
    
    # Hidden state projector (if needed)
    # This would depend on the specific dimensions of teacher and student hidden states
    teacher_hidden_dim = getattr(teacher_model, 'embed_size', 512)  # Teacher embedding size
    student_hidden_dim = student_model.hidden_size
    
    print(f"Creating hidden projector: {teacher_hidden_dim} -> {student_hidden_dim}")
    projectors['hidden'] = FeatureProjector(teacher_hidden_dim, student_hidden_dim)
    
    return projectors


def validate_distillation_setup(teacher_model, student_model, sample_batch):
    """
    Validate that the distillation setup is correct by running a forward pass
    """
    print("Validating distillation setup...")
    
    images, captions = sample_batch
    batch_size = images.size(0)
    
    # Teacher forward pass - ensure correct precision
    teacher_wrapper = TeacherWrapper(teacher_model)
    images_fp32 = images.float()
    captions_fp32 = captions.long()
    teacher_outputs = teacher_wrapper(images_fp32, captions_fp32)
    
    # Student forward pass
    student_logits, student_encoder_features, student_hidden_states, _ = student_model(images, captions)
    student_outputs = {
        'logits': student_logits,
        'encoder_features': student_encoder_features,
        'hidden_states': student_hidden_states
    }
    
    # Check dimensions
    print(f"Teacher logits shape: {teacher_outputs['logits'].shape}")
    print(f"Student logits shape: {student_outputs['logits'].shape}")
    print(f"Teacher encoder features shape: {teacher_outputs['encoder_features'].shape}")
    print(f"Student encoder features shape: {student_outputs['encoder_features'].shape}")
    
    # Create projectors and move to device
    projectors = create_feature_projectors(teacher_model, student_model)
    device = images.device
    for key in projectors:
        projectors[key] = projectors[key].to(device)
    
    # Test feature projection
    projected_teacher_features = projectors['encoder'](teacher_outputs['encoder_features'])
    print(f"Projected teacher features shape: {projected_teacher_features.shape}")
    
    # Test distillation loss
    vocab_size = teacher_outputs['logits'].size(-1)
    distill_loss = DistillationLoss(vocab_size=vocab_size)
    
    # Update teacher outputs with projected features
    teacher_outputs['encoder_features'] = projected_teacher_features
    
    total_loss, loss_dict = distill_loss(student_outputs, teacher_outputs, captions)
    
    print(f"Distillation loss validation successful!")
    print(f"Loss components: {loss_dict}")
    
    return projectors, distill_loss


# Utility functions for monitoring training
def compute_bleu_score(predicted_tokens, target_tokens, vocab):
    """Simple BLEU-1 score for monitoring"""
    pred_words = [vocab.itos[idx] for idx in predicted_tokens if idx not in [0, 1, 2]]  # Remove PAD, START, END
    target_words = [vocab.itos[idx] for idx in target_tokens if idx not in [0, 1, 2]]
    
    if len(target_words) == 0:
        return 0.0
    
    pred_set = set(pred_words)
    target_set = set(target_words)
    
    return len(pred_set.intersection(target_set)) / len(target_set)


def log_training_progress(epoch, batch_idx, loss_dict, learning_rate, total_batches):
    """Log training progress with loss components"""
    if batch_idx % 50 == 0:
        print(f"Epoch {epoch}, Batch {batch_idx}/{total_batches}")
        print(f"  LR: {learning_rate:.6f}")
        print(f"  Total Loss: {loss_dict['total_loss']:.4f}")
        print(f"  CE Loss: {loss_dict['ce_loss']:.4f}")
        print(f"  Token KD: {loss_dict['token_kd_loss']:.4f}")
        print(f"  Feature KD: {loss_dict['feature_kd_loss']:.4f}")
        print(f"  Hidden KD: {loss_dict['hidden_kd_loss']:.4f}")
        print("-" * 50)
