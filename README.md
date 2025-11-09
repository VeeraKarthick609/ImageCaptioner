# Knowledge Distillation for Image Captioning

This project implements knowledge distillation from a Vision Transformer (ViT) teacher model to a CNN-LSTM student model for image captioning.

## Architecture Overview

### Teacher Model (ViT-Transformer)
- **Encoder**: Vision Transformer (vit_small_patch16_224)
- **Decoder**: Transformer decoder with multi-head attention
- **Parameters**: ~25M parameters
- **Features**: 197 tokens (196 patches + 1 CLS token), 384/512 dimensions

### Student Model (CNN-LSTM)
- **Encoder**: ResNet-50 CNN backbone
- **Pre-projection**: Optional attention refinement layer
- **Projection**: Linear projection to token space
- **Decoder**: LSTM with attention mechanism
- **Parameters**: ~8M parameters (3x compression)
- **Features**: 49 tokens (7×7 spatial locations), 256 dimensions

## Knowledge Distillation Strategy

### 1. Token-Level Distillation (Most Important)
- **Location**: Right after decoder logits
- **Method**: KL divergence between soft distributions
- **Temperature**: 4.0 for softmax smoothing
- **Weight**: α = 0.7

### 2. Encoder Feature Distillation
- **Teacher**: ViT features (batch_size, 197, 384/512)
- **Student**: CNN features (batch_size, 49, 256)
- **Matching**: Global average pooling + attention-weighted features
- **Projection**: Linear layer to match dimensions
- **Weight**: β = 0.2

### 3. Decoder Hidden State Distillation
- **Method**: MSE + Cosine similarity loss
- **Matching**: Sequence-level hidden state alignment
- **Weight**: γ = 0.1

## File Structure

```
src/
├── teacher_model.py          # ViT-Transformer teacher model
├── student_model.py          # CNN-LSTM student model
├── distillation_utils.py     # KD loss functions and utilities
├── train_student_kd.py       # Knowledge distillation training script
├── evaluate_student.py       # Student vs teacher comparison
├── test_kd_pipeline.py       # Pipeline validation tests
├── data_loader.py           # Data loading utilities
├── train_teacher.py         # Teacher model training
└── evaluate_teacher.py      # Teacher model evaluation
```

## Usage

### 1. Test the Pipeline
```bash
cd src
python test_kd_pipeline.py
```
This validates all components before training.

### 2. Train the Student Model
```bash
cd src
python train_student_kd.py
```

### 3. Evaluate Student vs Teacher
```bash
cd src
python evaluate_student.py
```

## Training Configuration

### Hyperparameters
- **Learning Rate**: 2e-4 (higher than teacher due to smaller model)
- **Batch Size**: 16 (can use larger due to smaller model)
- **Epochs**: 30
- **Temperature**: 4.0
- **Distillation Weights**: α=0.7, β=0.2, γ=0.1

### Model Architecture
- **Student Embed Size**: 256 (vs 512 for teacher)
- **LSTM Hidden Size**: 512
- **LSTM Layers**: 2
- **Dropout**: 0.3
- **Attention Refinement**: Enabled

### Optimization
- **Optimizer**: AdamW with different LRs for encoder/decoder
- **Scheduler**: CosineAnnealingWarmRestarts
- **Mixed Precision**: Enabled
- **Gradient Clipping**: 1.0
- **Early Stopping**: Patience of 7 epochs

## Expected Performance

### Model Efficiency
- **Compression Ratio**: ~3x smaller (8M vs 25M parameters)
- **Inference Speed**: ~2-3x faster
- **Memory Usage**: ~2x less GPU memory

### Quality Metrics
- **BLEU-1**: Expected 85-95% of teacher performance
- **BLEU-2**: Expected 80-90% of teacher performance
- **METEOR**: Expected 85-95% of teacher performance

## Key Features

### Student Model Innovations
1. **CNN Encoder with Fine-tuning**: ResNet-50 with selective layer unfreezing
2. **Attention Refinement**: Pre-projection self-attention for feature enhancement
3. **LSTM with Attention**: Spatial attention over CNN features
4. **Efficient Architecture**: Optimized for inference speed

### Distillation Innovations
1. **Multi-level KD**: Token, feature, and hidden state distillation
2. **Feature Projection**: Automatic dimension matching between teacher/student
3. **Attention-based Matching**: Weighted feature alignment
4. **Temperature Scaling**: Soft target generation

### Training Innovations
1. **Mixed Precision**: Faster training with FP16
2. **Gradient Accumulation**: Effective larger batch sizes
3. **Differential Learning Rates**: Lower LR for pre-trained components
4. **Comprehensive Monitoring**: Loss decomposition and BLEU tracking

## Monitoring and Evaluation

### Training Metrics
- Total distillation loss
- Cross-entropy loss (ground truth)
- Token-level KD loss
- Feature KD loss
- Hidden state KD loss
- Validation BLEU scores

### Evaluation Metrics
- BLEU-1, BLEU-2 scores
- METEOR scores
- Inference time comparison
- Model size comparison
- Success rate
- Sample caption comparisons

## Troubleshooting

### Common Issues
1. **Dimension Mismatch**: Check feature projector configurations
2. **Memory Issues**: Reduce batch size or disable attention refinement
3. **Slow Training**: Enable mixed precision and check data loading
4. **Poor Performance**: Adjust distillation weights (α, β, γ)

### Debug Mode
Run with smaller models for debugging:
```python
# In student_model.py, reduce dimensions
embed_size = 128
hidden_size = 256
num_layers = 1
```

## Advanced Configuration

### Custom Distillation Weights
```python
# In train_student_kd.py
alpha = 0.8  # Increase for more focus on token-level KD
beta = 0.15  # Decrease if feature KD is unstable
gamma = 0.05 # Decrease if hidden state KD is unstable
```

### Model Variants
```python
# Smaller student for mobile deployment
student_model = CaptioningStudent(
    vocab_size=vocab_size,
    embed_size=128,
    hidden_size=256,
    num_layers=1,
    use_attention_refinement=False  # Disable for speed
)

# Larger student for better quality
student_model = CaptioningStudent(
    vocab_size=vocab_size,
    embed_size=384,
    hidden_size=768,
    num_layers=3,
    use_attention_refinement=True
)
```

## Results and Saved Models

### Model Checkpoints
- `saved_models/best_teacher_model.pth`: Pre-trained teacher model
- `saved_models/best_student_model.pth`: Best student model during training
- `saved_models/final_student_model.pth`: Final student model

### Training Logs
- `saved_models/student_training_history.json`: Training metrics
- `student_vs_teacher_report.json`: Comparison report

### Evaluation Reports
- Detailed performance comparison
- Sample caption comparisons
- Inference time analysis
- Model efficiency metrics

## Citation and References

This implementation is based on knowledge distillation techniques for neural machine translation and computer vision, adapted specifically for image captioning tasks.

Key papers:
- "Distilling the Knowledge in a Neural Network" (Hinton et al.)
- "Attention Is All You Need" (Vaswani et al.)
- "Show, Attend and Tell" (Xu et al.)

## Future Improvements

1. **Progressive Distillation**: Multi-stage distillation with intermediate models
2. **Adaptive Weights**: Dynamic adjustment of distillation weights during training
3. **Curriculum Learning**: Progressive difficulty in distillation targets
4. **Multi-Teacher Distillation**: Ensemble of teacher models
5. **Quantization**: Post-training quantization for further compression
