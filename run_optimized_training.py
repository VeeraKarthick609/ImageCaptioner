#!/usr/bin/env python3
"""
Optimized Training Runner for Enhanced Student Model
"""

import sys
import os
import time
import torch

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Run the optimized training with enhanced student model"""
    print("🚀 Starting Optimized Knowledge Distillation Training")
    print("=" * 60)
    print("🎯 Enhanced Features:")
    print("  • Larger student model (384 embed, 768 hidden, 3 layers)")
    print("  • EfficientNet backbone (or enhanced ResNet-50)")
    print("  • Advanced attention mechanisms")
    print("  • Focal loss for hard examples")
    print("  • Adaptive distillation weights")
    print("  • OneCycleLR scheduler")
    print("  • Optimized data loading")
    print("  • Mixed precision training")
    print("-" * 60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🔥 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("⚠️  No GPU detected - training will be slow on CPU")
    
    # Check if teacher model exists
    teacher_path = 'saved_models/best_teacher_model.pth'
    if not os.path.exists(teacher_path):
        print(f"❌ Teacher model not found: {teacher_path}")
        print("💡 Please train the teacher model first using:")
        print("   python src/train_teacher.py")
        return
    
    # Check if data exists
    data_paths = [
        'data/flickr8k/captions_clean.csv',
        'data/flickr8k/Images'
    ]
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"❌ Data not found: {path}")
            print("💡 Please ensure the Flickr8k dataset is properly set up")
            return
    
    print("✅ All prerequisites met!")
    print("\n🚀 Starting training...")
    
    try:
        # Import and run optimized training
        from train_student_kd_optimized import train_student_with_kd_optimized
        
        start_time = time.time()
        student_model, projectors = train_student_with_kd_optimized()
        total_time = time.time() - start_time
        
        print(f"\n🎉 Training completed successfully!")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"💾 Model saved as: saved_models/best_student_model_optimized.pth")
        print(f"📊 Training history: saved_models/optimized_training_history.json")
        
        # Performance summary
        total_params = sum(p.numel() for p in student_model.parameters())
        print(f"\n📈 Performance Summary:")
        print(f"  • Student parameters: {total_params:,}")
        print(f"  • Expected speedup: ~3-4x faster inference")
        print(f"  • Target accuracy: 85-95% of teacher performance")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        print("\n🔍 Error details:")
        print(traceback.format_exc())
        
        print("\n💡 Troubleshooting tips:")
        print("  • Check GPU memory (reduce batch_size if needed)")
        print("  • Ensure all dependencies are installed")
        print("  • Verify data paths are correct")
        print("  • Check teacher model is properly trained")

if __name__ == "__main__":
    main()
