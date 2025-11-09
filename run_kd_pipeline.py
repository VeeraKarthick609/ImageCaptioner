# run_kd_pipeline.py

"""
Complete Knowledge Distillation Pipeline Runner
This script runs the entire knowledge distillation pipeline from testing to evaluation.
"""

import os
import sys
import subprocess
import time

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    try:
        # Change to src directory
        os.chdir('src')
        
        # Run the command
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"[PASS] {description} completed successfully!")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print(f"[FAIL] {description} failed!")
            if result.stderr:
                print("Error:")
                print(result.stderr)
            return False
            
        # Change back to root directory
        os.chdir('..')
        return True
        
    except Exception as e:
        print(f"[FAIL] Error running {description}: {e}")
        os.chdir('..')
        return False

def check_prerequisites():
    """Check if all prerequisites are met"""
    print("CHECKING PREREQUISITES")
    print("="*60)
    
    # Check if teacher model exists
    teacher_model_path = "saved_models/best_teacher_model.pth"
    if not os.path.exists(teacher_model_path):
        print(f"[FAIL] Teacher model not found: {teacher_model_path}")
        print("  Please train the teacher model first using: python src/train_teacher.py")
        return False
    else:
        print(f"[PASS] Teacher model found: {teacher_model_path}")
    
    # Check if data exists
    data_path = "data/flickr8k/captions_clean.csv"
    images_path = "data/flickr8k/Images"
    
    if not os.path.exists(data_path):
        print(f"[FAIL] Data file not found: {data_path}")
        return False
    else:
        print(f"[PASS] Data file found: {data_path}")
    
    if not os.path.exists(images_path):
        print(f"[FAIL] Images directory not found: {images_path}")
        return False
    else:
        print(f"[PASS] Images directory found: {images_path}")
    
    # Check Python packages
    required_packages = ['torch', 'torchvision', 'timm', 'PIL', 'tqdm', 'numpy', 'matplotlib']
    for package in required_packages:
        try:
            __import__(package)
            print(f"[PASS] {package} is available")
        except ImportError:
            print(f"[FAIL] {package} is not installed")
            return False
    
    return True

def main():
    """Main pipeline runner - Train and evaluate student model"""
    print("KNOWLEDGE DISTILLATION TRAINING PIPELINE")
    print("="*80)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\nERROR: Prerequisites not met. Please fix the issues above.")
        return False
    
    print("\nSUCCESS: All prerequisites met! Starting training...")
    
    # Step 1: Train the student model
    print(f"\nINFO: Starting student model training at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if not run_command("python train_student_kd.py", "Student Model Training"):
        print("\nERROR: Student model training failed.")
        return False
    
    # Step 2: Evaluate the student model
    print(f"\nINFO: Starting model evaluation at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if not run_command("python evaluate_student.py", "Student Model Evaluation"):
        print("\nERROR: Student model evaluation failed.")
        return False
    
    # Success!
    print("\n" + "="*80)
    print("SUCCESS: KNOWLEDGE DISTILLATION TRAINING COMPLETED!")
    print("="*80)
    
    print("\nGenerated files:")
    print("- saved_models/best_student_model.pth (Best student model)")
    print("- saved_models/final_student_model.pth (Final student model)")
    print("- saved_models/student_training_history.json (Training metrics)")
    print("- student_vs_teacher_report.json (Comparison report)")
    
    print("\nNext steps:")
    print("1. Check the evaluation report for performance metrics")
    print("2. Test the student model on your own images")
    print("3. Deploy the student model for faster inference")
    
    return True


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\nSUCCESS: Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("\nERROR: Pipeline failed!")
        sys.exit(1)
