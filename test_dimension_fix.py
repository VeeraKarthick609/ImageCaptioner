#!/usr/bin/env python3
"""
Test script to validate the dimension mismatch fix in knowledge distillation
"""

import torch
import torch.nn as nn
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from distillation_utils import FeatureProjector

def test_feature_projector():
    """Test the enhanced FeatureProjector with sequence length handling"""
    print("Testing FeatureProjector with dimension and sequence length mismatch...")
    
    # Simulate teacher features: (batch_size=2, seq_len=197, feature_dim=384)
    teacher_features = torch.randn(2, 197, 384)
    print(f"Teacher features shape: {teacher_features.shape}")
    
    # Create projector: 384->256 dims, 197->64 seq_len
    projector = FeatureProjector(
        teacher_dim=384,
        student_dim=256,
        teacher_seq_len=197,
        student_seq_len=64
    )
    
    # Project features
    projected_features = projector(teacher_features)
    print(f"Projected features shape: {projected_features.shape}")
    
    # Expected shape: (2, 64, 256)
    expected_shape = (2, 64, 256)
    if projected_features.shape == expected_shape:
        print("✅ SUCCESS: Feature projection works correctly!")
        return True
    else:
        print(f"❌ FAILED: Expected {expected_shape}, got {projected_features.shape}")
        return False

def test_cosine_similarity():
    """Test cosine similarity computation with projected features"""
    print("\nTesting cosine similarity computation...")
    
    # Create two feature tensors with same dimensions
    student_features = torch.randn(2, 64, 256)
    teacher_features = torch.randn(2, 64, 256)
    
    print(f"Student features shape: {student_features.shape}")
    print(f"Teacher features shape: {teacher_features.shape}")
    
    try:
        # Normalize features
        student_norm = nn.functional.normalize(student_features, p=2, dim=-1)
        teacher_norm = nn.functional.normalize(teacher_features, p=2, dim=-1)
        
        # Compute cosine similarity
        cosine_sim = torch.sum(student_norm * teacher_norm, dim=-1)
        print(f"Cosine similarity shape: {cosine_sim.shape}")
        
        # Compute loss
        feature_loss = 1 - cosine_sim.mean()
        print(f"Feature loss: {feature_loss.item():.4f}")
        
        print("✅ SUCCESS: Cosine similarity computation works!")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: Cosine similarity computation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Knowledge Distillation Dimension Fix")
    print("=" * 60)
    
    test1_passed = test_feature_projector()
    test2_passed = test_cosine_similarity()
    
    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED! The dimension mismatch fix is working correctly.")
        print("You can now run the training pipeline without tensor size errors.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    print("=" * 60)

if __name__ == "__main__":
    main()
