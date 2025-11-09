# src/evaluate_student.py

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json
import os
import time
from collections import Counter, defaultdict
import random

from teacher_model import CaptioningTeacher
from student_model import CaptioningStudent
from data_loader import get_loader
from distillation_utils import create_feature_projectors

class StudentEvaluator:
    def __init__(self, student_model, teacher_model, vocab, device):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.vocab = vocab
        self.device = device
        self.student_model.eval()
        self.teacher_model.eval()
    
    def bleu_score(self, predicted, reference, n=1):
        """Calculate BLEU-n score"""
        pred_words = predicted.lower().split()
        ref_words = reference.lower().split()
        
        if len(pred_words) < n or len(ref_words) < n:
            return 0.0
        
        # Create n-grams
        pred_ngrams = [tuple(pred_words[i:i+n]) for i in range(len(pred_words)-n+1)]
        ref_ngrams = [tuple(ref_words[i:i+n]) for i in range(len(ref_words)-n+1)]
        
        pred_counter = Counter(pred_ngrams)
        ref_counter = Counter(ref_ngrams)
        
        # Calculate precision
        overlap = 0
        for ngram in pred_counter:
            overlap += min(pred_counter[ngram], ref_counter[ngram])
        
        precision = overlap / len(pred_ngrams) if pred_ngrams else 0
        return precision
    
    def meteor_score_simple(self, predicted, reference):
        """Simplified METEOR score (word overlap based)"""
        pred_words = set(predicted.lower().split())
        ref_words = set(reference.lower().split())
        
        if len(ref_words) == 0:
            return 0.0
        
        overlap = len(pred_words.intersection(ref_words))
        recall = overlap / len(ref_words)
        precision = overlap / len(pred_words) if pred_words else 0
        
        if precision + recall == 0:
            return 0.0
        
        f_score = 2 * precision * recall / (precision + recall)
        return f_score
    
    def measure_inference_time(self, image, num_runs=10):
        """Measure inference time for both models"""
        # Warm up
        for _ in range(3):
            with torch.no_grad():
                _ = self.student_model.caption_image(image, self.vocab, max_length=20)
                _ = self.teacher_model.caption_image(image, self.vocab, max_length=20)
        
        # Measure student
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = self.student_model.caption_image(image, self.vocab, max_length=20)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        student_time = (time.time() - start_time) / num_runs
        
        # Measure teacher
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = self.teacher_model.caption_image(image, self.vocab, max_length=20)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        teacher_time = (time.time() - start_time) / num_runs
        
        return student_time, teacher_time
    
    def compare_models_on_dataset(self, data_loader, num_samples=200):
        """Compare student and teacher models on dataset"""
        results = {
            'student': {
                'bleu1_scores': [],
                'bleu2_scores': [],
                'meteor_scores': [],
                'generated_captions': [],
                'inference_times': [],
                'successful_generations': 0
            },
            'teacher': {
                'bleu1_scores': [],
                'bleu2_scores': [],
                'meteor_scores': [],
                'generated_captions': [],
                'inference_times': [],
                'successful_generations': 0
            },
            'reference_captions': [],
            'total_samples': 0
        }
        
        print(f"Comparing models on {min(num_samples, len(data_loader) * data_loader.batch_size)} samples...")
        
        with torch.no_grad():
            for i, (imgs, captions) in enumerate(tqdm(data_loader)):
                if results['total_samples'] >= num_samples:
                    break
                
                for j in range(min(imgs.size(0), 3)):  # Evaluate up to 3 images per batch
                    if results['total_samples'] >= num_samples:
                        break
                    
                    img = imgs[j].to(self.device)
                    
                    # Get reference caption
                    ref_caption_tokens = captions[:, j].cpu().numpy()
                    ref_caption = ' '.join([
                        self.vocab.itos[token] for token in ref_caption_tokens 
                        if token not in [self.vocab.stoi["<START>"], self.vocab.stoi["<END>"], self.vocab.stoi["<PAD>"]]
                    ]).strip()
                    
                    results['reference_captions'].append(ref_caption)
                    
                    # Measure inference times
                    student_time, teacher_time = self.measure_inference_time(img, num_runs=5)
                    results['student']['inference_times'].append(student_time)
                    results['teacher']['inference_times'].append(teacher_time)
                    
                    # Generate captions
                    try:
                        # Student caption
                        student_tokens = self.student_model.caption_image(img, self.vocab, max_length=25)
                        student_caption = ' '.join(student_tokens).strip()
                        
                        if student_caption and len(student_caption.split()) > 2:
                            # Calculate metrics
                            bleu1 = self.bleu_score(student_caption, ref_caption, 1)
                            bleu2 = self.bleu_score(student_caption, ref_caption, 2)
                            meteor = self.meteor_score_simple(student_caption, ref_caption)
                            
                            results['student']['bleu1_scores'].append(bleu1)
                            results['student']['bleu2_scores'].append(bleu2)
                            results['student']['meteor_scores'].append(meteor)
                            results['student']['generated_captions'].append(student_caption)
                            results['student']['successful_generations'] += 1
                        else:
                            results['student']['generated_captions'].append("")
                            
                    except Exception as e:
                        print(f"Error generating student caption: {e}")
                        results['student']['generated_captions'].append("")
                    
                    try:
                        # Teacher caption
                        teacher_tokens = self.teacher_model.caption_image(img, self.vocab, max_length=25)
                        if isinstance(teacher_tokens, list) and len(teacher_tokens) > 0:
                            teacher_caption = teacher_tokens[0] if isinstance(teacher_tokens[0], str) else ' '.join(teacher_tokens)
                        else:
                            teacher_caption = ' '.join(teacher_tokens) if teacher_tokens else ""
                        
                        if teacher_caption and len(teacher_caption.split()) > 2:
                            # Calculate metrics
                            bleu1 = self.bleu_score(teacher_caption, ref_caption, 1)
                            bleu2 = self.bleu_score(teacher_caption, ref_caption, 2)
                            meteor = self.meteor_score_simple(teacher_caption, ref_caption)
                            
                            results['teacher']['bleu1_scores'].append(bleu1)
                            results['teacher']['bleu2_scores'].append(bleu2)
                            results['teacher']['meteor_scores'].append(meteor)
                            results['teacher']['generated_captions'].append(teacher_caption)
                            results['teacher']['successful_generations'] += 1
                        else:
                            results['teacher']['generated_captions'].append("")
                            
                    except Exception as e:
                        print(f"Error generating teacher caption: {e}")
                        results['teacher']['generated_captions'].append("")
                    
                    results['total_samples'] += 1
        
        return results
    
    def evaluate_single_image_comparison(self, image_path, show_image=True):
        """Compare both models on a single image"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        try:
            image = Image.open(image_path).convert('RGB')
            img_tensor = transform(image).to(self.device)
            
            # Generate captions
            with torch.no_grad():
                # Measure inference times
                student_time, teacher_time = self.measure_inference_time(img_tensor, num_runs=10)
                
                # Generate captions
                student_tokens = self.student_model.caption_image(img_tensor, self.vocab, max_length=25)
                student_caption = ' '.join(student_tokens).strip()
                
                teacher_tokens = self.teacher_model.caption_image(img_tensor, self.vocab, max_length=25)
                if isinstance(teacher_tokens, list) and len(teacher_tokens) > 0:
                    teacher_caption = teacher_tokens[0] if isinstance(teacher_tokens[0], str) else ' '.join(teacher_tokens)
                else:
                    teacher_caption = ' '.join(teacher_tokens) if teacher_tokens else ""
            
            if show_image:
                plt.figure(figsize=(15, 8))
                
                # Show image
                plt.subplot(1, 3, 1)
                plt.imshow(image)
                plt.axis('off')
                plt.title('Input Image')
                
                # Student caption
                plt.subplot(1, 3, 2)
                plt.text(0.1, 0.7, f"Student Model:\n\n'{student_caption}'", 
                        fontsize=12, wrap=True, verticalalignment='center')
                plt.text(0.1, 0.3, f"Inference Time: {student_time*1000:.1f}ms", 
                        fontsize=10, color='blue')
                plt.axis('off')
                plt.title('Student (CNN-LSTM)')
                
                # Teacher caption
                plt.subplot(1, 3, 3)
                plt.text(0.1, 0.7, f"Teacher Model:\n\n'{teacher_caption}'", 
                        fontsize=12, wrap=True, verticalalignment='center')
                plt.text(0.1, 0.3, f"Inference Time: {teacher_time*1000:.1f}ms", 
                        fontsize=10, color='red')
                plt.axis('off')
                plt.title('Teacher (ViT-Transformer)')
                
                plt.tight_layout()
                plt.show()
            
            return {
                'student_caption': student_caption,
                'teacher_caption': teacher_caption,
                'student_time': student_time,
                'teacher_time': teacher_time,
                'speedup': teacher_time / student_time if student_time > 0 else 0
            }
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def generate_comparison_report(self, results, save_path='student_vs_teacher_report.json'):
        """Generate comprehensive comparison report"""
        if not results['student']['bleu1_scores'] or not results['teacher']['bleu1_scores']:
            print("Insufficient data for comparison report!")
            return
        
        # Calculate statistics
        student_stats = {
            'avg_bleu1': np.mean(results['student']['bleu1_scores']),
            'avg_bleu2': np.mean(results['student']['bleu2_scores']),
            'avg_meteor': np.mean(results['student']['meteor_scores']),
            'avg_inference_time': np.mean(results['student']['inference_times']),
            'success_rate': results['student']['successful_generations'] / results['total_samples'],
            'std_bleu1': np.std(results['student']['bleu1_scores'])
        }
        
        teacher_stats = {
            'avg_bleu1': np.mean(results['teacher']['bleu1_scores']),
            'avg_bleu2': np.mean(results['teacher']['bleu2_scores']),
            'avg_meteor': np.mean(results['teacher']['meteor_scores']),
            'avg_inference_time': np.mean(results['teacher']['inference_times']),
            'success_rate': results['teacher']['successful_generations'] / results['total_samples'],
            'std_bleu1': np.std(results['teacher']['bleu1_scores'])
        }
        
        # Calculate relative performance
        performance_ratio = {
            'bleu1_ratio': student_stats['avg_bleu1'] / teacher_stats['avg_bleu1'],
            'bleu2_ratio': student_stats['avg_bleu2'] / teacher_stats['avg_bleu2'],
            'meteor_ratio': student_stats['avg_meteor'] / teacher_stats['avg_meteor'],
            'speedup': teacher_stats['avg_inference_time'] / student_stats['avg_inference_time']
        }
        
        # Model size comparison
        student_params = sum(p.numel() for p in self.student_model.parameters())
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        compression_ratio = teacher_params / student_params
        
        # Create report
        report = {
            'model_comparison': {
                'student_performance': student_stats,
                'teacher_performance': teacher_stats,
                'relative_performance': performance_ratio,
                'model_efficiency': {
                    'student_params': int(student_params),
                    'teacher_params': int(teacher_params),
                    'compression_ratio': float(compression_ratio),
                    'speedup': float(performance_ratio['speedup'])
                }
            },
            'sample_comparisons': []
        }
        
        # Add sample comparisons
        num_samples = min(20, len(results['student']['generated_captions']))
        for i in range(num_samples):
            if (i < len(results['student']['generated_captions']) and 
                i < len(results['teacher']['generated_captions']) and
                i < len(results['reference_captions'])):
                
                sample = {
                    'reference': results['reference_captions'][i],
                    'student_caption': results['student']['generated_captions'][i],
                    'teacher_caption': results['teacher']['generated_captions'][i],
                    'student_bleu1': float(results['student']['bleu1_scores'][i]) if i < len(results['student']['bleu1_scores']) else 0.0,
                    'teacher_bleu1': float(results['teacher']['bleu1_scores'][i]) if i < len(results['teacher']['bleu1_scores']) else 0.0
                }
                report['sample_comparisons'].append(sample)
        
        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("STUDENT vs TEACHER MODEL COMPARISON REPORT")
        print("="*80)
        
        print(f"\nMODEL EFFICIENCY:")
        print(f"  Compression Ratio: {compression_ratio:.2f}x smaller")
        print(f"  Inference Speedup: {performance_ratio['speedup']:.2f}x faster")
        print(f"  Student Parameters: {student_params:,}")
        print(f"  Teacher Parameters: {teacher_params:,}")
        
        print(f"\nPERFORMANCE COMPARISON:")
        print(f"  BLEU-1 - Student: {student_stats['avg_bleu1']:.4f}, Teacher: {teacher_stats['avg_bleu1']:.4f} (Ratio: {performance_ratio['bleu1_ratio']:.3f})")
        print(f"  BLEU-2 - Student: {student_stats['avg_bleu2']:.4f}, Teacher: {teacher_stats['avg_bleu2']:.4f} (Ratio: {performance_ratio['bleu2_ratio']:.3f})")
        print(f"  METEOR - Student: {student_stats['avg_meteor']:.4f}, Teacher: {teacher_stats['avg_meteor']:.4f} (Ratio: {performance_ratio['meteor_ratio']:.3f})")
        
        print(f"\nINFERENCE TIMES:")
        print(f"  Student: {student_stats['avg_inference_time']*1000:.1f}ms")
        print(f"  Teacher: {teacher_stats['avg_inference_time']*1000:.1f}ms")
        
        print(f"\nSUCCESS RATES:")
        print(f"  Student: {student_stats['success_rate']:.2%}")
        print(f"  Teacher: {teacher_stats['success_rate']:.2%}")
        
        print(f"\nSample Comparisons:")
        print("-" * 60)
        for i, sample in enumerate(report['sample_comparisons'][:5]):
            print(f"{i+1}. Reference: {sample['reference']}")
            print(f"   Student: {sample['student_caption']} (BLEU: {sample['student_bleu1']:.3f})")
            print(f"   Teacher: {sample['teacher_caption']} (BLEU: {sample['teacher_bleu1']:.3f})")
            print()
        
        return report

def main():
    """Main evaluation function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    eval_loader, dataset = get_loader(
        root_folder="data/flickr8k/Images",
        annotation_file="data/flickr8k/captions_clean.csv",
        transform=transform,
        batch_size=8,
        num_workers=2,
        shuffle=False
    )
    
    vocab_size = len(dataset.vocab)
    
    # Load teacher model
    print("Loading teacher model...")
    teacher_checkpoint = torch.load('saved_models/best_teacher_model.pth', map_location=device)
    teacher_model = CaptioningTeacher(
        vocab_size=vocab_size,
        embed_size=512,
        num_heads=8,
        num_decoder_layers=4,
        dropout=0.15
    ).to(device)
    teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
    
    # Load student model
    print("Loading student model...")
    student_checkpoint = torch.load('saved_models/best_student_model.pth', map_location=device)
    config = student_checkpoint['model_config']
    
    student_model = CaptioningStudent(
        vocab_size=vocab_size,
        embed_size=config['embed_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        use_attention_refinement=True
    ).to(device)
    student_model.load_state_dict(student_checkpoint['student_state_dict'])
    
    print(f"Models loaded successfully!")
    
    # Create evaluator
    evaluator = StudentEvaluator(student_model, teacher_model, dataset.vocab, device)
    
    # Compare models on dataset
    print("Starting model comparison...")
    results = evaluator.compare_models_on_dataset(eval_loader, num_samples=100)
    
    # Generate report
    report = evaluator.generate_comparison_report(results)
    
    print(f"\nDetailed comparison report saved to: student_vs_teacher_report.json")
    
    # Test on specific images
    print("\n" + "="*80)
    print("SINGLE IMAGE COMPARISONS")
    print("="*80)
    
    image_dir = "data/flickr8k/Images"
    if os.path.exists(image_dir):
        all_images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        test_images = random.sample(all_images, min(3, len(all_images)))
        
        for img_name in test_images:
            img_path = os.path.join(image_dir, img_name)
            print(f"\nEvaluating: {img_name}")
            comparison = evaluator.evaluate_single_image_comparison(img_path, show_image=False)
            if comparison:
                print(f"Student: {comparison['student_caption']}")
                print(f"Teacher: {comparison['teacher_caption']}")
                print(f"Speedup: {comparison['speedup']:.2f}x")
    
    return report

if __name__ == "__main__":
    main()
