# src/evaluate_teacher.py

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json
import os
from collections import Counter, defaultdict
import random

from models import CaptioningTeacher
from data_loader import get_loader

class CaptionEvaluator:
    def __init__(self, model, vocab, device):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.model.eval()
    
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
    
    def caption_length_stats(self, captions):
        """Analyze caption length statistics"""
        lengths = [len(cap.split()) for cap in captions]
        return {
            'mean': np.mean(lengths),
            'std': np.std(lengths),
            'min': np.min(lengths),
            'max': np.max(lengths),
            'median': np.median(lengths)
        }
    
    def vocabulary_diversity(self, captions):
        """Calculate vocabulary diversity metrics"""
        all_words = []
        for cap in captions:
            all_words.extend(cap.lower().split())
        
        unique_words = set(all_words)
        total_words = len(all_words)
        
        return {
            'total_words': total_words,
            'unique_words': len(unique_words),
            'vocabulary_diversity': len(unique_words) / total_words if total_words > 0 else 0,
            'most_common': Counter(all_words).most_common(10)
        }
    
    def evaluate_on_dataset(self, data_loader, num_samples=500):
        """Evaluate model on a subset of the dataset"""
        results = {
            'bleu1_scores': [],
            'bleu2_scores': [],
            'meteor_scores': [],
            'generated_captions': [],
            'reference_captions': [],
            'successful_generations': 0,
            'total_samples': 0
        }
        
        print(f"Evaluating on {min(num_samples, len(data_loader))} samples...")
        
        with torch.no_grad():
            for i, (imgs, captions) in enumerate(tqdm(data_loader)):
                if i >= num_samples:
                    break
                    
                for j in range(min(imgs.size(0), 5)):  # Evaluate up to 5 images per batch
                    if results['total_samples'] >= num_samples:
                        break
                        
                    img = imgs[j].to(self.device)
                    
                    # Get reference caption (first caption for this image)
                    ref_caption_tokens = captions[:, j].cpu().numpy()
                    ref_caption = ' '.join([
                        self.vocab.itos[token] for token in ref_caption_tokens 
                        if token not in [self.vocab.stoi["<START>"], self.vocab.stoi["<END>"], self.vocab.stoi["<PAD>"]]
                    ]).strip()
                    
                    # Generate caption
                    try:
                        generated_tokens = self.model.caption_image(img, self.vocab, max_length=25)
                        generated_caption = ' '.join(generated_tokens).strip()
                        
                        if generated_caption and len(generated_caption.split()) > 2:
                            # Calculate metrics
                            bleu1 = self.bleu_score(generated_caption, ref_caption, 1)
                            bleu2 = self.bleu_score(generated_caption, ref_caption, 2)
                            meteor = self.meteor_score_simple(generated_caption, ref_caption)
                            
                            results['bleu1_scores'].append(bleu1)
                            results['bleu2_scores'].append(bleu2)
                            results['meteor_scores'].append(meteor)
                            results['generated_captions'].append(generated_caption)
                            results['reference_captions'].append(ref_caption)
                            results['successful_generations'] += 1
                            
                    except Exception as e:
                        print(f"Error generating caption: {e}")
                    
                    results['total_samples'] += 1
        
        return results
    
    def evaluate_single_image(self, image_path, show_image=True):
        """Evaluate model on a single image"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        try:
            image = Image.open(image_path).convert('RGB')
            img_tensor = transform(image).to(self.device)
            
            # Generate caption
            with torch.no_grad():
                generated_tokens = self.model.caption_image(img_tensor, self.vocab, max_length=25)
                caption = ' '.join(generated_tokens).strip()
            
            if show_image:
                plt.figure(figsize=(10, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(image)
                plt.axis('off')
                plt.title('Input Image')
                
                plt.subplot(1, 2, 2)
                plt.text(0.1, 0.5, f"Generated Caption:\n\n'{caption}'", 
                        fontsize=14, wrap=True, verticalalignment='center')
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            
            return caption
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def generate_evaluation_report(self, results, save_path='evaluation_report.json'):
        """Generate comprehensive evaluation report"""
        if not results['bleu1_scores']:
            print("No successful caption generations found!")
            return
        
        # Calculate average metrics
        avg_bleu1 = np.mean(results['bleu1_scores'])
        avg_bleu2 = np.mean(results['bleu2_scores']) 
        avg_meteor = np.mean(results['meteor_scores'])
        
        # Caption length statistics
        length_stats = self.caption_length_stats(results['generated_captions'])
        
        # Vocabulary diversity
        vocab_diversity = self.vocabulary_diversity(results['generated_captions'])
        
        # Create report
        report = {
            'model_performance': {
                'success_rate': results['successful_generations'] / results['total_samples'],
                'average_bleu1': float(avg_bleu1),
                'average_bleu2': float(avg_bleu2),
                'average_meteor': float(avg_meteor),
                'bleu1_std': float(np.std(results['bleu1_scores'])),
                'total_evaluated': results['total_samples'],
                'successful_generations': results['successful_generations']
            },
            'caption_statistics': {
                'length_stats': {k: float(v) for k, v in length_stats.items()},
                'vocabulary_diversity': vocab_diversity
            },
            'sample_captions': [
                {
                    'generated': results['generated_captions'][i],
                    'reference': results['reference_captions'][i],
                    'bleu1': float(results['bleu1_scores'][i])
                }
                for i in range(min(20, len(results['generated_captions'])))
            ]
        }
        
        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("TEACHER MODEL EVALUATION REPORT")
        print("="*60)
        print(f"Success Rate: {report['model_performance']['success_rate']:.2%}")
        print(f"Average BLEU-1: {avg_bleu1:.4f}")
        print(f"Average BLEU-2: {avg_bleu2:.4f}")
        print(f"Average METEOR: {avg_meteor:.4f}")
        print(f"Caption Length: {length_stats['mean']:.1f} ± {length_stats['std']:.1f} words")
        print(f"Vocabulary Diversity: {vocab_diversity['vocabulary_diversity']:.4f}")
        print(f"Total Samples Evaluated: {results['total_samples']}")
        
        print("\nSample Generated Captions:")
        print("-" * 40)
        for i in range(min(10, len(results['generated_captions']))):
            print(f"{i+1}. Generated: {results['generated_captions'][i]}")
            print(f"   Reference: {results['reference_captions'][i]}")
            print(f"   BLEU-1: {results['bleu1_scores'][i]:.3f}\n")
        
        return report

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset for evaluation
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
    
    # Load trained model
    checkpoint = torch.load('saved_models/best_teacher_model.pth', map_location=device)
    vocab_size = checkpoint['vocab_size']
    
    model = CaptioningTeacher(
        vocab_size=vocab_size,
        embed_size=512,
        num_heads=8,
        num_decoder_layers=4,
        dropout=0.15
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded! Validation loss was: {checkpoint['val_loss']:.4f}")
    
    # Create evaluator
    evaluator = CaptionEvaluator(model, dataset.vocab, device)
    
    # Evaluate on dataset
    print("Starting dataset evaluation...")
    results = evaluator.evaluate_on_dataset(eval_loader, num_samples=200)
    
    # Generate report
    report = evaluator.generate_evaluation_report(results)
    
    print(f"\nDetailed report saved to: evaluation_report.json")
    
    # Optional: Evaluate on specific images
    print("\n" + "="*60)
    print("SINGLE IMAGE EVALUATION")
    print("="*60)
    
    # Test on a few random images from your dataset
    test_images = []
    image_dir = "data/flickr8k/Images"
    if os.path.exists(image_dir):
        all_images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        test_images = random.sample(all_images, min(3, len(all_images)))
        
        for img_name in test_images:
            img_path = os.path.join(image_dir, img_name)
            print(f"\nEvaluating: {img_name}")
            caption = evaluator.evaluate_single_image(img_path, show_image=False)
            if caption:
                print(f"Caption: {caption}")
    
    return report

if __name__ == "__main__":
    main()