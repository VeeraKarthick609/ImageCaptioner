# src/evaluate.py

import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import time
import pandas as pd
from PIL import Image
import os

# Import our models and data loader
from models import CaptioningTeacher
from student_model import CaptioningStudent
from data_loader import FlickrDataset # Use this to get vocab and file list

# Import evaluation metrics from the pycocoevalcap library
# Make sure you have it installed: pip install pycocoevalcap
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

def generate_caption(model, image_tensor, vocabulary, device, max_length=50):
    """
    Generates a caption for a single image using greedy search.
    """
    model.eval()
    
    with torch.no_grad():
        # --- Feature Extraction (Done ONCE) ---
        image_tensor = image_tensor.to(device).unsqueeze(0)
        
        # This part now adapts to the model type
        if isinstance(model, CaptioningTeacher):
            image_features = model.encoder.forward_features(image_tensor)
            image_features = model.encoder_projection(image_features)
            image_features = image_features.permute(1, 0, 2)
        else: # Student Model
            image_features = model.encoder.forward_features(image_tensor)
            if len(image_features.shape) == 4: # Handle CNN-like features
                B, C, H, W = image_features.shape
                image_features = image_features.view(B, C, H*W).permute(2, 0, 1)
            else: # Handle ViT-like features
                image_features = image_features.permute(1, 0, 2)
            image_features = model.encoder_projection(image_features)

        # --- Word Generation Loop ---
        inputs = torch.tensor([vocabulary.stoi["<START>"]]).to(device)
        caption = []
        
        for _ in range(max_length):
            decoder_input = inputs.unsqueeze(1) # Shape: (current_seq_len, 1)
            
            # Now we use the pre-computed image_features
            embed_captions = model.embedding(decoder_input)
            embed_captions = model.pos_encoder(embed_captions)

            seq_len = decoder_input.size(0)
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

            decoder_output = model.decoder(embed_captions, image_features, tgt_mask=tgt_mask)
            decoder_output = model.pre_output_norm(decoder_output)
            outputs = model.fc_out(decoder_output)
            
            # We want the prediction for the last word in the sequence
            predicted_idx = outputs[-1, 0, :].argmax()
            
            if predicted_idx.item() == vocabulary.stoi["<END>"]:
                break
            
            caption.append(vocabulary.itos[predicted_idx.item()])
            inputs = torch.cat((inputs, predicted_idx.unsqueeze(0)), dim=0)

    return " ".join(caption)


def evaluate_model(model, df, vocab, transform, device):
    """
    Main evaluation function.
    Calculates BLEU, CIDEr, etc., and measures average inference speed.
    """
    model.to(device)
    model.eval()

    # The evaluation tools require data in a specific dictionary format
    predictions = {}
    references = {}
    
    # Group all reference (ground-truth) captions by their image ID
    for _, row in df.iterrows():
        img_id = row['image']
        caption = row['caption']
        if img_id not in references:
            references[img_id] = []
        references[img_id].append(str(caption))

    # --- Inference and Timing Loop ---
    total_inference_time = 0
    num_images_processed = 0
    
    print(f"Generating captions for evaluation with {model.__class__.__name__}...")
    unique_img_ids = df['image'].unique()
    
    for img_id in tqdm(unique_img_ids):
        image_path = os.path.join("data/flickr8k/Images", img_id)
        if not os.path.exists(image_path):
            continue
            
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image)
        
        # Time the caption generation process
        start_time = time.time()
        generated_caption = generate_caption(model, image_tensor, vocab, device)
        end_time = time.time()
        
        total_inference_time += (end_time - start_time)
        num_images_processed += 1
        
        # The library expects a list of captions for each image
        predictions[img_id] = [generated_caption]

    # --- Score Calculation ---
    print("Calculating performance scores...")
    
    # The pycocoevalcap library expects integer keys, not string image IDs.
    # We create a mapping to handle this.
    final_refs = {i: references[img_id] for i, img_id in enumerate(predictions.keys())}
    final_preds = {i: predictions[img_id] for i, img_id in enumerate(predictions.keys())}

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    
    final_scores = {}
    for scorer, method in scorers:
        score, _ = scorer.compute_score(final_refs, final_preds)
        if isinstance(method, list):
            for i, m in enumerate(method):
                final_scores[m] = score[i]
        else:
            final_scores[method] = score

    # --- Final Report for this model ---
    avg_inference_time_ms = (total_inference_time / num_images_processed) * 1000
    
    return final_scores, avg_inference_time_ms


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading dataset metadata...")
    # Use the FlickrDataset class to conveniently get the vocabulary and DataFrame
    dataset = FlickrDataset(
        root_dir="data/flickr8k/Images",
        captions_file="data/flickr8k/captions_clean.csv",
        transform=transform
    )
    vocab = dataset.vocab
    vocab_size = len(vocab)
    df = dataset.df

    # --- Load Teacher Model ---
    print("\n--- Loading Teacher Model ---")
    teacher_checkpoint = torch.load('saved_models/best_teacher_model.pth', map_location=device)
    teacher_model = CaptioningTeacher(
        vocab_size=vocab_size,
        embed_size=512,
        num_heads=8,
        num_decoder_layers=4,
        dropout=0.15
    )
    teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
    
    # --- Load Student Model ---
    print("\n--- Loading Student Model ---")
    student_checkpoint = torch.load('saved_models/best_student_model.pth', map_location=device)
    student_model = CaptioningStudent(
        vocab_size=vocab_size,
        # IMPORTANT: Use the parameters of the student model you actually trained
        # (e.g., the slightly larger one from Strategy 2, or the original)
        embed_size=256,
        num_heads=4,
        num_decoder_layers=2,
        dropout=0.1
    )
    student_model.load_state_dict(student_checkpoint['model_state_dict'])
    
    # --- Run Evaluations ---
    teacher_scores, teacher_time = evaluate_model(teacher_model, df, vocab, transform, device)
    student_scores, student_time = evaluate_model(student_model, df, vocab, transform, device)
    
    # --- Display Final Comparison Table ---
    print("\n\n" + "="*80)
    print("--- FINAL PROJECT RESULTS ---".center(80))
    print("="*80)
    
    teacher_size = os.path.getsize('saved_models/best_teacher_model.pth') / (1024**2)
    student_size = os.path.getsize('saved_models/best_student_model.pth') / (1024**2)

    print(f"{'Metric':<25} | {'Teacher Model':<20} | {'Student Model':<20} | {'Improvement/Retained'}")
    print("-" * 80)
    print(f"{'Model Size (MB)':<25} | {teacher_size:<20.1f} | {student_size:<20.1f} | {f'{teacher_size/student_size:.1f}x Smaller'}")
    print(f"{'Inference Time (ms/img)':<25} | {teacher_time:<20.1f} | {student_time:<20.1f} | {f'{teacher_time/student_time:.1f}x Faster'}")
    print("-" * 80)
    print(f"{'CIDEr Score':<25} | {teacher_scores['CIDEr']:<20.3f} | {student_scores['CIDEr']:<20.3f} | {f'{(student_scores["CIDEr"]/teacher_scores["CIDEr"])*100:.1f}% Retained'}")
    print(f"{'BLEU-4 Score':<25} | {teacher_scores['Bleu_4']:<20.3f} | {student_scores['Bleu_4']:<20.3f} | {f'{(student_scores["Bleu_4"]/teacher_scores["Bleu_4"])*100:.1f}% Retained'}")
    print(f"{'BLEU-1 Score':<25} | {teacher_scores['Bleu_1']:<20.3f} | {student_scores['Bleu_1']:<20.3f} |")
    print(f"{'ROUGE_L Score':<25} | {teacher_scores['ROUGE_L']:<20.3f} | {student_scores['ROUGE_L']:<20.3f} |")
    print(f"{'METEOR Score':<25} | {teacher_scores['METEOR']:<20.3f} | {student_scores['METEOR']:<20.3f} |")
    print("="*80)

if __name__ == "__main__":
    main()