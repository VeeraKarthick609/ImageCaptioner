# src/evaluate.py

import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import time
import pandas as pd
from PIL import Image
import os
import numpy as np

# Import our models and data loader
from models import CaptioningTeacher
from student_model import CaptioningStudent # This is your CNN-LSTM student model
from data_loader import FlickrDataset

# Import evaluation metrics
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

def generate_caption_transformer(model, image_tensor, vocabulary, device, max_length=50):
    """
    Generates a caption for a single image using a Transformer-based model (the Teacher).
    This uses a greedy decoding approach.
    """
    model.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device).unsqueeze(0)
        
        # Encoder forward pass (once)
        image_features = model.encoder.forward_features(image_tensor)
        image_features = model.encoder_projection(image_features)
        image_features = image_features.permute(1, 0, 2)

        inputs = torch.tensor([vocabulary.stoi["<START>"]]).to(device)
        caption = []
        
        for _ in range(max_length):
            decoder_input = inputs.unsqueeze(1)
            
            embed_captions = model.embedding(decoder_input)
            embed_captions = model.pos_encoder(embed_captions)
            seq_len = decoder_input.size(0)
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            
            decoder_output = model.decoder(embed_captions, image_features, tgt_mask=tgt_mask)
            decoder_output = model.pre_output_norm(decoder_output)
            outputs = model.fc_out(decoder_output)
            
            predicted_idx = outputs[-1, 0, :].argmax()
            
            if predicted_idx.item() == vocabulary.stoi["<END>"]:
                break
            
            caption.append(vocabulary.itos[predicted_idx.item()])
            inputs = torch.cat((inputs, predicted_idx.unsqueeze(0)), dim=0)

    return " ".join(caption)

def generate_caption_lstm(model, image_tensor, vocabulary, device, max_length=50):
    """
    Generates a caption for a single image using the CNN-LSTM Student model.
    """
    model.eval()
    caption = []
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device).unsqueeze(0)
        
        # --- Encoder Forward Pass (to get initial hidden state) ---
        features = model.encoder(image_tensor)
        features = features.view(features.size(0), -1)
        features = model.relu(model.project_features(features))
        
        # Initialize LSTM state with image features
        num_layers = model.lstm.num_layers
        hidden = features.unsqueeze(0).repeat(num_layers, 1, 1)
        cell = features.unsqueeze(0).repeat(num_layers, 1, 1)

        # --- Word Generation Loop ---
        # Start with the <START> token
        inputs = torch.tensor([vocabulary.stoi["<START>"]]).to(device)
        
        for _ in range(max_length):
            # The LSTM expects input shape: (batch_size, 1, embed_size)
            # We add a dimension for seq_len=1
            embeddings = model.embedding(inputs).unsqueeze(1)
            
            # LSTM forward pass
            outputs, (hidden, cell) = model.lstm(embeddings, (hidden, cell))
            
            # Get the vocabulary scores
            outputs = model.fc_out(outputs.squeeze(1))
            predicted_idx = outputs.argmax(1)
            
            # Stop if we generate the <END> token
            if predicted_idx.item() == vocabulary.stoi["<END>"]:
                break

            # Append the word to our caption and set it as the next input
            caption.append(vocabulary.itos[predicted_idx.item()])
            inputs = predicted_idx

    return " ".join(caption)


def evaluate_model(model, df, vocab, transform, device):
    """
    Main evaluation function. Now handles both model types.
    """
    model.to(device)
    model.eval()

    predictions = {}
    references = {}
    
    for _, row in df.iterrows():
        img_id = row['image']
        caption_text = row['caption']
        if img_id not in references:
            references[img_id] = []
        references[img_id].append(str(caption_text))

    total_inference_time = 0
    num_images_processed = 0
    
    print(f"Generating captions for evaluation with {model.__class__.__name__}...")
    unique_img_ids = df['image'].unique()
    
    for img_id in tqdm(unique_img_ids):
        image_path = os.path.join("data/flickr8k/Images", img_id)
        if not os.path.exists(image_path): continue
            
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image)
        
        start_time = time.time()
        
        # --- UPDATED: CHOOSE THE CORRECT GENERATION FUNCTION ---
        if isinstance(model, CaptioningTeacher):
            generated_caption = generate_caption_transformer(model, image_tensor, vocab, device)
        elif isinstance(model, CaptioningStudent):
            generated_caption = generate_caption_lstm(model, image_tensor, vocab, device)
        else:
            raise TypeError("Model type not recognized for evaluation.")
            
        end_time = time.time()
        
        total_inference_time += (end_time - start_time)
        num_images_processed += 1
        predictions[img_id] = [generated_caption]

    print("Calculating performance scores...")
    final_refs = {i: references[img_id] for i, img_id in enumerate(predictions.keys())}
    final_preds = {i: predictions[img_id] for i, img_id in enumerate(predictions.keys())}

    scorers = [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]), (Meteor(), "METEOR"), (Rouge(), "ROUGE_L"), (Cider(), "CIDEr")]
    final_scores = {}
    for scorer, method in scorers:
        score, _ = scorer.compute_score(final_refs, final_preds)
        if isinstance(method, list):
            for i, m in enumerate(method):
                final_scores[m] = score[i]
        else:
            final_scores[method] = score

    avg_inference_time_ms = (total_inference_time / num_images_processed) * 1000
    return final_scores, avg_inference_time_ms


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading dataset metadata...")
    dataset = FlickrDataset(
        root_dir="data/flickr8k/Images",
        captions_file="data/flickr8k/captions_clean.csv",
        transform=transform
    )
    vocab = dataset.vocab
    vocab_size = len(vocab)
    df = dataset.df

    # --- Load Teacher Model (Transformer) ---
    print("\n--- Loading Teacher Model ---")
    teacher_checkpoint = torch.load('saved_models/best_teacher_model.pth', map_location=device)
    teacher_model = CaptioningTeacher(vocab_size=vocab_size, embed_size=512, num_heads=8, num_decoder_layers=4, dropout=0.15)
    teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
    
    # --- Load Student Model (CNN-LSTM) ---
    print("\n--- Loading CNN-LSTM Student Model ---")
    student_checkpoint = torch.load('saved_models/best_student_model_lstm_staged.pth', map_location=device)
    student_model = CaptioningStudent(
        vocab_size=vocab_size,
        embed_size=256,
        hidden_size=512,
        num_layers=2,
        dropout=0.5
    )
    student_model.load_state_dict(student_checkpoint['model_state_dict'])
    
    # --- Run Evaluations ---
    teacher_scores, teacher_time = evaluate_model(teacher_model, df, vocab, transform, device)
    student_scores, student_time = evaluate_model(student_model, df, vocab, transform, device)
    
    # --- Display Final Comparison Table ---
    print("\n\n" + "="*80)
    print("--- FINAL PROJECT RESULTS (TEACHER vs. CNN-LSTM STUDENT) ---".center(80))
    print("="*80)
    
    teacher_size = os.path.getsize('saved_models/best_teacher_model.pth') / (1024**2)
    student_size = os.path.getsize('saved_models/best_student_model_lstm_staged.pth') / (1024**2)

    print(f"{'Metric':<25} | {'Teacher (ViT-TF)':<20} | {'Student (CNN-LSTM)':<20} | {'Improvement/Retained'}")
    print("-" * 80)
    print(f"{'Model Size (MB)':<25} | {teacher_size:<20.1f} | {student_size:<20.1f} | {f'{teacher_size/student_size:.1f}x Smaller'}")
    print(f"{'Inference Time (ms/img)':<25} | {teacher_time:<20.1f} | {student_time:<20.1f} | {f'{teacher_time/student_time:.1f}x Faster' if student_time < teacher_time else ''}")
    print("-" * 80)
    print(f"{'CIDEr Score':<25} | {teacher_scores['CIDEr']:<20.3f} | {student_scores['CIDEr']:<20.3f} | {f'{(student_scores["CIDEr"]/teacher_scores["CIDEr"])*100:.1f}% Retained'}")
    print(f"{'BLEU-4 Score':<25} | {teacher_scores['Bleu_4']:<20.3f} | {student_scores['Bleu_4']:<20.3f} | {f'{(student_scores["Bleu_4"]/teacher_scores["Bleu_4"])*100:.1f}% Retained'}")
    print(f"{'BLEU-1 Score':<25} | {teacher_scores['Bleu_1']:<20.3f} | {student_scores['Bleu_1']:<20.3f} |")
    print(f"{'ROUGE_L Score':<25} | {teacher_scores['ROUGE_L']:<20.3f} | {student_scores['ROUGE_L']:<20.3f} |")
    print(f"{'METEOR Score':<25} | {teacher_scores['METEOR']:<20.3f} | {student_scores['METEOR']:<20.3f} |")
    print("="*80)

if __name__ == "__main__":
    main()