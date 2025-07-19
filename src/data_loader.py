# src/data_loader.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import spacy

# --- Setup Spacy ---
try:
    spacy_eng = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy 'en_core_web_sm' model not found. Please run 'python -m spacy download en_core_web_sm'")
    exit()

# --- Vocabulary Class ---
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(str(text))]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] = frequencies.get(word, 0) + 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]

# --- Dataset Class ---
class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        print(f"[Dataset __init__] Trying to load captions from: {captions_file}")
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        print(f"[Dataset __init__] DataFrame loaded. It has {len(self.df)} rows.")
        
        if len(self.df) == 0:
            raise ValueError("The DataFrame is empty. Please check if captions_clean.csv was created correctly.")

        self.transform = transform
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img_path = os.path.join(self.root_dir, img_id)
        
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"[ERROR] Cannot find image at path: {img_path}")
            raise

        if self.transform:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<START>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<END>"])

        return img, torch.tensor(numericalized_caption)

# --- Collate Function ---
class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)
        return imgs, targets

# --- Loader Function ---
def get_loader(root_folder, annotation_file, transform, batch_size=32, num_workers=0, shuffle=True, pin_memory=True):
    dataset = FlickrDataset(root_folder, annotation_file, transform=transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers,
                        shuffle=shuffle, pin_memory=pin_memory, collate_fn=Collate(pad_idx=pad_idx))
    return loader, dataset

if __name__ == "__main__":
    # Define paths
    original_captions_file = "data/flickr8k/captions.txt"
    cleaned_captions_file = "data/flickr8k/captions_clean.csv"
    root_image_folder = "data/flickr8k/Images"

    # --- Step 1: Force creation of the clean CSV ---
    print("[DEBUG] Starting CSV creation process.")
    
    if os.path.exists(cleaned_captions_file):
        os.remove(cleaned_captions_file)
        print(f"[DEBUG] Removed old version of {cleaned_captions_file}.")

    data_for_df = {'image': [], 'caption': []}
    lines_parsed = 0
    try:
        with open(original_captions_file, 'r', encoding='utf-8') as f:
            # We will also skip the header line "image,caption"
            for i, line in enumerate(f.readlines()[1:]):
                line = line.strip()
                
                # --- THIS IS THE FIX ---
                parts = line.strip().split(',', 1)
                # --- END OF FIX ---

                if i < 5:
                    print(f"[DEBUG] Line {i+1}: '{line[:40]}...' -> Split into {len(parts)} parts.")

                if len(parts) == 2:
                    image_id_full, caption = parts
                    # In this format, the image_id doesn't have the #0, #1 part, so we can use it directly
                    data_for_df['image'].append(image_id_full)
                    data_for_df['caption'].append(caption)
                    lines_parsed += 1
    except FileNotFoundError:
        print(f"[FATAL ERROR] The original captions file was not found at: {original_captions_file}")
        exit()

    print(f"[DEBUG] Parsing complete. Total lines successfully parsed: {lines_parsed}")

    if lines_parsed > 0:
        df = pd.DataFrame(data_for_df)
        df.to_csv(cleaned_captions_file, index=False)
        print(f"[DEBUG] Successfully saved {cleaned_captions_file}.")
    else:
        print("[FATAL ERROR] No lines were parsed from the captions file.")
        exit()

    # --- Step 2: Test the DataLoader ---
    print("\n" + "="*20 + " TESTING DATALOADER " + "="*20)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # We can use the less-buggy debug version of the script now
    # Find the full script from the previous response if this fails
    data_loader, dataset = get_loader(
        root_folder=root_image_folder,
        annotation_file=cleaned_captions_file,
        transform=transform,
        batch_size=4
    )

    print(f"\n[SUCCESS] DataLoader created.")
    print(f"Vocabulary Size: {len(dataset.vocab)}")

    imgs, captions = next(iter(data_loader))
    print("\n--- One Batch ---")
    print("Images shape:", imgs.shape)
    print("Captions shape:", captions.shape)
    
    caption_text = [dataset.vocab.itos[token.item()] for token in captions[:, 0]]
    print("\nExample Caption (text):")
    print(" ".join(caption_text))