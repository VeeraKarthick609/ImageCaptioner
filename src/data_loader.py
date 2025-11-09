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
        img_path = os.path.join(self.root_dir, "Images", img_id)
        
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"[WARNING] Cannot find image at path: {img_path}, using placeholder")
            # Create a placeholder image (black image)
            img = Image.new('RGB', (224, 224), color='black')

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
def get_loader(root_folder, annotation_file='data/flickr8k/captions_clean.csv', transform=None, batch_size=32, num_workers=0, shuffle=True, pin_memory=True):
    # If no transform is provided, use default transforms for the model
    if transform is None:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    dataset = FlickrDataset(root_folder, annotation_file, transform=transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    
    # Ensure drop_last=True to avoid batch size mismatches during training
    # Also ensure batch_size doesn't exceed dataset size
    actual_batch_size = min(batch_size, len(dataset))
    
    # Use a more conservative batch size to reduce shape mismatches
    if actual_batch_size > 16:
        actual_batch_size = 16  # Use smaller batches for more stable training
    
    loader = DataLoader(dataset=dataset, batch_size=actual_batch_size, num_workers=num_workers,
                        shuffle=shuffle, pin_memory=pin_memory, collate_fn=Collate(pad_idx=pad_idx), drop_last=True)
    return loader, dataset
