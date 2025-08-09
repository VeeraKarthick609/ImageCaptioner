# src/student_model.py

import torch
import torch.nn as nn
import torchvision.models as models

class CaptioningStudent(nn.Module):
    """
    A classic and simple Student Model using a CNN (ResNet) encoder 
    and an LSTM decoder.
    """
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_layers=1, dropout=0.5):
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            embed_size (int): Dimension of the word embeddings.
            hidden_size (int): Dimension of the LSTM hidden states.
            num_layers (int): Number of layers in the LSTM.
        """
        super(CaptioningStudent, self).__init__()
        
        # --- CNN ENCODER ---
        # We use a pre-trained ResNet-50.
        # We'll remove the final fully connected layer (the classifier)
        # to get the image features.
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1] # Remove the last fc layer
        self.encoder = nn.Sequential(*modules)
        
        # The output of ResNet-50 is 2048-dimensional. We need a linear layer
        # to project this feature vector to the LSTM's hidden state dimension.
        self.project_features = nn.Linear(resnet.fc.in_features, hidden_size)
        self.relu = nn.ReLU()
        
        # Freeze the encoder parameters. We only use it for feature extraction.
        for param in self.encoder.parameters():
            param.requires_grad_(False)
            
        # --- LSTM DECODER ---
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # The LSTM takes word embeddings as input and outputs hidden states.
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # The final linear layer maps the LSTM's hidden state to vocabulary scores.
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, images, captions):
        """
        The forward pass for training with teacher forcing.
        """
        # --- ENCODER FORWARD ---
        # 1. Extract features from the image
        features = self.encoder(images) # Shape: (batch_size, 2048, 1, 1)
        features = features.view(features.size(0), -1) # Shape: (batch_size, 2048)
        
        # 2. Project features to serve as the initial LSTM state
        features = self.relu(self.project_features(features)) # Shape: (batch_size, hidden_size)
        
        # The LSTM expects its hidden state as a tuple (h_0, c_0)
        # We will use the image features for both.
        # Shape of h_0 and c_0: (num_layers, batch_size, hidden_size)
        num_layers = self.lstm.num_layers
        h_0 = features.unsqueeze(0).repeat(num_layers, 1, 1)
        c_0 = features.unsqueeze(0).repeat(num_layers, 1, 1)

        # --- DECODER FORWARD ---
        # 1. Embed the captions (we don't need the <END> token for input)
        # Our dataloader provides captions of shape (seq_len, batch), we need (batch, seq_len)
        captions = captions.permute(1, 0)
        captions_input = captions[:, :-1]
        
        embeddings = self.embedding(captions_input) # Shape: (batch_size, seq_len-1, embed_size)
        
        # 2. Pass embeddings through the LSTM
        # The initial hidden and cell states are our image features
        outputs, _ = self.lstm(embeddings, (h_0, c_0)) # Shape: (batch_size, seq_len-1, hidden_size)
        
        # 3. Apply dropout and the final linear layer
        outputs = self.dropout(outputs)
        outputs = self.fc_out(outputs) # Shape: (batch_size, seq_len-1, vocab_size)
        
        # We need to reshape for the loss function to match the Transformer output
        # (seq_len, batch, vocab_size)
        outputs = outputs.permute(1, 0, 2)
        
        return outputs

    def get_image_features(self, images):
        """
        A helper function to be used during feature distillation.
        This isolates the feature extraction part.
        """
        with torch.no_grad():
            features = self.encoder(images)
            features = features.view(features.size(0), -1)
        return features