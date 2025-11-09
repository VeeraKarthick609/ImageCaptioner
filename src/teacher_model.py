# src/models.py

import torch
import torch.nn as nn
import timm
import math

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal Positional Encoding.
    From PyTorch tutorial: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CaptioningTeacher(nn.Module):
    def __init__(self, vocab_size, embed_size=384, num_heads=12, num_decoder_layers=6, dropout=0.1):
        super(CaptioningTeacher, self).__init__()
        
        # --- ENCODER ---
        # Use a smaller, more efficient ViT model
        self.encoder = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
        
        # Get the actual feature dimension from the model
        encoder_dim = self.encoder.num_features  # Should be 384 for vit_small
        
        # Unfreeze some encoder layers for fine-tuning
        # Keep early layers frozen, allow later layers to adapt
        for i, (name, param) in enumerate(self.encoder.named_parameters()):
            if 'blocks.8' in name or 'blocks.9' in name or 'blocks.10' in name or 'blocks.11' in name or 'norm' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Project encoder features to decoder dimension if they don't match
        self.encoder_projection = nn.Linear(encoder_dim, embed_size) if encoder_dim != embed_size else nn.Identity()
        
        # --- DECODER ---
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Initialize embeddings with smaller variance
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        self.pos_encoder = PositionalEncoding(d_model=embed_size, dropout=dropout)
        
        # Use fewer decoder layers to reduce overfitting
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size, 
            nhead=num_heads, 
            dim_feedforward=embed_size * 2,  # Reduced from default 4x
            dropout=dropout,
            batch_first=False
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Add layer normalization before final projection
        self.pre_output_norm = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        
        # Initialize output layer with smaller weights
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, images, captions):
        # --- ENCODER FORWARD ---
        with torch.amp.autocast("cuda"):
            image_features = self.encoder.forward_features(images)  # Shape: (batch_size, 197, encoder_dim)
            image_features = self.encoder_projection(image_features)  # Shape: (batch_size, 197, embed_size)
            image_features = image_features.permute(1, 0, 2)  # Shape: (197, batch_size, embed_size)

        # --- DECODER FORWARD ---
        embed_captions = self.embedding(captions)
        embed_captions = self.pos_encoder(embed_captions)
        
        # Create causal mask
        seq_len = captions.size(0)
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=captions.device), diagonal=1).bool()
        
        # Decoder forward pass
        decoder_output = self.decoder(
            tgt=embed_captions, 
            memory=image_features, 
            tgt_mask=tgt_mask
        )
        
        # Apply normalization and dropout before final projection
        decoder_output = self.pre_output_norm(decoder_output)
        decoder_output = self.dropout(decoder_output)
        output = self.fc_out(decoder_output)
        
        return output
    
    def caption_image(
            self,
            image,
            vocabulary,
            max_length: int = 20,
            beam_size: int = 5,
            length_penalty: float = 0.6,     # 0=no penalty, ~0.6–1.0 common (GNMT: ((5+L)/6)^α)
            early_stopping: bool = True,
            num_return_sequences: int = 1,   # up to beam_size
        ):
        """
        Beam search captioning.
        Returns a list of strings (size=num_return_sequences).
        """

        self.eval()
        device = next(self.parameters()).device
        start_id = vocabulary.stoi.get("<START>", vocabulary.stoi.get("<UNK>"))
        end_id   = vocabulary.stoi.get("<END>", None)
        assert start_id is not None, "Vocabulary must define <START> or <UNK>."
        if num_return_sequences > beam_size:
            num_return_sequences = beam_size

        # 1) Encode image once
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(device)

            memory = self.encoder.forward_features(image)           # (1, L, E)
            memory = self.encoder_projection(memory)                # (1, L, E)
            memory = memory.permute(1, 0, 2)                        # (L, 1, E)
            L, _, E = memory.shape

            # Repeat memory across beams so each beam has its own slot
            memory = memory.expand(L, beam_size, E).contiguous()    # (L, B, E)

            # 2) Beam init
            # sequences: list of lists of token ids, start with <START>
            seqs = torch.full((1, beam_size), start_id, dtype=torch.long, device=device)  # (t=1, B)
            # scores are log-probs; only beam 0 is “active” at step 0
            scores = torch.full((beam_size,), float("-inf"), device=device)
            scores[0] = 0.0

            # keep finished hypotheses here: list of (tokens_tensor, normalized_score)
            finished = []

            # 3) Decoding loop
            for step in range(1, max_length + 1):
                # Embed current prefixes
                tgt = self.embedding(seqs)              # (t, B, E)
                tgt = self.pos_encoder(tgt)             # (t, B, E)

                # causal mask for current length t
                t = tgt.size(0)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(t).to(device)

                # decoder forward
                dec = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)   # (t, B, E)
                dec = self.pre_output_norm(dec)
                logits = self.fc_out(dec[-1])           # last step only → (B, V)

                # log-probabilities for expansion
                log_probs = torch.log_softmax(logits, dim=-1)  # (B, V)

                # combine with running scores to get all candidate scores
                candidate_scores = scores.unsqueeze(1) + log_probs  # (B, V)

                # flatten and select top beam_size next candidates
                topk_scores, topk_ids = torch.topk(candidate_scores.view(-1), k=beam_size, dim=-1)
                next_beam_ids = topk_ids // log_probs.size(-1)      # origin beam index (B')
                next_token_ids = (topk_ids % log_probs.size(-1)).long()  # new token (B')

                # build new sequences
                new_seqs = []
                new_scores = []
                new_memory = memory  # same ref; memory already has B beams

                for i in range(beam_size):
                    origin = int(next_beam_ids[i])
                    token = int(next_token_ids[i])
                    score = float(topk_scores[i])

                    # append token to origin sequence
                    # seqs: (t, B) → take origin column, cat new token row
                    new_seq_i = torch.cat(
                        [seqs[:, origin], torch.tensor([token], device=device, dtype=torch.long)],
                        dim=0
                    )  # (t+1,)

                    # if EOS predicted → finalize this hypothesis
                    if end_id is not None and token == end_id:
                        # length normalization (GNMT-style)
                        Lhyp = new_seq_i.size(0)  # includes <START> and <END>
                        lp = ((5.0 + Lhyp) / 6.0) ** length_penalty if length_penalty > 0 else 1.0
                        finished.append((new_seq_i, score / lp))
                    else:
                        new_seqs.append(new_seq_i.unsqueeze(1))  # keep as (t+1, 1)
                        new_scores.append(score)

                # early stop if all beams ended
                if early_stopping and len(new_seqs) == 0:
                    break

                # if we still have live beams, stack them back to (t+1, B)
                if len(new_seqs) > 0:
                    # we may have fewer than beam_size live beams if some ended; refill by keeping best
                    # pad by taking top remaining candidates if needed (rare)
                    B_live = len(new_seqs)
                    seqs = torch.cat(new_seqs, dim=1)                 # (t+1, B_live)
                    scores = torch.tensor(new_scores, device=device)  # (B_live,)

                    # To keep constant beam_size width for the next step, we can re-expand memory to B_live.
                    # Here memory is (L, B, E) and B may shrink; simplest is to slice the needed beams.
                    # But since memory is the same for all beams (same image), we can just expand again:
                    memory = memory[:, :B_live, :]                     # (L, B_live, E)

                    # If B_live < beam_size, we’ll carry fewer beams forward—fine. We still select topk from what remains.
                    beam_size = B_live
                else:
                    # nothing live; will exit loop (because early_stopping handled above)
                    break

            # 4) If no finished hypotheses, finalize from live beams
            if len(finished) == 0:
                for b in range(seqs.size(1)):
                    Lhyp = seqs.size(0)
                    lp = ((5.0 + Lhyp) / 6.0) ** length_penalty if length_penalty > 0 else 1.0
                    finished.append((seqs[:, b], float(scores[b]) / lp))

            # 5) Sort finished by normalized score (desc) and return top-k
            finished.sort(key=lambda x: x[1], reverse=True)
            outs = []
            for i in range(min(num_return_sequences, len(finished))):
                tokens = finished[i][0].tolist()
                # drop the leading <START>, and any trailing <END>
                if len(tokens) > 0 and tokens[0] == start_id:
                    tokens = tokens[1:]
                if end_id is not None and end_id in tokens:
                    end_pos = tokens.index(end_id)
                    tokens = tokens[:end_pos]
                words = [vocabulary.itos[idx] for idx in tokens]
                outs.append(" ".join(words))

            return outs

    # def caption_image(self, image, vocabulary, max_length=20):  # Reduced max length
    #     """
    #     Generate caption using beam search for better quality
    #     """
    #     self.eval()
    #     device = next(self.parameters()).device
        
    #     with torch.no_grad():
    #         # Get image features
    #         if image.dim() == 3:
    #             image = image.unsqueeze(0)
    #         image = image.to(device)
            
    #         image_features = self.encoder.forward_features(image)
    #         image_features = self.encoder_projection(image_features)
    #         image_features = image_features.permute(1, 0, 2)

    #         # Simple greedy decoding
    #         result_caption = []
    #         inputs = torch.tensor([[vocabulary.stoi.get("<START>", vocabulary.stoi["<UNK>"])]]).to(device).T
            
    #         for _ in range(max_length):
    #             embed_captions = self.embedding(inputs)
    #             embed_captions = self.pos_encoder(embed_captions)
                
    #             seq_len = inputs.size(0)
    #             tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
                
    #             decoder_output = self.decoder(embed_captions, image_features, tgt_mask=tgt_mask)
    #             decoder_output = self.pre_output_norm(decoder_output)
    #             output = self.fc_out(decoder_output)
                
    #             # Get the most likely next word
    #             predicted_idx = output[-1, 0, :].argmax().item()
                
    #             # Stop if we predict <END>
    #             if vocabulary.itos[predicted_idx] == "<END>":
    #                 break
                
    #             result_caption.append(predicted_idx)
                
    #             # Add predicted token to input sequence
    #             new_token = torch.tensor([[predicted_idx]]).to(device)
    #             inputs = torch.cat([inputs, new_token], dim=0)
        
    #     return [vocabulary.itos[idx] for idx in result_caption]
     