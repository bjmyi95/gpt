#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from torch.nn import functional as F

# Setting up parameters
batch_config = 64
sequence_length = 256
iterations = 5000
check_interval = 500
adapt_rate = 3e-4
execution_mode = 'cuda' if torch.cuda.is_available() else 'cpu'
validation_steps = 200
embed_size = 384
attention_heads = 6
transformer_blocks = 6
dropout_rate = 0.2

torch.manual_seed(1337)

# Load data
# Use wget or any other tool to download your dataset
with open('input.txt', 'r', encoding='utf-8') as dataset:
    all_text = dataset.read()

# Preprocessing
unique_chars = sorted(list(set(all_text)))
vocab_length = len(unique_chars)
char_to_index = { ch:i for i, ch in enumerate(unique_chars) }
index_to_char = { i:ch for i, ch in enumerate(unique_chars) }
text_encoding = lambda s: [char_to_index[c] for c in s]
text_decoding = lambda l: ''.join([index_to_char[i] for i in l])

# Preparing datasets
encoded_text = torch.tensor(text_encoding(all_text), dtype=torch.long)
split_size = int(0.9 * len(encoded_text))
train_set = encoded_text[:split_size]
validation_set = encoded_text[split_size:]

# Batch preparation
def prepare_batch(partition):
    dataset = train_set if partition == 'train' else validation_set
    random_start = torch.randint(len(dataset) - sequence_length, (batch_config,))
    input_sequence = torch.stack([dataset[i:i+sequence_length] for i in random_start])
    target_sequence = torch.stack([dataset[i+1:i+sequence_length+1] for i in random_start])
    return input_sequence.to(execution_mode), target_sequence.to(execution_mode)

@torch.no_grad()
def validate_model():
    results = {}
    model.eval()
    for phase in ['train', 'val']:
        total_loss = torch.zeros(validation_steps)
        for step in range(validation_steps):
            inputs, labels = prepare_batch(phase)
            predictions, current_loss = model(inputs, labels)
            total_loss[step] = current_loss.item()
        results[phase] = total_loss.mean()
    model.train()
    return results

# Transformer components
class AttentionHead(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.k_layer = nn.Linear(embed_size, head_dim, bias=False)
        self.q_layer = nn.Linear(embed_size, head_dim, bias=False)
        self.v_layer = nn.Linear(embed_size, head_dim, bias=False)
        self.attention_mask = torch.tril(torch.ones(sequence_length, sequence_length))
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, input_tensor):
        batch_size, seq_len, _ = input_tensor.shape
        keys = self.k_layer(input_tensor)
        queries = self.q_layer(input_tensor)
        weights = queries @ keys.transpose(-2, -1) * (keys.shape[-1] ** -0.5)
        weights = weights.masked_fill(self.attention_mask[:seq_len, :seq_len] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout_layer(weights)
        values = self.v_layer(input_tensor)
        attention_output = weights @ values
        return attention_output

class MultiHeadedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_heads = nn.ModuleList([AttentionHead(embed_size // attention_heads) for _ in range(attention_heads)])
        self.projection_layer = nn.Linear(embed_size, embed_size)
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, input_tensor):
        attention_outputs = torch.cat([head(input_tensor) for head in self.attention_heads], dim=-1)
        attention_outputs = self.dropout_layer(self.projection_layer(attention_outputs))
        return attention_outputs

class PositionWiseFeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff_network = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout_rate),
        )

    def forward(self, input_tensor):
        return self.ff_network(input_tensor)

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention_layer = MultiHeadedAttention()
        self.feed_forward_layer = PositionWiseFeedForward()
        self.first_norm_layer = nn.LayerNorm(embed_size)
        self.second_norm_layer = nn.LayerNorm(embed_size)

    def forward(self, input_tensor):
        attention_input = self.first_norm_layer(input_tensor)
        attention_output = self.self_attention_layer(attention_input)
        intermediate_output = input_tensor + attention_output
        ff_input = self.second_norm_layer(intermediate_output)
        ff_output = self.feed_forward_layer(ff_input)
        transformer_output = intermediate_output + ff_output
        return transformer_output

# Complete model
class CustomLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_length, embed_size)
        self.position_embeddings = nn.Embedding(sequence_length, embed_size)
        self.transformer_layers = nn.Sequential(*[TransformerBlock() for _ in range(transformer_blocks)])
        self.final_norm_layer = nn.LayerNorm(embed_size)
        self.prediction_head = nn.Linear(embed_size, vocab_length)
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_indices, targets=None):
        batch_size, seq_len = input_indices.shape
        token_embeds = self.token_embeddings(input_indices)
        position_embeds = self.position_embeddings(torch.arange(seq_len, device=execution_mode))
        transformer_input = token_embeds + position_embeds
        transformer_output = self.transformer_layers(transformer_input)
        transformer_output = self.final_norm_layer(transformer_output)
        logits = self.prediction_head(transformer_output)

        loss = None
        if targets is not None:
            logits_flat = logits.view(batch_size * seq_len, -1)
            targets_flat = targets.view(batch_size * seq_len)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def sample_text(self, seed_text, generation_length):
        model.eval()
        generated_text = seed_text
        for _ in range(generation_length):
            encoded_input = torch.tensor([text_encoding(generated_text)], dtype=torch.long, device=execution_mode)
            predictions, _ = self.forward(encoded_input)
            last_pred = predictions[0, -1, :]
            probabilities = F.softmax(last_pred, dim=-1)
            next_char_idx

