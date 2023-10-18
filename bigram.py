#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as Func

# Set parameters
num_batches = 32
seq_length = 8
iterations = 3000
assessment_interval = 300
alpha = 0.01
processing_device = 'cuda' if torch.cuda.is_available() else 'cpu'
evaluation_iterations = 200

# Set a random seed for reproducibility
torch.manual_seed(42)

# Download data (i.e., Shakespeare text) before running
with open('shakespeare.txt', 'r', encoding='utf-8') as file:
    content = file.read()

# Character mappings
unique_chars = sorted(set(content))
vocab_length = len(unique_chars)
char_to_index = {char: index for index, char in enumerate(unique_chars)}
index_to_char = {index: char for index, char in enumerate(unique_chars)}

# Encoding and decoding functions
encode_text = lambda string: [char_to_index[char] for char in string]
decode_text = lambda code: ''.join([index_to_char[index] for index in code])

# Prepare data
encoded_text = torch.tensor(encode_text(content), dtype=torch.long)
split_size = int(0.9 * len(encoded_text))
training_data, validation_data = encoded_text[:split_size], encoded_text[split_size:]

# Batch preparation
def prepare_batch(dataset):
    random_start = torch.randint(len(dataset) - seq_length, (num_batches,))
    input_sequence = torch.stack([dataset[start:start+seq_length] for start in random_start])
    target_sequence = torch.stack([dataset[start+1:start+seq_length+1] for start in random_start])
    return input_sequence.to(processing_device), target_sequence.to(processing_device)

# Loss estimation
@torch.no_grad()
def evaluate_loss():
    results = {}
    model.eval()
    for data_type in ['training', 'validation']:
        data = training_data if data_type == 'training' else validation_data
        all_losses = torch.zeros(evaluation_iterations)
        for num in range(evaluation_iterations):
            input_seq, target_seq = prepare_batch(data)
            predictions, current_loss = model(input_seq, target_seq)
            all_losses[num] = current_loss.item()
        results[data_type] = all_losses.mean().item()
    model.train()
    return results

# Define the model
class BigramModel(nn.Module):
    def __init__(self, vocabulary_size):
        super(BigramModel, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, vocabulary_size)

    def forward(self, sequences, labels=None):
        prediction_scores = self.embedding(sequences)
        loss = None if labels is None else Func.cross_entropy(prediction_scores.view(-1, vocab_length), labels.view(-1))
        return prediction_scores, loss

    def predict(self, sequences, max_length):
        with torch.no_grad():
            for _ in range(max_length):
                scores, _ = self(sequences)
                probabilities = Func.softmax(scores[:, -1], dim=-1)
                next_char = torch.multinomial(probabilities, 1)
                sequences = torch.cat((sequences, next_char), dim=1)
        return sequences

# Initialize model and optimizer
model = BigramModel(vocab_length).to(processing_device)
optimizer = torch.optim.AdamW(model.parameters(), lr=alpha)

# Training loop
for step in range(iterations):
    if step % assessment_interval == 0:
        loss_metrics = evaluate_loss()
        print(f"Iteration {step}: Training loss {loss_metrics['training']:.4f}, Validation loss {loss_metrics['validation']:.4f}")

    input_batch, target_batch = prepare_batch(training_data)
    _, batch_loss = model(input_batch, target_batch)

    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

# Text generation
start_sequence = torch.zeros((1, 1), dtype=torch.long, device=processing_device)
generated_sequence = model.predict(start_sequence, 500)
print(decode_text(generated_sequence[0].cpu().tolist()))

