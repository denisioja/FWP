"""
This script trains the Fast Weights model on preprocessed Shakespeare data.
It handles data loading, model definition, training loop, and evaluation.
The goal is to predict the next character given a sequence of previosu characters.

Key functionalities:
- Load preprocessed data and tokenizer.
- Define the Fast Weights model architecture.
- Implement the training loop with cross-entropy loss.
- Evaluate model performance on validation data.
- Save trained model.
"""

import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Custom Dataset
class ShakespeareDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.long), torch.tensor(self.outputs[idx], dtype=torch.long)
# 1
# # Fast Weights Model
# class FastWeightsModel(nn.Module):
#     def __init__(self, vocab_size, embed_dim, hidden_dim, seq_length):
#         super(FastWeightsModel, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
#         self.controller = nn.Linear(embed_dim * seq_length, hidden_dim)
#         self.fast_weights = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))
#         self.transform = nn.Linear(seq_length * embed_dim, hidden_dim)
#         self.output_layer = nn.Linear(hidden_dim, vocab_size)

#     def forward(self, x):
#         # Embedding: (batch_size, seq_length, embedding_dim) -> (batch_size, seq_length * embedding_dim)
#         embed = self.embedding(x).view(x.size(0), -1)
#         transformed_embed = self.transform(embed)  # Transform to hidden_dim

#         # Controller: Produces update vector with the same dimension as the hidden_dim
#         update = self.controller(embed)  # (batch_size, hidden_dim)

#         # Debug: Ensure dimensions match
#         print(f"Embed size: {embed.shape}, Transformed Embed size: {transformed_embed.shape}, Update size: {update.shape}")

#         # Outer product: Compute the fast weight updates (adjusting to square matrices)
#         flattened_update = update.view(-1, update.size(1))  # Ensure correct shape for outer product
#         updated_fast_weights = self.fast_weights + torch.bmm(flattened_update.unsqueeze(2), flattened_update.unsqueeze(1)).mean(0)
        
#         # Update the fast weights
#         self.fast_weights = nn.Parameter(updated_fast_weights)

#         # Fast network: Apply updated fast weights for predictions
#         output = torch.matmul(transformed_embed, self.fast_weights)  # (batch_size, hidden_dim)
#         output = self.output_layer(output)  # Final linear transformation

#         return output

# 2
class FastWeightsModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, max_seq_length):
        super(FastWeightsModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.controller = nn.Linear(embed_dim * max_seq_length, hidden_dim)
        self.fast_weights = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))
        self.transform = nn.Linear(max_seq_length * embed_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self.max_seq_length = max_seq_length

    def forward(self, x):
        # Embedding: (batch_size, seq_length, embedding_dim) -> (batch_size, seq_length * embedding_dim)
        embed = self.embedding(x)
        embed = embed.view(x.size(0), -1)

        # Controller: Produces update vector with the same dimension as the hidden_dim
        update = self.controller(embed)

        # Outer product: Compute the fast weight updates
        flattened_update = update.view(-1, update.size(1))
        updated_fast_weights = self.fast_weights + torch.bmm(flattened_update.unsqueeze(2), flattened_update.unsqueeze(1)).mean(0)

        # Update the fast weights
        self.fast_weights = nn.Parameter(updated_fast_weights)

        # Fast network: Apply updated fast weights for predictions
        transformed_embed = self.transform(embed)
        output = torch.matmul(transformed_embed, self.fast_weights)
        output = self.output_layer(output)

        return output


if __name__ == "__main__":
    # Load preprocessed data
    with open("C:/Users/Denis/Desktop/FWP/data/processed/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("C:/Users/Denis/Desktop/FWP/data/processed/sequences.pkl", "rb") as f:
        data = pickle.load(f)

    char_to_idx = tokenizer["char_to_idx"]
    idx_to_char = tokenizer["idx_to_char"]
    input_seqs, output_seqs = data["inputs"], data["outputs"]

    # Hyperparameters
    vocab_size = len(char_to_idx)
    embed_dim = 64
    hidden_dim = 128
    seq_length = 100
    batch_size = 64
    epochs = 10
    lr = 0.001

    # Datasets and DataLoaders
    dataset = ShakespeareDataset(input_seqs, output_seqs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, Loss, Optimizer
    model = FastWeightsModel(vocab_size, embed_dim, hidden_dim, seq_length)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training Loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

    # Save Model
    os.makedirs("C:/Users/Denis/Desktop/FWP/models", exist_ok=True)
    torch.save(model.state_dict(), "C:/Users/Denis/Desktop/FWP/models/fast_weights_model.pth")
























