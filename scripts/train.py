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
    def __init__(slef, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.long), torch.tensor(self.outputs[idx], dtype=toch.long)
    
# Fast Weights Model
class FastWeightsModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, seq_length):
        super(FastWeightsModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.controller = nn.Linear(embed_dim * seq_length, hidden_dim)
        self.fast_weights = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embed = self.embedding(x).view(x-size(0), -1) # Flatten sequence into one vector
        update = self.controller(embed)
        self.fast_weights = self.fast_weights + torch.outer(update, update)
        output = torch.matmult(self.fast_weights, update.unsqueeze(1).squeeze(1))
        return self.output_layer(output)
    
if __name__ == "__main__":
    # Load preprocessed data
    with open("../data/processed/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("../data/preprocessed/sequences.pkl", "rb") as f:
        data = pickle.load(f)

    char_to_idx = tokenizer["char_to_idx"]
    idx_to_char = tokenizer["idx_to_char"]
    input_seqs, output_seqs = data["inputs"], data["outputs"]

    #Hyperparameters
    vocab_size = len(char_to_idk)
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

    # Training loop
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
    os.makedirs("../models", exist_ok=True)
    torch.save(model.state_dict(), "../models/fast_weights_model.pth")
























