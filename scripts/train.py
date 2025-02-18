import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from config import *  # Import hyperparameters

# Define Dataset class
class ShakespeareDataset(Dataset):
    def __init__(self, input_sequences, output_sequences):
        self.input_sequences = torch.tensor(input_sequences, dtype=torch.long)
        self.output_sequences = torch.tensor(output_sequences, dtype=torch.long)

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        return self.input_sequences[idx], self.output_sequences[idx]

# Define Fast Weights Model
class FastWeightsModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, seq_length):
        super(FastWeightsModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Take last output for prediction
        return x

if __name__ == "__main__":
    # Load tokenizer and training data
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    with open(SEQUENCES_PATH, "rb") as f:
        data = pickle.load(f)

    char_to_idx = tokenizer["char_to_idx"]
    idx_to_char = tokenizer["idx_to_char"]
    input_seqs, output_seqs = data["inputs"], data["outputs"]

    # Update dynamic hyperparameters
    VOCAB_SIZE = len(char_to_idx)

    # Create dataset and DataLoader
    dataset = ShakespeareDataset(input_seqs, output_seqs)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model, loss, and optimizer
    model = FastWeightsModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, SEQ_LENGTH)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop with loss tracking
    loss_history = []
    
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)  # Save loss for visualization
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")

    # Save trained model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Plot training loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), loss_history, marker='o', linestyle='-', color='b')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.grid()
    plt.savefig("loss_plot.png")  # Save plot as an image
    plt.show()
