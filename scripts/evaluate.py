"""
This script evaluates the trained Fast Weights model on validation data to
measure its performance. It calculates the loss and accuracy of the model
and visualizes its predictions.

Key functionalities:
- Load the trained model and tokenizer.
- Evaluate the model on validation sequences.
- Display sample predictions to assess quality
"""

import pickle
import torch
from torch.utils.data import DataLoader
from train import ShakespeareDataset, FastWeightsModel

if __name__ == "__main__":
    # Load tokenizer and validation data
    with open("../data/processed/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("../data/processed/sequences.pkl", "rb") as f:
        data = pickle.load(f)

    char_to_idx = tokenizer["char_to_idx"]
    idx_to_char = tokenizer["idx_to_char"]
    input_seqs, output_seqs = data["inputs"], data["outputs"]

    # Hyperparameters
    vocab_size = len(char_to_idx)
    embed_dim = 64
    hidden_dim = 128
    seq_length = 100

    # Create dataset and DataLoader for evaluation
    dataset = ShakespeareDataset(input_seqs, output_seqs)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Load model
    model = FastWeightsModel(vocab_size, embed_dim, hidden_dim, seq_length)
    model.load_state_dict(torch.load("../models/fast_weights_model.pth"))
    model.eval()

    # Evaluation
    total_loss = 0
    total_correct = 0
    total_samples = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            # Calculate accuracy
            predictions = torch.argmax(outputs, dim=1)
            total_correct += (predictions == targets).sum().item()
            total_samples += targets.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    # Example Predictions
    print("\nSample Predictions:")
    sample_input = input_seqs[0]
    sample_target = output_seqs[0]
    sample_input_tensor = torch.tensor(sample_input, dtype=torch.long).unsqueeze(0)

    output = model(sample_input_tensor)
    predicted_char = idx_to_char[torch.argmax(output, dim=1).item()]
    input_text = ''.join([idx_to_char[idx] for idx in sample_input])
    target_char = idx_to_char[sample_target]

    print(f"Input: {input_text}")
    print(f"Target: {target_char}")
    print(f"Predicted: {predicted_char}")






















    