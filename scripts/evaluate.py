import pickle
import torch
from torch.utils.data import DataLoader
from train import ShakespeareDataset, FastWeightsModel
from config import *  # Import hyperparameters

if __name__ == "__main__":
    # Load tokenizer and validation data
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    with open(SEQUENCES_PATH, "rb") as f:
        data = pickle.load(f)

    char_to_idx = tokenizer["char_to_idx"]
    idx_to_char = tokenizer["idx_to_char"]
    input_seqs, output_seqs = data["inputs"], data["outputs"]

    # Update dynamic hyperparameters
    VOCAB_SIZE = len(char_to_idx)

    # Create dataset and DataLoader for evaluation
    dataset = ShakespeareDataset(input_seqs, output_seqs)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    model = FastWeightsModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, SEQ_LENGTH)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Evaluation
    total_loss = 0
    total_correct = 0
    total_samples = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1))  

            total_loss += loss.item() * inputs.size(0)

            predictions = torch.argmax(outputs, dim=1)
            total_correct += (predictions == targets.view(-1)).sum().item()
            total_samples += targets.numel()

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
