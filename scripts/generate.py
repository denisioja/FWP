"""
This script generates text using the trained Fast Weights model. It takes
a seed string, uses the model to predict subsequent characters, and 
generates a sequence of specified length.

Key functionalities:
- Load the trained model and tokenizer.
- Generate text by iteratively predicting the next character.
- Save generated text to a file for analysis.
"""

import pickle
import torch
from train import FastWeightsModel

if __name__ == "__main__":
    # Load tokenizer
    with open("C:/Users/Denis/Desktop/FWP/data/processed/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    char_to_idx = tokenizer["char_to_idx"]
    idx_to_char = tokenizer["idx_to_char"]

    # Hyperparameters
    vocab_size = len(char_to_idx)
    embed_dim = 64
    hidden_dim = 128
    seq_length = 100

    # Load model
    model = FastWeightsModel(vocab_size, embed_dim, hidden_dim, seq_length)
    model.load_state_dict(torch.load("C:/Users/Denis/Desktop/FWP/models/fast_weights_model.pth", weights_only=True))
    model.eval()

    # Text generation parameters
    seed_text = "To be, or not to be, that is the question: "
    generate_length = 100

    # Generate text
    input_sequence = [char_to_idx[char] for char in seed_text if char in char_to_idx]
    input_tensor = torch.tensor([input_sequence[-seq_length:]], dtype=torch.long)

    generated_text = seed_text

    with torch.no_grad():
        for _ in range(generate_length):
            output = model(input_tensor)
            predicted_idx = torch.argmax(output, dim=1).item()

            # Append predicted character
            generated_text += idx_to_char[predicted_idx]

            # Update input sequence
            input_sequence.append(predicted_idx)
            input_tensor = torch.tensor([input_sequence[-seq_length:]], dtype=torch.long)

    print("Generated Text:")
    print(generated_text)

    # Save to file
    with open("C:/Users/Denis/Desktop/FWP/output/generated_text.txt", "w") as f:
        f.write(generated_text)






















