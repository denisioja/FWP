import pickle
import torch
from train import FastWeightsModel
from config import *  # Import hyperparameters

def sample_with_temperature(logits, temperature=TEMPERATURE):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()

if __name__ == "__main__":
    # Load tokenizer
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    char_to_idx = tokenizer["char_to_idx"]
    idx_to_char = tokenizer["idx_to_char"]

    # Update dynamic hyperparameters
    VOCAB_SIZE = len(char_to_idx)

    # Load model
    model = FastWeightsModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, SEQ_LENGTH)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Text generation parameters
    seed_text = "To be, or not to be, that is the question: "
    # More seed texts to try
    # seed_text = "Now is the winter of our discontent"
    # seed_text ="All the world's a stage"
    # seed_text ="This above all: to thine own self be true"
    # seed_text ="The fool doth think he is wise, but the wise man knows himself to be a fool"
    # seed_text ="Love all, trust a few, do wrong to none"

    # Generate text
    input_sequence = [char_to_idx[char] for char in seed_text if char in char_to_idx]
    if len(input_sequence) < SEQ_LENGTH:
        input_sequence = [char_to_idx[" "]] * (SEQ_LENGTH - len(input_sequence)) + input_sequence
    input_tensor = torch.tensor([input_sequence[-SEQ_LENGTH:]], dtype=torch.long)

    generated_text = seed_text

    with torch.no_grad():
        for _ in range(GENERATE_LENGTH):
            output = model(input_tensor)
            predicted_idx = sample_with_temperature(output, TEMPERATURE)

            generated_text += idx_to_char[predicted_idx]

            input_sequence.append(predicted_idx)
            input_tensor = torch.tensor([input_sequence[-SEQ_LENGTH:]], dtype=torch.long)

    print("Generated Text:")
    print(generated_text)

    # Save to file
    with open(OUTPUT_TEXT_PATH, "w") as f:
        f.write(generated_text)
