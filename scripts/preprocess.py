# This script takes raw Shakespeare text, tokenizes it into characters,
# maps each character to a numerical index, and cretes fiexed-length input
# sequences for training and validation.

"""
Key functionalities:
- Read raw text data.
- Tokenize into unique characters.
- Create input-output pairs for character prediction.
- Save prepeocessed data to disk.
"""

import os
import pickle
from typing import List, Dict

def load_text(file_path: str) -> str:
    with open(file_path, 'r') as f:
        return f.read()

def tokenize_text(text: str) -> Dict[str, int]:
    unique_chars = sorted(set(text))
    return {char: idx for idx, char in enumerate(unique_chars)}

def create_sequences(text: str, seq_length: int, char_to_idx: Dict[str, int]):
    input_seqs = []
    output_seqs = []
    for i in range(len(text) - seq_length):
        input_seq = text[i:i + seq_length]
        output_seq = text[i + seq_length]
        input_seqs.append([char_to_idx[char] for char in input_seq])
        output_seqs.append(char_to_idx[output_seq])
    return input_seqs, output_seqs

if __name__ == "__main__":
    # Load data
    raw_text = load_text("C:/Users/Denis/Desktop/FWP/data/shakespeare.txt")

    # Tokenize
    char_to_idx = tokenize_text(raw_text)
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    # Create sequences
    seq_length = 200 # Modified based on the parameter in config.py
    input_seqs, output_seqs = create_sequences(raw_text, seq_length, char_to_idx)

    # Save processed data
    os.makedirs("C:/Users/Denis/Desktop/FWP/data/processed", exist_ok=True)
    with open("C:/Users/Denis/Desktop/FWP/data/processed/tokenizer.pkl", "wb") as f:
        pickle.dump({"char_to_idx": char_to_idx, "idx_to_char": idx_to_char}, f)
    with open("C:/Users/Denis/Desktop/FWP/data/processed/sequences.pkl", "wb") as f:
        pickle.dump({"inputs": input_seqs, "outputs": output_seqs}, f)
