# config.py - Centralized Hyperparameters

# Model Hyperparameters
VOCAB_SIZE = None  # Will be set dynamically after loading tokenizer
EMBED_DIM = 128
HIDDEN_DIM = 256
SEQ_LENGTH = 200

# Training Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

# Generation Parameters
TEMPERATURE = 0.6  # Adjust for diversity 0.8
GENERATE_LENGTH = 500

# File Paths
TOKENIZER_PATH = "C:/Users/Denis/Desktop/FWP/data/processed/tokenizer.pkl"
SEQUENCES_PATH = "C:/Users/Denis/Desktop/FWP/data/processed/sequences.pkl"
MODEL_PATH = "C:/Users/Denis/Desktop/FWP/models/fast_weights_model.pth"
OUTPUT_TEXT_PATH = "C:/Users/Denis/Desktop/FWP/output/generated_text.txt"
