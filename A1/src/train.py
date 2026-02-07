"""
Training model for Word2Vec using skipGram model
./src/train.py ./data/train_data
"""
import sys
import glob
import os

PLOT_DIR = '../output'
MODEL_DIR = '../model/word2vec_model.pkl'

def load_training_data(input_dir):
    print(f"Loading data from {input_dir}")
    input_file_paths = sorted(glob.glob(os.path.join(input_dir, "*.txt")))

    given_text_data = []

    for input_file_path in input_file_paths:
        print(f"Loading data from {input_file_path}")
        with open(input_file_path, 'r', encoding='utf-8') as f:
            given_text_data.append(f.read())

    print("Data loaded")
    return given_text_data

def main():
    if len(sys.argv) < 2:
        print("Usage: python train.py <training_data_dir>")
        sys.exit(1)
    
    from src.word2vecModel.vocabulary import Vocabulary
    from src.word2vecModel.trainer import Word2VecTrainer

    print("Training Word2Vec model with skip-grams")

    input_data = sys.argv[1]

    # Step 1: Load training data
    given_text_data = load_training_data(input_data)

    # Step 2: Build vocabulary
    vocab = Vocabulary()
    vocab.build_vocab(given_text_data)

    # Step 3: Train Word2Vec
    trainer = Word2VecTrainer(vocab)
    trainer.train(given_text_data, save_dir=PLOT_DIR)

    # Step 4: Save model
    trainer.save_model(MODEL_DIR)

    print("Training completed successfully!")

if __name__ == '__main__':
    main()