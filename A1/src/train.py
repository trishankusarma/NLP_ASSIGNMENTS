"""
Training model for Word2Vec using skipGram model
./src/train.py ./data/train_data
"""
import sys
import glob # using glob to grab all the input data with .txt extention
import os
from src.word2vec import Vocabulary, Word2VecTrainer

def load_training_data(input_dir):
    # this function will help u load all the training data from the input directory
    print(f"Loading data from ${input_dir}")
    input_file_paths = glob.glob(os.path.join(input_dir, "*.txt"))

    given_text_data = []

    for input_file_path in input_file_paths:
        print(f"Loading data from ${input_file_path}")
        with open(input_file_path, 'r', encoding = 'utf-8') as f:
            text = f.read()
            given_text_data.append(text)
    
    print("Data loaded")
    
    return given_text_data
 
def main():
    print("Training Word2Vec model with skip_grams")

    input_data = sys.argv[1]

    # Step 1 : Loading training data
    given_text_data = load_training_data(input_data)

    # Step 2: Building vocabulary
    vocab = Vocabulary()  # Adjust min_freq as needed
    vocab.build_vocab(given_text_data)

    # Step 3: Train the word2Vec model 
    trainer = Word2VecTrainer(vocab)
    trainer.train(given_text_data)

    # Step 4: Save model
    trainer.save_model('../model/word2vec_model.pkl')
    
    print("Training completed successfully!")

if __name__ == '__main__':
    main()