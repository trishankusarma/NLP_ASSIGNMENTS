#!/usr/bin/env python3
"""
Task 1: Author Verification and Ranking
Usage: python task1_inference.py <test_file> <output_dir>
"""
import sys
import json
import os
from src.word2vecModel.trainer import Word2VecTrainer
from src.AuthorAttribution import AuthorAttributor
from src.utils import task1_inference

MODEL_DIR = '../model/word2vec_model.pkl'
OUTPUT_FILE_NAME = 'task1_predictions.jsonl'

def main():
    if len(sys.argv) != 3:
        print("Usage: python task1_inference.py <test_file> <output_dir>")
        sys.exit(1)
    
    test_file = sys.argv[1]
    output_dir = sys.argv[2]

    # Load model
    print("Loading model...")
    model, vocab, save_dict = Word2VecTrainer.load_model(MODEL_DIR)

    # Create attributor
    attributor = AuthorAttributor(model, vocab)

    # Load test data
    print(f"Loading test data from {test_file}...")
    with open(test_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    
    # Prepare output
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, OUTPUT_FILE_NAME)

    with open(output_file, 'w', encoding='utf-8') as f_out:
        output = task1_inference(attributor, queries, output_needed = True)
        f_out.write(json.dumps(output) + '\n')
    
    print(f"Predictions saved to {output_file}")

if __name__ == '__main__':
    main()