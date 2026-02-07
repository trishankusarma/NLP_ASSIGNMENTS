#!/usr/bin/env python3
"""
Task 2: Author Clustering
Usage: python task2_inference.py <test_file> <output_dir>
"""

import sys
import json
import os
from word2vec import Word2VecTrainer
from author_attribution import AuthorAttributor

MODEL_DIR = '../model/word2vec_model.pkl'
OUTPUT_FILE_NAME = 'task2_predictions.jsonl'

def main():
    if len(sys.argv) != 3:
        print("Usage: python task2_inference.py <test_file> <output_dir>")
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
        test_data = json.load(f)
    
    num_authors = test_data['num_authors']
    min_chunks_per_author = test_data['min_chunks_per_author']
    chunks = test_data['chunks']

    print(f"Clustering {len(chunks)} chunks into {num_authors} authors...")
    print(f"Minimum chunks per author: {min_chunks_per_author}")

    # Cluster texts
    cluster_assignments = attributor.task2_cluster_authors(chunks, num_authors, min_chunks_per_author)

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, OUTPUT_FILE_NAME)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cluster_assignments, f)
    
    print(f"Predictions saved to {output_file}")

if __name__ == '__main__':
    main()