#!/usr/bin/env python3
"""
Task 2: Author Clustering with Test-Time Adaptation
"""
import sys
import json
import os
from src.word2vecModel.trainer import Word2VecTrainer
from src.AuthorAttribution import AuthorAttributor
from src.utils import task2_inference

MODEL_DIR = './model/word2vec_model.pkl'
OUTPUT_FILE_NAME = 'task2_predictions.json'


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
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    cluster_assignments = task2_inference(attributor, test_data, fine_tune = True)

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, OUTPUT_FILE_NAME)
    
    with open(output_file, 'w') as f:
        json.dump(cluster_assignments, f)
    
    print(f"\nSaved to {output_file}")
    
    # Show distribution
    from collections import Counter
    counts = Counter(cluster_assignments)
    print("\nCluster Distribution:")
    for cid in sorted(counts.keys()):
        print(f"  Cluster {cid}: {counts[cid]} chunks")


if __name__ == '__main__':
    main()