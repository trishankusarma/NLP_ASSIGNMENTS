#!/usr/bin/env python3
"""
Evaluate clustering results using Hungarian algorithm
Usage: python eval.py <predictions_file> <ground_truth_file>
"""

import json
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment


def evaluate_clustering(predicted_labels, true_labels, num_clusters):
    """
    Evaluate clustering accuracy using the Hungarian algorithm.
    
    Args:
        predicted_labels: List of predicted cluster IDs (0 to k-1)
        true_labels: List of true author IDs (0 to k-1)
        num_clusters: Number of clusters (k)
        
    Returns:
        accuracy: Fraction of correctly assigned chunks under optimal mapping
        mapping: Dictionary mapping predicted cluster ID to true author ID
        confusion_matrix: The confusion matrix used
    """
    # Create confusion matrix
    # C[i][j] = number of chunks in predicted cluster i that belong to true author j
    confusion_matrix = np.zeros((num_clusters, num_clusters), dtype=int)
    
    for pred, true in zip(predicted_labels, true_labels):
        if 0 <= pred < num_clusters and 0 <= true < num_clusters:
            confusion_matrix[pred][true] += 1
    
    print("\nConfusion Matrix (rows=predicted, cols=true):")
    print(confusion_matrix)
    
    # Hungarian algorithm: maximize sum of assignments
    # linear_sum_assignment minimizes, so we negate
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    
    # Calculate accuracy
    total_correct = confusion_matrix[row_ind, col_ind].sum()
    total_samples = len(predicted_labels)
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    # Create mapping from predicted cluster -> true author
    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    
    print("\nOptimal Mapping (predicted -> true):")
    for pred, true in sorted(mapping.items()):
        count = confusion_matrix[pred][true]
        print(f"  Cluster {pred} -> Author {true} ({count} chunks)")
    
    return accuracy, mapping, confusion_matrix


def main():
    if len(sys.argv) != 3:
        print("Usage: python eval.py <predictions_file> <test_file_with_ground_truth>")
        print("\nExample:")
        print("  python eval.py ./output/task2_predictions.json ./split_data/test/task2_test.json")
        sys.exit(1)
    
    predictions_file = sys.argv[1]
    ground_truth_file = sys.argv[2]
    
    # Load predictions
    print(f"Loading predictions from {predictions_file}...")
    with open(predictions_file, 'r') as f:
        predicted_labels = json.load(f)
    
    # Load ground truth
    print(f"Loading ground truth from {ground_truth_file}...")
    with open(ground_truth_file, 'r') as f:
        test_data = json.load(f)
    
    true_labels = test_data.get('_ground_truth', None)
    num_authors = test_data['num_authors']
    
    if true_labels is None:
        print("ERROR: No ground truth labels found in test file!")
        print("Make sure the test file has '_ground_truth' field.")
        sys.exit(1)
    
    # Validate
    if len(predicted_labels) != len(true_labels):
        print(f"ERROR: Mismatch in lengths!")
        print(f"  Predictions: {len(predicted_labels)}")
        print(f"  Ground truth: {len(true_labels)}")
        sys.exit(1)
    
    print(f"\nEvaluating clustering...")
    print(f"  Total chunks: {len(predicted_labels)}")
    print(f"  Number of authors: {num_authors}")
    
    # Evaluate
    accuracy, mapping, conf_matrix = evaluate_clustering(
        predicted_labels, 
        true_labels, 
        num_authors
    )
    
    # Print results
    print("\n" + "="*60)
    print(f"HUNGARIAN ALGORITHM ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("="*60)
    
    # Per-cluster accuracy
    print("\nPer-Cluster Accuracy:")
    for pred_cluster in range(num_authors):
        true_author = mapping[pred_cluster]
        correct = conf_matrix[pred_cluster][true_author]
        total = conf_matrix[pred_cluster].sum()
        if total > 0:
            cluster_acc = correct / total
            print(f"  Cluster {pred_cluster}: {correct}/{total} = {cluster_acc:.4f}")
    
    return accuracy


if __name__ == '__main__':
    main()