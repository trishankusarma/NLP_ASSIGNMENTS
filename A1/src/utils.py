import matplotlib.pyplot as plt
import numpy as np
import os
from src.hungarian_eval import evaluate_clustering

def moving_average(x, window=100):
    return np.convolve(x, np.ones(window)/window, mode='valid')

def plot_loss_curve(flat_losses, save_dir):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    smoothed = moving_average(flat_losses, window=1000)

    plt.figure()
    plt.plot(smoothed)
    plt.xlabel("Training Step")
    plt.ylabel("Smoothed Loss")
    plt.title("Smoothed Training Loss")
    plt.grid(True)

    # Full save path
    save_path = os.path.join(save_dir, 'loss_plot.png')

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Loss curve saved to {save_path}")

def task1_inference(attributor, queries, output_needed = False):
    print(f"Processing {len(queries)} queries...")
    overall_score = 0
    output = []
    rank_list_of_queries = []

    for query_data in queries:
        query_id = query_data['query_id']
        query_text = query_data['query_text']
        candidates = query_data['candidates']

        #ground truth -- taken only to check evaluation score 
        ground_truth = query_data["_ground_truth"] 
        
        # Get candidate IDs and texts
        candidate_ids = list(candidates.keys())
        candidate_texts = [candidates[cid] for cid in candidate_ids]

        # Rank candidates
        ranked_ids = attributor.task1_rank_candidates(query_text, candidate_texts, candidate_ids)
        score = None

        # fetch rank of ground truth candidate
        for index, author_id in enumerate(ranked_ids):
            if author_id == ground_truth:
                score = 1/(index+1)
                rank_list_of_queries.append(index+1)
                break
            
        overall_score += score
                
        if output_needed:
            # Write output
            output.append({
                "query_id": query_id,
                "ranked_candidates": ranked_ids
            })
    
    MRR = overall_score/len(queries)
    
    print(f"Overall Score achieved is {overall_score} and the rank_list is : {rank_list_of_queries}")
    print(f"MRR : {MRR}")
    return output

def task2_inference(attributor, test_data, fine_tune = True):
    num_authors = test_data['num_authors']
    chunks = test_data['chunks']
    min_chunks_per_author = test_data['min_chunks_per_author']

    # Cluster with fine-tuning enabled
    cluster_assignments = attributor.task2_cluster_authors(
        chunks, 
        num_authors,
        min_chunks_per_author,
        fine_tune=fine_tune  # Enable test-time adaptation
    )

    accuracy, mapping, conf_matrix = evaluate_clustering(
        cluster_assignments, 
        test_data["_ground_truth"], 
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

    return cluster_assignments

def normalize(X, norm='l2', axis=1):
    """
    Normalize samples individually to unit norm (sklearn.preprocessing.normalize replacement)
    
    Args:
        X: numpy array of shape (n_samples, n_features)
        norm: 'l1', 'l2', or 'max'
        axis: axis along which to normalize (1 for row-wise)
    
    Returns:
        Normalized array of same shape as X
    """
    X = np.array(X, dtype=np.float64)
    
    if norm == 'l2':
        # L2 normalization (Euclidean norm)
        norms = np.linalg.norm(X, axis=axis, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        return X / norms
    
    elif norm == 'l1':
        # L1 normalization (Manhattan norm)
        norms = np.abs(X).sum(axis=axis, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return X / norms
    
    elif norm == 'max':
        # Max normalization
        norms = np.abs(X).max(axis=axis, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return X / norms
    
    else:
        raise ValueError(f"Unsupported norm: {norm}")


def cosine_similarity(X, Y=None):
    """
    Cosine similarity = dot(X, Y) / (||X|| * ||Y||)
    """
    X = np.array(X, dtype=np.float64)
    
    if Y is None:
        Y = X
    else:
        Y = np.array(Y, dtype=np.float64)
    
    # Normalize X and Y to unit vectors (L2 norm)
    X_normalized = normalize(X, norm='l2', axis=1)
    Y_normalized = normalize(Y, norm='l2', axis=1)
    
    # Cosine similarity is just dot product of normalized vectors
    similarity = np.dot(X_normalized, Y_normalized.T)
    
    # Clip to handle numerical errors (should be in [-1, 1])
    similarity = np.clip(similarity, -1.0, 1.0)
    
    return similarity
