import json
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

def evaluate_clustering(predicted_labels, true_labels, num_clusters):
    confusion_matrix = np.zeros((num_clusters, num_clusters), dtype=int)

    for p, t in zip(predicted_labels, true_labels):
        if 0 <= p < num_clusters and 0 <= t < num_clusters:
            confusion_matrix[p][t] += 1

    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)

    total_correct = confusion_matrix[row_ind, col_ind].sum()
    accuracy = total_correct / len(predicted_labels)

    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    return accuracy, mapping

TASK1_GOLD_PATH = "test_task1_gold.json"
TASK2_GOLD_PATH = "test_task2_gold.json"
TASK2_TEST_PATH = "test_task2.json"

def evaluate_task1(predictions_jsonl_path):
    """
    predictions_jsonl format:
    {"query_id": "...", "ranked_candidates": ["cand_3", "cand_1", ...]}
    """

    # Load gold
    with open(TASK1_GOLD_PATH, "r") as f:
        gold = json.load(f)

    # Load predictions
    predictions = []
    with open(predictions_jsonl_path, "r") as f:
        for line in f:
            predictions.append(json.loads(line))

    reciprocal_ranks = []
    top1 = 0
    top5 = 0
    ranks = []

    for pred in predictions:
        qid = pred["query_id"]
        ranked = pred["ranked_candidates"]
        true_cand = gold[qid]["true_candidate"]

        if true_cand in ranked:
            rank = ranked.index(true_cand) + 1
            reciprocal_ranks.append(1.0 / rank)
            ranks.append(rank)

            if rank == 1:
                top1 += 1
            if rank <= 5:
                top5 += 1
        else:
            reciprocal_ranks.append(0.0)
            ranks.append(len(ranked) + 1)

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    top1_acc = top1 / len(predictions)
    top5_acc = top5 / len(predictions)
    mean_rank = sum(ranks) / len(ranks)

    print("\n===== TASK 1 EVALUATION =====")
    print(f"Queries evaluated        : {len(predictions)}")
    print(f"MRR (required)           : {mrr:.4f}")
    print(f"Top-1 Accuracy           : {top1_acc:.4f}")
    print(f"Top-5 Accuracy           : {top5_acc:.4f}")
    print(f"Mean Rank                : {mean_rank:.2f}")

    return {
        "MRR": mrr,
        "Top1": top1_acc,
        "Top5": top5_acc,
        "MeanRank": mean_rank
    }


def evaluate_task2(predictions_json_path):
    """
    predictions_json format:
    [0, 2, 0, 1, 2, 1, ...]
    """

    # Load gold
    with open(TASK2_GOLD_PATH, "r") as f:
        true_labels = json.load(f)

    # Load test meta (for k)
    with open(TASK2_TEST_PATH, "r") as f:
        test_data = json.load(f)

    num_clusters = test_data["num_authors"]

    # Load predictions
    with open(predictions_json_path, "r") as f:
        predicted_labels = json.load(f)

    accuracy, mapping = evaluate_clustering(
        predicted_labels,
        true_labels,
        num_clusters
    )

    # Cluster size diagnostics
    cluster_sizes = defaultdict(int)
    for p in predicted_labels:
        cluster_sizes[p] += 1

    print("\n===== TASK 2 EVALUATION =====")
    print(f"Chunks evaluated         : {len(predicted_labels)}")
    print(f"Num authors (k)          : {num_clusters}")
    print(f"Hungarian Accuracy       : {accuracy:.4f}")
    print("\nCluster → Author mapping:")
    for c, a in mapping.items():
        print(f"  Cluster {c} → Author {a}")

    print("\nCluster sizes:")
    for c in sorted(cluster_sizes):
        print(f"  Cluster {c}: {cluster_sizes[c]} chunks")

    return {
        "HungarianAccuracy": accuracy,
        "Mapping": mapping,
        "ClusterSizes": dict(cluster_sizes)
    }
