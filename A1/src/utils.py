import matplotlib.pyplot as plt
import numpy as np
import os

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