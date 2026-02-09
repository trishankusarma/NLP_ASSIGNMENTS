import numpy as np
from k_means_constrained import KMeansConstrained
from collections import Counter
import torch
import torch.optim as optim
import re
from src.utils import (
    normalize, cosine_similarity
)

class AuthorAttributor:
    """
    Author attribution using Word2Vec embeddings + stylometric features
    """

    def __init__(self, model, vocab, training_texts=None):
        self.model = model
        self.vocab = vocab
        self.model.eval()

    # WORD2VEC AGGREGATION
    def text_to_embedding(self, text):
        indices = self.vocab.encode(text)

        embeddings = []
        for idx in indices:
            word = self.vocab.idx2word[idx]
            if word.isalnum():   # keep only real words
                embeddings.append(self.model.get_embedding(idx, self.vocab))

        if not embeddings:
            return np.zeros(self.model.embedding_dim)

        return np.mean(embeddings, axis=0)

    # COMBINED REPRESENTATION - moved only to document embeddings
    def get_combined_representation(self, text):
        w2v = self.text_to_embedding(text)
        return w2v

    # TASK 1 — RANKING
    def task1_rank_candidates(self, query_text, candidate_texts, candidate_ids):
        query_emb = self.get_combined_representation(query_text).reshape(1, -1)

        candidate_embs = np.array([
            self.get_combined_representation(t) for t in candidate_texts
        ])

        sims = cosine_similarity(query_emb, candidate_embs)[0]
        ranked = np.argsort(sims)[::-1]

        return [candidate_ids[i] for i in ranked]
    
    def task2_cluster_authors(self, texts, num_authors, min_chunks_per_author, fine_tune=False):
        """
        Cluster using KMeansConstrained for balanced clusters
        
        Args:
            texts: Text chunks
            num_authors: Number of authors
            fine_tune: Whether to fine-tune first
        """
        print(f"\nClustering {len(texts)} chunks into {num_authors} authors...")
        
        if fine_tune:
            self.fine_tune_on_test_chunks(texts, num_epochs=3, lr=0.005)
        
        # Get embeddings
        embeddings = np.array([
            self.get_combined_representation(t) for t in texts
        ])
        
        print(f"Embedding shape: {embeddings.shape}")
        
        # Diagnostics
        sim_matrix = cosine_similarity(embeddings)
        avg_sim = (sim_matrix.sum() - len(embeddings)) / (len(embeddings) * (len(embeddings) - 1))
        print(f"Avg pairwise similarity: {avg_sim:.4f}")
        
        # L2 normalize
        embeddings = normalize(embeddings, norm='l2')
        
        # Calculate size constraints
        n_chunks = len(texts)
        size_min = min_chunks_per_author  # Minimum per cluster
        size_max = size_min + 1              # Maximum per cluster
        
        print(f"\nSize constraints: min={size_min}, max={size_max} per cluster")
        
        # KMeansConstrained clustering
        clf = KMeansConstrained(
            n_clusters=num_authors,
            size_min=size_min,
            size_max=size_max,
            n_init=100,           # Try 100 initializations
            max_iter=500,
            random_state=42
        )
        
        labels = clf.fit_predict(embeddings)
        
        # Show results
        counts = Counter(labels)
        sizes = sorted(counts.values(), reverse=True)
        
        print(f"\nCluster distribution: {sizes}")
        print(f"Min: {min(sizes)}, Max: {max(sizes)}")
        print(f"Inertia: {clf.inertia_:.4f}")
        
        return labels.tolist()

    def fine_tune_on_test_chunks(self, texts, num_epochs=2, lr=0.0005):
        """Fast fine-tuning with batched updates"""
        print(f"Fine-tuning on {len(texts)} test chunks...")
        
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Collect skip-gram pairs
        pairs = []
        for text in texts:
            indices = self.vocab.encode(text)
            if len(indices) < 5:
                continue
            
            for i, center_idx in enumerate(indices):
                if center_idx == self.vocab.unk_idx:
                    continue
                
                word = self.vocab.idx2word[center_idx]
                if not word.isalnum():
                    continue
                
                window = 2
                start = max(0, i - window)
                end = min(len(indices), i + window + 1)
                
                for j in range(start, end):
                    if i != j:
                        pairs.append((center_idx, indices[j]))
        
        if not pairs:
            print("  No valid pairs for fine-tuning")
            self.model.eval()
            return
        
        print(f"Generated {len(pairs)} pairs")
        
        # Batch training
        batch_size = 512
        for epoch in range(num_epochs):
            np.random.shuffle(pairs)
            epoch_loss = 0
            n_batches = 0
            
            for batch_start in range(0, len(pairs), batch_size):
                batch = pairs[batch_start:batch_start + batch_size]
                if len(batch) < 10:
                    continue
                
                centers = torch.tensor([p[0] for p in batch], dtype=torch.long)
                contexts = torch.tensor([p[1] for p in batch], dtype=torch.long)
                
                center_emb = self.model.in_embeddings(centers)
                context_emb = self.model.out_embeddings(contexts)
                
                pos_scores = (center_emb * context_emb).sum(dim=1)
                loss = -torch.log(torch.sigmoid(pos_scores) + 1e-10).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            if n_batches > 0:
                print(f"  Epoch {epoch+1}/{num_epochs}: Loss = {epoch_loss/n_batches:.4f}")
        
        self.model.eval()
        print("Fine-tuning complete\n")