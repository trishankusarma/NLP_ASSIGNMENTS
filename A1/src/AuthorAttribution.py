import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from collections import Counter
import torch
import re
import random

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

    # TASK 2 — CLUSTERING
    def task2_cluster_authors(self, texts, num_authors):
        """Cluster texts by author using K-Means"""
        print(f"Clustering {len(texts)} chunks into {num_authors} authors...")
        
        embeddings = np.array([
            self.get_combined_representation(t) for t in texts
        ])
        
        # L2 normalize
        embeddings = normalize(embeddings, norm='l2')
        
        # K-Means with strong initialization
        kmeans = KMeans(
            n_clusters=num_authors,
            n_init=50,           # Try 50 initializations
            max_iter=500,
            random_state=42,
            algorithm='lloyd'
        )
        
        labels = kmeans.fit_predict(embeddings)
        
        print(f"✓ Inertia: {kmeans.inertia_:.4f}")
        
        return labels.tolist()