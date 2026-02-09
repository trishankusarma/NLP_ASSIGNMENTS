import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import torch
import re


class AuthorAttributor:
    """
    Author attribution using Word2Vec embeddings + stylometric features
    """

    def __init__(self, model, vocab, training_texts=None):
        self.model = model
        self.vocab = vocab
        self.model.eval()

    # IDF (DOCUMENT FREQUENCY)
    def _compute_idf_values(self, texts):
        """
        IDF(token) = log((N + 1) / (df + 1))
        """
        print("Computing document-frequency IDF values...")

        doc_freq = Counter()
        N = len(texts)

        for text in texts:
            tokens = set(self.vocab.tokenize(text))
            for tok in tokens:
                if tok.isalnum():  # IDF only for real words
                    doc_freq[tok] += 1

        self.idf = {
            tok: np.log((N + 1) / (df + 1))
            for tok, df in doc_freq.items()
        }

        print(f"Computed IDF for {len(self.idf)} tokens")

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

    # COMBINED REPRESENTATION
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
        embeddings = np.array([
            self.get_combined_representation(t) for t in texts
        ])

        kmeans = KMeans(
            n_clusters=num_authors,
            n_init=20,
            max_iter=500,
            random_state=42
        )

        return kmeans.fit_predict(embeddings).tolist()
