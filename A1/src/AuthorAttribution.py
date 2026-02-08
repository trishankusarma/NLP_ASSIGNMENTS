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

        # Compute document-frequency based IDF (CORRECT definition)
        if training_texts is None:
            raise ValueError("training_texts required to compute IDF correctly")

        # self._compute_idf_values(training_texts)

        # Precompute style feature normalization stats
        # self._fit_style_scaler(training_texts)

    # ------------------------------------------------------------------
    # IDF (DOCUMENT FREQUENCY — NOT TOKEN FREQUENCY)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # WORD2VEC AGGREGATION
    # ------------------------------------------------------------------
    def text_to_embedding(self, text):
        # """
        # TF-IDF weighted Word2Vec embedding
        # TF-IDF applied ONLY to real words
        # """
        # indices = self.vocab.encode(text)
        # tokens = [self.vocab.idx2word[i] for i in indices]

        # embeddings = []
        # weights = []

        # token_counts = Counter(tokens)
        # total_tokens = len(tokens)

        # for tok, idx in zip(tokens, indices):
        #     emb = self.model.get_embedding(idx)

        #     # TF-IDF ONLY for real words
        #     if tok.isalnum():
        #         tf = token_counts[tok] / total_tokens
        #         idf = self.idf.get(tok, 1.0)
        #         weight = tf * idf
        #     else:
        #         weight = 0.1  # small constant for punctuation / char ngrams

        #     embeddings.append(emb)
        #     weights.append(weight)

        # embeddings = np.array(embeddings)
        # weights = np.array(weights)

        # if weights.sum() == 0:
        #     weights = np.ones(len(weights)) / len(weights)
        # else:
        #     weights /= weights.sum()

        # return np.average(embeddings, axis=0, weights=weights)

        indices = self.vocab.encode(text)

        embeddings = []
        for idx in indices:
            word = self.vocab.idx2word[idx]
            if word.isalnum():   # keep only real words
                embeddings.append(self.model.get_embedding(idx, self.vocab))

        if not embeddings:
            return np.zeros(self.model.embedding_dim)

        return np.mean(embeddings, axis=0)

    # ------------------------------------------------------------------
    # STYLOMETRIC FEATURES
    # ------------------------------------------------------------------
    def compute_style_features(self, text):
        tokens = self.vocab.tokenize(text)
        words = [t for t in tokens if t.isalnum()]

        features = {}

        features["avg_word_length"] = np.mean([len(w) for w in words]) if words else 0
        features["ttr"] = len(set(words)) / len(words) if words else 0

        punct = re.findall(r"[.,;:!?—]", text)
        features["punct_ratio"] = len(punct) / len(text) if text else 0
        features["comma_ratio"] = text.count(",") / len(text) if text else 0
        features["semicolon_ratio"] = text.count(";") / len(text) if text else 0

        sentences = re.split(r"[.!?]", text)
        sent_lengths = [len(s.split()) for s in sentences if s.strip()]
        features["avg_sentence_length"] = np.mean(sent_lengths) if sent_lengths else 0
        features["sentence_length_std"] = np.std(sent_lengths) if sent_lengths else 0

        function_words = {
            "the","a","an","and","or","but","in","on","at","to","for","of","with",
            "by","from","as","is","was","are","be","been","being","have","has","had",
            "do","does","did","will","would","should","could","may","might","must",
            "i","you","he","she","it","we","they","this","that"
        }

        fw_count = sum(1 for w in words if w in function_words)
        features["function_word_ratio"] = fw_count / len(words) if words else 0

        return features

    # ------------------------------------------------------------------
    # STYLE FEATURE NORMALIZATION
    # ------------------------------------------------------------------
    def _fit_style_scaler(self, texts):
        print("Fitting stylometric feature scaler...")

        feats = []
        for text in texts:
            f = self.compute_style_features(text)
            feats.append(list(f.values()))

        feats = np.array(feats)
        self.style_mean = feats.mean(axis=0)
        self.style_std = feats.std(axis=0) + 1e-8

        self.style_keys = list(self.compute_style_features(texts[0]).keys())

    def _normalize_style(self, style_dict):
        vec = np.array([style_dict[k] for k in self.style_keys])
        return (vec - self.style_mean) / self.style_std

    # ------------------------------------------------------------------
    # COMBINED REPRESENTATION
    # ------------------------------------------------------------------
    def get_combined_representation(self, text):
        w2v = self.text_to_embedding(text)
        # style = self.compute_style_features(text)
        # style_vec = self._normalize_style(style)

        # return np.concatenate([w2v, style_vec])
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
