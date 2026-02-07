import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import torch
from collections import Counter

# This file deals with all sort of stuff necessary for inference of the 2 tasks
class AuthorAttributor:
    """Author attribution using Word2Vec embeddings"""
    
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab
        self.model.eval()
        
        # Precompute IDF values for efficiency
        self._compute_idf_values()
    
    def _compute_idf_values(self):
        """
        Precompute IDF (Inverse Document Frequency) values for all tokens
        IDF(token) = log(total_documents / documents_containing_token)
        """
        # For Word2Vec, we use corpus-level statistics
        total_token_count = sum(self.vocab.word_counts.values())
        
        self.idf = {}
        for word, count in self.vocab.word_counts.items():
            # IDF = log(total_tokens / token_frequency)
            # Adding smoothing to avoid division by zero
            self.idf[word] = np.log((total_token_count + 1) / (count + 1))
        
        print(f"Computed IDF values for {len(self.idf)} tokens")
    
    def text_to_embedding(self, text, method='tfidf'):
        """
        Convert text to a single embedding vector
        
        Args:
            text: input text string
            method: aggregation method 
                - 'average': simple mean of word embeddings
                - 'weighted_average': inverse frequency weighting
                - 'tfidf': proper TF-IDF weighting (RECOMMENDED)
        
        Returns:
            embedding vector (numpy array)
        """
        indices = self.vocab.encode(text)
        
        if len(indices) == 0:
            return np.zeros(self.model.embedding_dim)
        
        # Get embeddings for all words
        embeddings = []
        words = []
        for idx in indices:
            emb = self.model.get_embedding(idx)
            embeddings.append(emb)
            words.append(self.vocab.idx2word[idx])
        
        embeddings = np.array(embeddings)
        
        # Choose aggregation method
        if method == 'tfidf':
            # TF-IDF weighted average
            weights = self._compute_tfidf_weights(words, indices)
            text_embedding = np.average(embeddings, axis=0, weights=weights)
            
        elif method == 'weighted_average':
            # Simple inverse frequency weighting
            weights = []
            for idx in indices:
                word = self.vocab.idx2word[idx]
                freq = self.vocab.word_counts[word]
                weight = 1.0 / (1.0 + np.log(freq))
                weights.append(weight)
            
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize
            text_embedding = np.average(embeddings, axis=0, weights=weights)
            
        else:  # 'average'
            # Simple average
            text_embedding = np.mean(embeddings, axis=0)
        
        return text_embedding
    
    def _compute_tfidf_weights(self, words, indices):
        """
        Compute TF-IDF weights for words in a document
        
        TF (Term Frequency) = count of word in document / total words in document
        IDF (Inverse Document Frequency) = precomputed in _compute_idf_values()
        TF-IDF = TF * IDF
        """
        # Compute term frequencies in this document
        word_counts = Counter(words)
        total_words = len(words)
        
        # Compute TF-IDF for each word
        tfidf_scores = []
        for word in words:
            # TF: frequency in this document
            tf = word_counts[word] / total_words
            
            # IDF: precomputed inverse document frequency
            idf = self.idf.get(word, 1.0)  # Default to 1.0 if not found
            
            # TF-IDF score
            tfidf = tf * idf
            tfidf_scores.append(tfidf)
        
        # Normalize weights to sum to 1
        tfidf_scores = np.array(tfidf_scores)
        if np.sum(tfidf_scores) > 0:
            tfidf_scores = tfidf_scores / np.sum(tfidf_scores)
        else:
            # Fallback to uniform weights
            tfidf_scores = np.ones(len(tfidf_scores)) / len(tfidf_scores)
        
        return tfidf_scores
    
    def compute_style_features(self, text):
        """
        Compute enhanced stylometric features for a text
        These supplement the Word2Vec embeddings with author-specific patterns
        """
        tokens = self.vocab.tokenize(text)
        words_only = [t for t in tokens if '_' not in t and '<' not in t and '>' not in t]
        
        features = {}
        
        # Basic statistics
        features['avg_word_length'] = np.mean([len(word) for word in words_only]) if words_only else 0
        features['text_length'] = len(tokens)
        
        # Punctuation patterns (strong author signal!)
        punct_count = sum(1 for char in text if char in '.,;:!?')
        features['punct_ratio'] = punct_count / len(text) if len(text) > 0 else 0
        
        # Specific punctuation preferences
        features['comma_ratio'] = text.count(',') / len(text) if len(text) > 0 else 0
        features['semicolon_ratio'] = text.count(';') / len(text) if len(text) > 0 else 0
        features['exclaim_ratio'] = text.count('!') / len(text) if len(text) > 0 else 0
        
        # Vocabulary richness (type-token ratio)
        unique_tokens = set(tokens)
        features['ttr'] = len(unique_tokens) / len(tokens) if tokens else 0
        
        # Function words ratio (VERY strong author signal!)
        function_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'should', 'could', 'may', 'might', 'must',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that'
        }
        func_count = sum(1 for w in words_only if w.lower() in function_words)
        features['function_word_ratio'] = func_count / len(words_only) if words_only else 0
        
        # Sentence-level features
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if sentences:
            sent_lengths = [len(s.split()) for s in sentences]
            features['avg_sentence_length'] = np.mean(sent_lengths)
            features['sentence_length_std'] = np.std(sent_lengths)
        else:
            features['avg_sentence_length'] = 0
            features['sentence_length_std'] = 0
        
        # Character-level features
        features['avg_chars_per_word'] = len(text) / len(words_only) if words_only else 0
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        return features
    
    def get_combined_representation(self, text, include_style=True):
        """
        Get combined representation: Word2Vec + stylometric features
        
        Args:
            text: input text
            include_style: whether to include stylometric features (recommended True)
        
        Returns:
            Combined feature vector
        """
        # Word2Vec embedding with TF-IDF weighting
        w2v_embedding = self.text_to_embedding(text, method='tfidf')
        
        if not include_style:
            return w2v_embedding
        
        # Enhanced stylometric features
        style_features = self.compute_style_features(text)
        
        # Create feature vector (scaled appropriately)
        style_vector = np.array([
            style_features['avg_word_length'],           # ~5-8
            style_features['punct_ratio'] * 100,         # ~0-5
            style_features['ttr'] * 10,                  # ~0-10
            style_features['function_word_ratio'] * 100, # ~20-40 (IMPORTANT!)
            style_features['avg_sentence_length'],       # ~10-30
            style_features['sentence_length_std'],       # ~5-15
            style_features['comma_ratio'] * 1000,        # ~0-50
            style_features['semicolon_ratio'] * 1000,    # ~0-10
            style_features['exclaim_ratio'] * 1000,      # ~0-20
            style_features['avg_chars_per_word'],        # ~4-6
        ])
        
        # Concatenate embeddings and style features
        combined = np.concatenate([w2v_embedding, style_vector])
        return combined

    def task1_rank_candidates(self, query_text, candidate_texts, candidate_ids):
        """
        Task 1: Rank candidates by similarity to query
        
        Args:
            query_text: query text string
            candidate_texts: list of candidate text strings
            candidate_ids: list of candidate IDs
        
        Returns:
            ranked list of candidate IDs (most similar first)
        """
        # Get query embedding
        query_emb = self.get_combined_representation(query_text)
        
        # Get candidate embeddings
        candidate_embs = []
        for text in candidate_texts:
            emb = self.get_combined_representation(text)
            candidate_embs.append(emb)
        
        candidate_embs = np.array(candidate_embs)
        query_emb = query_emb.reshape(1, -1)
        
        # Compute cosine similarities
        similarities = cosine_similarity(query_emb, candidate_embs)[0]
        
        # Rank by similarity (descending)
        ranked_indices = np.argsort(similarities)[::-1]
        ranked_ids = [candidate_ids[idx] for idx in ranked_indices]
        
        return ranked_ids
    
    def task2_cluster_authors(self, texts, num_authors, min_chunks_per_author=None):
        """
        Task 2: Cluster texts by author
        
        Args:
            texts: list of text strings
            num_authors: number of clusters (k)
            min_chunks_per_author: minimum chunks per author (informational)
        
        Returns:
            list of cluster assignments (one per text)
        """
        # Get embeddings for all texts
        embeddings = []
        for text in texts:
            emb = self.get_combined_representation(text)
            embeddings.append(emb)
        
        embeddings = np.array(embeddings)
        
        # K-means clustering with multiple initializations
        kmeans = KMeans(
            n_clusters=num_authors, 
            random_state=42, 
            n_init=20,        # Try more initializations for better clustering
            max_iter=500      # Allow more iterations to converge
        )
        cluster_assignments = kmeans.fit_predict(embeddings)
        
        return cluster_assignments.tolist()
    
    def fine_tune_on_texts(self, texts, epochs=2):
        """
        Fine-tune embeddings on new texts (for test-time adaptation in Task 2)
        This is allowed per the assignment specifications
        """
        print("Fine-tuning on test texts...")
        from src.word2vecModel.trainer import Word2VecTrainer
        
        # Create a temporary trainer with lower learning rate
        trainer = Word2VecTrainer(self.vocab)
        trainer.model = self.model
        trainer.epochs = epochs
        trainer.learning_rate = 0.001  # Lower LR for fine-tuning
        
        # Train on new texts
        trainer.train(texts)
        
        print("Fine-tuning completed")