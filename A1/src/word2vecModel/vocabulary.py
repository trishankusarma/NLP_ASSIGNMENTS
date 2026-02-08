import numpy as np
from collections import Counter
import re

from config.hyper_parameters import (
    VOCABULARY_MIN_FREQ,
    NEG_SAMPLING_EXP_POWER,
    USE_CHAR_NGRAMS,
    CHAR_NGRAM_RANGE
)

"""
Vocabulary manager for Word2Vec (Skip-gram)
Optimized for stylometric author attribution
"""

class Vocabulary:
    def __init__(self):
        self.min_freq = VOCABULARY_MIN_FREQ

        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        self.vocab_size = 0

        # Character n-gram support (FastText-style)
        self.use_char_ngrams = USE_CHAR_NGRAMS
        self.char_ngram_range = CHAR_NGRAM_RANGE

        # UNK token
        self.UNK = "<UNK>"
        self.unk_idx = None

        # Token regex: words + punctuation (stylometry-critical)
        self.TOKEN_REGEX = r"[a-zA-Z0-9']+|[.,!?;:—\"]"
    
    def get_char_ngram_ids(self, word):
        ngrams = self.get_char_ngrams(word)
        return [
            self.word2idx[ng]
            for ng in ngrams
            if ng in self.word2idx
        ]

    def get_char_ngrams(self, word):
        """
        Generate character n-grams with boundary symbols
        Only applied to alphanumeric words
        """
        word = f"<{word}>"
        min_n, max_n = self.char_ngram_range
        max_n = min(max_n, len(word))

        ngrams = [word]  # full word always included

        for n in range(min_n, max_n + 1):
            for i in range(len(word) - n + 1):
                ngrams.append(word[i:i+n])

        return ngrams

    def tokenize(self, text):
        """
        Tokenize text into:
        - lowercase words
        - punctuation tokens
        - optional char n-grams (for words only)
        """
        text = text.lower()
        tokens = re.findall(self.TOKEN_REGEX, text)

        final_tokens = []

        for tok in tokens:
            final_tokens.append(tok)

            # Add char n-grams ONLY for alphanumeric words
            if self.use_char_ngrams and tok.isalnum():
                final_tokens.extend(self.get_char_ngrams(tok))

        return final_tokens

    def build_vocab(self, text_data):
        """
        Build vocabulary with frequency filtering
        Adds <UNK> at index 0
        """
        print("Building vocabulary...")

        for i, text in enumerate(text_data):
            tokens = self.tokenize(text)
            self.word_counts.update(tokens)
            print(f"Document {i}: {len(tokens)} tokens")

        # Add UNK first
        idx = 0
        self.word2idx[self.UNK] = idx
        self.idx2word[idx] = self.UNK
        self.unk_idx = idx
        idx += 1

        # Stable ordering: most common tokens first
        for token, count in self.word_counts.most_common():
            if count >= self.min_freq:
                self.word2idx[token] = idx
                self.idx2word[idx] = token
                idx += 1

        self.vocab_size = len(self.word2idx)
        print(f"Vocabulary size (including UNK): {self.vocab_size}")

    def encode(self, text):
        """
        Convert text into indices (UNK for unseen tokens)
        """
        tokens = self.tokenize(text)
        return [
            self.word2idx.get(tok, self.unk_idx)
            for tok in tokens
        ]

    def get_negative_sampling_distribution(self):
        """
        Create smoothed unigram distribution for negative sampling
        UNK is excluded
        """
        print("Creating negative sampling distribution...")

        freq = np.zeros(self.vocab_size)

        for token, idx in self.word2idx.items():
            if idx == self.unk_idx:
                continue
            freq[idx] = self.word_counts.get(token, 1)

        freq = np.power(freq, NEG_SAMPLING_EXP_POWER)
        freq = freq / np.sum(freq)

        return freq