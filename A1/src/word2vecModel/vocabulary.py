import numpy as np
from collections import Counter
import re

from config.hyper_parameters import (
    VOCABULARY_MIN_FREQ,
    NEG_SAMPLING_EXP_POWER,
    USE_CHAR_NGRAMS,
    CHAR_NGRAM_RANGE,
    USE_WORDS_NGRAMS,
    WORD_NGRAM_RANGE
)

""" This class manages the Vocabulary structure for training """
class Vocabulary:
    def __init__(self):
        self.min_freq = VOCABULARY_MIN_FREQ
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        self.vocab_size = 0

        # for n-gram support
        self.use_char_ngrams = USE_CHAR_NGRAMS
        self.char_ngram_range = CHAR_NGRAM_RANGE
        self.use_word_ngrams = USE_WORDS_NGRAMS
        self.word_ngram_range = WORD_NGRAM_RANGE

    def get_char_ngrams(self, word):
        """
        Get character n-grams for a word (FastText-style)
        Adds special boundary symbols <> to capture prefixes/suffixes
        "quick" → "<qu", "qui", "uic", "ick", "ck>"
        """
        word = f"<{word}>"
        char_ngrams = []
        min_n, max_n = self.char_ngram_range
        
        for n in range(min_n, max_n + 1):
            for i in range(len(word) - n + 1):
                ngram = word[i:i+n]
                char_ngrams.append(ngram)
        
        return char_ngrams
    
    def get_word_ngrams(self, words):
        """
        Get word n-grams from a list of words
        Example: ["the", "quick", "brown"] -> ["the", "quick", "brown", "the_quick", "quick_brown"]
        """
        word_ngrams = []
        min_n, max_n = self.word_ngram_range
        
        for n in range(min_n, max_n + 1):
            for i in range(len(words) - n + 1):
                ngram = "_".join(words[i:i+n])
                word_ngrams.append(ngram)
        
        return word_ngrams

    def tokenize(self, text):
        """
        Tokenization with n-gram support
        Returns a list of tokens (words, character n-grams, or word n-grams)
        """
        # Basic tokenization: lowercase and extract words
        text = text.lower()
        # Keep alphanumeric and basic punctuation
        TOKEN_REGEX = r"[a-zA-Z0-9']+|[.,!?;:]"
        words = re.findall(TOKEN_REGEX, text.lower())
        
        tokens = []
        
        # Adding word n-grams if enabled
        if self.use_word_ngrams:
            tokens.extend(self.get_word_ngrams(words))
        else:
            # Just use individual words
            tokens.extend(words)
        
        # Adding character n-grams if enabled (for each word)
        if self.use_char_ngrams:
            for word in words:
                char_ngrams = self.get_char_ngrams(word)
                tokens.extend(char_ngrams)
        
        return tokens
        
    def build_vocab(self, given_text_data):
        """Build vocabulary from given_text_data"""
        
        print("Tokenizing the given text data and buidling the vocabulary")
        # Counting words
        for index, text in enumerate(given_text_data):
            tokens = self.tokenize(text)
            self.word_counts.update(tokens)

            print(f"Tokens generated for document {index} is len(tokens)")
        
        # Filter by min_freq and build mappings
        idx = 0
        for word, count in self.word_counts.items():
            if count >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word 
                idx += 1
        
        self.vocab_size = len(self.word2idx)
        print(f"Overall Vocabulary size: {self.vocab_size}")
    
    def get_negative_sampling_distribution(self):
        """Create distribution for negative sampling (raised to 3/4 power)"""
        
        print("Creating distribution for negative sampling")
        freq = np.zeros(self.vocab_size)
        for word, idx in self.word2idx.items():
            freq[idx] = self.word_counts[word]
        
        # Smooth the distribution
        freq = np.power(freq, NEG_SAMPLING_EXP_POWER)
        freq = freq / np.sum(freq)
        return freq
    
    def encode(self, input_text):
        """Convert text to list of indices :: this is for each text document """
        tokens = self.tokenize(input_text)
        return [self.word2idx[word] for word in tokens if word in self.word2idx]