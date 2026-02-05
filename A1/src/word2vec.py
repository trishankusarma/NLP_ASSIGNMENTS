import torch
import torch.nn as nn
import numpy as np
from collections import Counter
import time
import os

from config.hyper_parameters import (
    VOCABULARY_MIN_FREQ,
    EMBEDDING_DIM,
    WINDOW_SIZE,
    NUM_NEGATIVE_SAMPLES,
    LEARNING_RATE,
    EPOCHS,
    BATCH_SIZE,
    NEG_SAMPLING_EXP_POWER
)

""" This class manages the Vocabulary structure for training """
class Vocabulary:
    def __init__(self):
        self.min_freq = VOCABULARY_MIN_FREQ
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        self.vocab_size = 0
    
    def tokenize(self, text):
        """Tokenizer :: Convert to lowercase and split the text based on whitespaces."""
        """Will update this function accordingly"""
        tokens = text.lower().split()
        return tokens
        
    def build_vocab(self, given_text_data):
        """Build vocabulary from given_text_data"""
        
        print("Tokenizing the given text data and buidling the vocabulary")
        # Counting words
        for text in given_text_data:
            tokens = self.tokenize(text)
            self.word_counts.update(tokens)
        
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

"""Skip-gram Word2Vec model implementation"""
class SkipGramModel(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Input embeddings (center word)
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Output embeddings (context word)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize embeddings
        self.in_embeddings.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)
        self.out_embeddings.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)

    def forward(self, center_words, context_words, negative_samples):
        """
        Args:
            center_words: tensor of shape (batch_size,)
            context_words: tensor of shape (batch_size,)
            negative_samples: tensor of shape (batch_size, num_negative_samples)
        Returns:
            loss: scalar tensor
        """
        batch_size = center_words.size(0)
        
        # Get embeddings
        center_embeds = self.in_embeddings(center_words)  # (batch_size, embedding_dim)
        context_embeds = self.out_embeddings(context_words)  # (batch_size, embedding_dim)
        neg_embeds = self.out_embeddings(negative_samples)  # (batch_size, num_neg, embedding_dim)
        
        eps = 1e-10
        # Positive score
        # dot product of center_words and context words 
        pos_score = torch.sum(center_embeds * context_embeds, dim=1)  # (batch_size,) 
        # Goal: Maximize probability that context word appears near center word :: So Minimize :: -log( sigmoid(pos_score) + eps )
        pos_loss = -torch.log(torch.sigmoid(pos_score) + eps) 
                
        # Negative scores
        # neg_embeds = (b, n, k) and center_embeds = (b, k) -> center_embeds.unsqueeze(2) -> (b, k, 1)
        # bmm -> batch dot product -> (b, n, 1) -> squeeze -> (b, n)
        neg_score = torch.bmm(neg_embeds, center_embeds.unsqueeze(2)).squeeze(2)  # (batch_size, num_neg)
        # Goal: Maximize probability that negative words DON'T appear near center word :: So Minimize -log(sigmoid(-neg_score) + eps )
        neg_loss = -torch.sum(torch.log(torch.sigmoid(-neg_score) + eps), dim=1)
        
        # Total loss
        loss = torch.mean(pos_loss + neg_loss)
        return loss

""" Word2Vec Model Class - Trainer for the Skip-Gram model with Negative Sampling """
class Word2VecTrainer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.embedding_dim = EMBEDDING_DIM
        self.window_size = WINDOW_SIZE
        self.num_negative_samples = NUM_NEGATIVE_SAMPLES
        self.learning_rate = LEARNING_RATE
        self.epochs = EPOCHS
        self.batch_size = BATCH_SIZE
        
        self.model = SkipGramModel(vocab.vocab_size, EMBEDDING_DIM)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
        # Negative sampling distribution
        self.neg_dist = vocab.get_negative_sampling_distribution()
    
    def generate_training_data(self, input_text):
        """Generate (center, context) pairs from input_text"""
        print("Generating training pairs...(center_word, context_word)")
        """[text_doc1, text_doc2,...., text_docn]"""
        training_pairs = []
        for text in input_text:
            # now for each text document
            indices = self.vocab.encode(text)
            for i, center_word_idx in enumerate(indices):
                # Get context window :: fetching the start and end idx of each center word
                start = max(0, i - self.window_size)
                end = min(len(indices), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i != j:
                        context = indices[j]
                        training_pairs.append((center_word_idx, context)) # considering skip-gram
                
        # Finally we can expect window*vocab_size of pairs
        print(f"Generated {len(training_pairs)} training pairs")
        return training_pairs

    def train(self, given_texts):
        """Train the Word2Vec model"""
        training_pairs = self.generate_training_data(given_texts)
        
        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0
            np.random.shuffle(training_pairs)
            num_batches = len(training_pairs) // self.batch_size

            start_time = time.time()

            for batch_id in range(num_batches):
                # step 1: prepare batch
                start_id = batch_id * self.batch_size
                end_id = start_id + self.batch_size

                # this is our training batch
                curr_batch = training_pairs[start_id:end_id]
                
                # step 2: separating the center words and context words
                center_words = torch.tensor([pair[0] for pair in curr_batch], dtype=torch.long)
                context_words = torch.tensor([pair[1] for pair in curr_batch], dtype=torch.long)

                # step 3: sample negative pairs
                negative_samples = np.random.choice(
                    self.vocab.vocab_size,
                    size=(self.batch_size, self.num_negative_samples),
                    p=self.neg_dist
                )
                negative_samples = torch.tensor(negative_samples, dtype=torch.long) # convert to long format

                # Step 4: Forward pass
                self.optimizer.zero_grad()
                loss = self.model(center_words, context_words, negative_samples)
                
                # Step 5: Backward pass
                loss.backward()
                # addding gradient cliping to avoid exploding gradient
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                
                total_loss += loss.item()

                if (batch_id+1) % 100 == 0:
                    time_elapsed = time.time() - start_time
                    print(f"Epoch: {epoch+1}/{self.epochs} :: Batch : {batch_id+1}/{num_batches} :: loss : {loss.item():.4f} :: Cummulative time elapsed : {time_elapsed:.2f}s")
            
            overall_time_for_batch = time.time() - start_time
            avg_loss = total_loss / num_batches
            print(f"Epoch : {epoch+1} completed :: Avg_loss : {avg_loss} :: Overall time taken for batch : {overall_time_for_batch:.2f}s")
    
    def save_model(self, path):
        """Save model and vocabulary"""
        
        print("Saving model...")
        # Create directory if it does not exist
        dir_name = os.path.dirname(path)
        if dir_name != "":
            os.makedirs(dir_name, exist_ok=True)

        # keep modifying accordingly
        save_dict = {
            'model_state': self.model.state_dict(),
            'vocab': self.vocab,
            'embedding_dim': self.embedding_dim,
            'window_size': self.window_size
        }
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"Model saved to {path}")
    
@staticmethod
def load_model(path):
    """Load model and vocabulary"""
    with open(path, 'rb') as f:
        save_dict = pickle.load(f)
        
    vocab = save_dict['vocab']
    embedding_dim = save_dict['embedding_dim']
        
    model = SkipGramModel(vocab.vocab_size, embedding_dim)
    model.load_state_dict(save_dict['model_state'])
    model.eval()
        
    return model, vocab, save_dict


    
