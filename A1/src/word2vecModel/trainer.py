import torch
import torch.nn as nn
import numpy as np
from collections import Counter
import time
import os
from .vocabulary import Vocabulary
from .skip_gram_model import SkipGramModel
from src.utils import (
    plot_loss_curve
)
from config.hyper_parameters import (
    EMBEDDING_DIM,
    WINDOW_SIZE,
    NUM_NEGATIVE_SAMPLES,
    LEARNING_RATE,
    EPOCHS,
    BATCH_SIZE
)

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

    def train(self, given_texts, save_dir = '../output/training_loss_curve.png'):
        """Train the Word2Vec model"""
        training_pairs = self.generate_training_data(given_texts)
        
        self.model.train()
        track_losses = []

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

                track_losses.append(loss.item())
                
                total_loss += loss.item()

                if (batch_id+1) % 100 == 0:
                    time_elapsed = time.time() - start_time
                    print(f"Epoch: {epoch+1}/{self.epochs} :: Batch : {batch_id+1}/{num_batches} :: loss : {loss.item():.4f} :: Cummulative time elapsed : {time_elapsed:.2f}s")
            
            overall_time_for_batch = time.time() - start_time
            avg_loss = total_loss / num_batches
            print(f"Epoch : {epoch+1} completed :: Avg_loss : {avg_loss} :: Overall time taken for batch : {overall_time_for_batch:.2f}s")
        
        # Plot loss curves
        plot_loss_curve(track_losses, save_dir)
    
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