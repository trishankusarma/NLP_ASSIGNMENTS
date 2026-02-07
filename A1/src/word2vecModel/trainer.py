import torch
import torch.nn as nn
import numpy as np
from collections import Counter
import time
import json
import os
import sys
import pickle
from .vocabulary import Vocabulary
from .skip_gram_model import SkipGramModel
from src.utils import (
    plot_loss_curve, task1_inference
)
from config.hyper_parameters import (
    EMBEDDING_DIM,
    WINDOW_SIZE,
    NUM_NEGATIVE_SAMPLES,
    LEARNING_RATE,
    EPOCHS,
    BATCH_SIZE
)
from tqdm import tqdm
from src.AuthorAttribution import AuthorAttributor

# getting gpu
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(f"Using device: {device}")

TASK_1_VAL_DIR = './split_data/test/task1_test.json'

""" Word2Vec Model Class - Trainer for the Skip-Gram model with Negative Sampling """
class Word2VecTrainer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.embedding_dim = EMBEDDING_DIM
        self.window_size = WINDOW_SIZE
        self.num_negative_samples = NUM_NEGATIVE_SAMPLES
        self.epochs = EPOCHS
        self.batch_size = BATCH_SIZE
        
        self.model = SkipGramModel(vocab.vocab_size, EMBEDDING_DIM).to(device)
        self.initial_lr = LEARNING_RATE

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.initial_lr
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        # Negative sampling distribution
        self.neg_dist = vocab.get_negative_sampling_distribution()

        # Create Noise Table
        print("Pre-computing Noise Table for speed...")
        table_size = 10**7  # Standard size for Word2Vec
        
        # We sample indices proportional to the negative sampling distribution
        # This only happens ONCE during initialization
        self.noise_table = np.random.choice(
            self.vocab.vocab_size, 
            size=table_size, 
            p=self.neg_dist
        ).astype(np.int32)
        
        self.table_ptr = 0 # Pointer to track where we are in the table

    def get_negative_samples(self, num_samples):
        """Ultra-fast sampling from pre-computed noise table"""
        # If we reach the end of the table, wrap around to the start
        if self.table_ptr + num_samples >= len(self.noise_table):
            self.table_ptr = 0
            np.random.shuffle(self.noise_table) # Optional: reshuffle for variety

        samples = self.noise_table[self.table_ptr : self.table_ptr + num_samples]
        self.table_ptr += num_samples
        return samples
    
    def generate_training_batches(self, input_text):
        centers, contexts = [], []
        for text in input_text:
            indices = self.vocab.encode(text)
            for i, center_word_idx in enumerate(indices):
                start = max(0, i - self.window_size)
                end = min(len(indices), i + self.window_size + 1)
                for j in range(start, end):
                    if i != j:
                        centers.append(center_word_idx)
                        contexts.append(indices[j])
                        if len(centers) == self.batch_size:
                            yield (np.array(centers, dtype=np.int32), 
                                   np.array(contexts, dtype=np.int32))
                            centers, contexts = [], []
        
        # FINAL YIELD: Catch the last partial batch if it exists
        if len(centers) > 0:
            yield (np.array(centers, dtype=np.int32), 
                   np.array(contexts, dtype=np.int32))
        # return centers, contexts

    def train(self, given_texts, save_dir = '../output/training_loss_curve.png'):
        """Train the Word2Vec model with generator-based batches"""
        
        # Load test data for validation
        print(f"Loading test data from ...{TASK_1_VAL_DIR}")
        with open(TASK_1_VAL_DIR, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        
        attributor = AuthorAttributor(self.model, self.vocab)
        self.model.train()
        track_losses = []

        for epoch in range(self.epochs):
            # 1. Shuffle documents at the start of each epoch for randomness
            np.random.shuffle(given_texts)

            total_loss = 0
            start_time = time.time()

            # 2. Use the generator and enumerate to get batch_id
            batch_gen = self.generate_training_batches(given_texts)
            
            # Note: We use enumerate here so (batch_id+1) % 10000 works
            for batch_id, (centers_np, contexts_np) in enumerate(tqdm(batch_gen, desc=f"Epoch {epoch+1}")): 
                
                current_batch_size = centers_np.shape[0] 

                center_words = torch.from_numpy(centers_np).long().to(device)
                context_words = torch.from_numpy(contexts_np).long().to(device)

                # USE current_batch_size INSTEAD OF self.batch_size
                num_neg = current_batch_size * self.num_negative_samples
                neg_samples_np = self.get_negative_samples(num_neg).reshape(
                    current_batch_size, self.num_negative_samples
                )
                negative_samples = torch.from_numpy(neg_samples_np).long().to(device)

                self.optimizer.zero_grad()
                loss = self.model(center_words, context_words, negative_samples)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

                track_losses.append(loss.item())
                total_loss += loss.item()

                # Periodic Logging
                if (batch_id + 1) % 2000 == 0:
                    avg = np.mean(track_losses[-100:])
                    time_elapsed = time.time() - start_time
                    tqdm.write(f"Batch : {batch_id+1} :: loss : {avg:.4f} :: Time : {time_elapsed:.2f}s")
            
            # 3. Post-epoch updates
            self.scheduler.step()
            task1_inference(attributor, queries)
            
            avg_epoch_loss = total_loss / (batch_id + 1)
            print(f"Epoch {epoch+1} completed :: Avg_loss: {avg_epoch_loss:.4f}")
    
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
            
        model = SkipGramModel(vocab.vocab_size, embedding_dim).to(device)
        model.load_state_dict(save_dict['model_state'])
        model.eval()
            
        return model, vocab, save_dict