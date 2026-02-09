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

torch.set_num_threads(8)
torch.set_num_interop_threads(8)

device = "cpu"

TASK_1_VAL_DIR = './split_data/test/task1_test.json'

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

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.initial_lr
        )

        self.neg_dist = vocab.get_negative_sampling_distribution()

        print("Pre-computing Noise Table...")
        table_size = 10**7
        self.noise_table = np.random.choice(
            self.vocab.vocab_size,
            size=table_size,
            p=self.neg_dist
        ).astype(np.int32)

        self.table_ptr = 0

        # Cache punctuation indices (used only for center-word filtering)
        self.punct_indices = {
            idx for tok, idx in vocab.word2idx.items()
            if not tok.isalnum() and tok != vocab.UNK
        }

    def get_negative_samples(self, num_samples):
        if self.table_ptr + num_samples >= len(self.noise_table):
            self.table_ptr = 0
            np.random.shuffle(self.noise_table)

        samples = self.noise_table[self.table_ptr:self.table_ptr + num_samples]
        self.table_ptr += num_samples
        return samples

    def generate_training_batches(self, input_texts):
        centers, contexts = [], []

        for text in input_texts:
            indices = self.vocab.encode(text)

            for i, center_idx in enumerate(indices):

                # Skip UNK as center
                if center_idx == self.vocab.unk_idx:
                    continue

                # Skip punctuation as center
                if center_idx in self.punct_indices:
                    continue

                # Dynamic window 
                window = np.random.randint(1, self.window_size + 1)
                start = max(0, i - window)
                end = min(len(indices), i + window + 1)

                for j in range(start, end):
                    if i == j:
                        continue

                    centers.append(center_idx)
                    contexts.append(indices[j])

                    if len(centers) == self.batch_size:
                        yield (
                            np.array(centers, dtype=np.int32),
                            np.array(contexts, dtype=np.int32)
                        )
                        centers, contexts = [], []

        if len(centers) > 0:
            yield (
                np.array(centers, dtype=np.int32),
                np.array(contexts, dtype=np.int32)
            )

    def train(self, given_texts, save_dir='../output/training_loss_curve.png'):
        print(f"Loading validation data from {TASK_1_VAL_DIR}")
        with open(TASK_1_VAL_DIR, 'r', encoding='utf-8') as f:
            queries = json.load(f)

        attributor = AuthorAttributor(self.model, self.vocab, training_texts=given_texts)
        self.model.train()
        track_losses = []

        for epoch in range(self.epochs):
            lr = self.initial_lr * (1 - epoch / self.epochs)
            lr = max(lr, 1e-5)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            print(f"\nEpoch {epoch+1}/{self.epochs} | LR = {lr:.6f}")

            np.random.shuffle(given_texts)
            total_loss = 0
            batch_count = 0
            start_time = time.time()

            batch_gen = self.generate_training_batches(given_texts)

            for centers_np, contexts_np in tqdm(batch_gen, ncols=100):
                current_batch_size = centers_np.shape[0]
                if current_batch_size < 4:
                    continue

                center_words = torch.from_numpy(centers_np).long().to(device)
                context_words = torch.from_numpy(contexts_np).long().to(device)

                num_neg = current_batch_size * self.num_negative_samples
                neg_samples_np = self.get_negative_samples(num_neg).reshape(
                    current_batch_size, self.num_negative_samples
                )
                negative_samples = torch.from_numpy(neg_samples_np).long().to(device)

                self.optimizer.zero_grad()
                loss = self.model(center_words, context_words, negative_samples)
                loss.backward()
                self.optimizer.step()

                # Embedding normalization (important for clustering)
                with torch.no_grad():
                    self.model.in_embeddings.weight.div_(
                        self.model.in_embeddings.weight.norm(dim=1, keepdim=True) + 1e-8
                    )

                track_losses.append(loss.item())
                total_loss += loss.item()
                batch_count += 1

            print(
                f"Epoch {epoch+1} done | "
                f"Avg Loss = {total_loss / max(batch_count, 1):.4f} | "
                f"Time = {time.time() - start_time:.2f}s"
            )

            print("Validating on Task 1...")
            task1_inference(attributor, queries)

        plot_loss_curve(track_losses, save_dir)
        print("Training complete")

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model_state': self.model.state_dict(),
                'vocab': self.vocab,
                'embedding_dim': self.embedding_dim,
                'window_size': self.window_size
            }, f)

    @staticmethod
    def load_model(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)

        vocab = data['vocab']
        model = SkipGramModel(vocab.vocab_size, data['embedding_dim']).to(device)
        model.load_state_dict(data['model_state'])
        model.eval()

        return model, vocab, data
