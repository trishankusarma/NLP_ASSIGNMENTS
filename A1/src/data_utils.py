#!/usr/bin/env python3
"""
Sentence-based data split (FastText-style philosophy)
Training uses SENTENCES
Evaluation uses CHUNKS drawn from FULL corpus (train + test)
"""

import os
import glob
import json
import random
import re
import numpy as np
from collections import defaultdict

from config.hyper_parameters import (
    SEED,
    MINM_CHUNK_SIZE,
    MAXM_CHUNK_SIZE,
    TRAIN_RATIO
)

# Output
OUTPUT_DIR_TRAIN = 'split_data/train'
OUTPUT_DIR_TEST  = 'split_data/test'
TASK1_TEST_JSON  = 'task1_test.json'
TASK2_TEST_JSON  = 'task2_test.json'

# Evaluation scale
NUM_QUERIES = 100
NUM_CANDIDATES = 10

NUM_TASK2_CASES = 100
NUM_AUTHORS = 10
MINM_CHUNKS_PER_AUTHOR = 10

random.seed(SEED)
np.random.seed(SEED)

class DataSplitter:

    def __init__(self, data_dir):
        self.data_dir = data_dir

    # Sentence splitting
    def split_into_sentences(self, text):
        return [
            s.strip()
            for s in re.split(r'(?<=[.!?])\s+', text)
            if len(s.strip().split()) >= 5
        ]

    # Chunking (NO overlap, eval only)
    def text_to_chunks(self, text):
        words = text.split()
        chunks = []
        i = 0

        while i < len(words):
            size = random.randint(MINM_CHUNK_SIZE, MAXM_CHUNK_SIZE)
            chunk = words[i:i + size]

            if len(chunk) >= MINM_CHUNK_SIZE:
                chunks.append(" ".join(chunk))

            i += size

        return chunks

    # Load and split
    def load_and_split(self):
        train_data = defaultdict(list)
        eval_chunks = defaultdict(list)

        files = sorted(glob.glob(os.path.join(self.data_dir, "author_*.txt")))
        print(f"Found {len(files)} author files")

        for path in files:
            author = os.path.basename(path).replace(".txt", "")
            text = open(path, encoding="utf-8").read()

            sentences = self.split_into_sentences(text)
            random.shuffle(sentences)

            split_idx = int(len(sentences) * TRAIN_RATIO)
            train_sents = sentences[:split_idx]
            heldout_sents = sentences[split_idx:]

            # Training
            train_data[author] = train_sents

            # Evaluation pool = ALL text
            full_text = " ".join(train_sents + heldout_sents)
            eval_chunks[author] = self.text_to_chunks(full_text)

            print(
                f"{author}: "
                f"train sentences={len(train_sents)}, "
                f"eval chunks={len(eval_chunks[author])}"
            )

        return train_data, eval_chunks

    # Save training data
    def save_train_data(self, train_data):
        os.makedirs(OUTPUT_DIR_TRAIN, exist_ok=True)

        for author, sents in train_data.items():
            path = os.path.join(OUTPUT_DIR_TRAIN, f"{author}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(sents))
            print(f"Saved {path}")

    # Task 1 (200 queries)
    def generate_task1_test(self, eval_chunks):
        test_cases = []
        authors = list(eval_chunks.keys())

        while len(test_cases) < NUM_QUERIES:
            author = random.choice(authors)
            chunks = eval_chunks[author]

            if len(chunks) < 3:
                continue

            query = random.choice(chunks)
            pos = random.choice([c for c in chunks if c != query])

            candidates = {'cand_0': pos}
            others = [a for a in authors if a != author]

            for i in range(1, NUM_CANDIDATES):
                oa = random.choice(others)
                candidates[f'cand_{i}'] = random.choice(eval_chunks[oa])

            items = list(candidates.items())
            random.shuffle(items)
            gt = next(k for k, v in items if v == pos)

            test_cases.append({
                'query_id': f'query_{len(test_cases)}',
                'query_text': query,
                'candidates': dict(items),
                '_ground_truth': gt
            })

        return test_cases

    # Task 2 (200 clustering problems)
    def generate_task2_test(self, eval_chunks):
        valid_authors = [
            a for a in eval_chunks
            if len(eval_chunks[a]) >= MINM_CHUNKS_PER_AUTHOR
        ]

        cases = []

        for _ in range(NUM_TASK2_CASES):
            chosen = random.sample(valid_authors, NUM_AUTHORS)

            texts, labels = [], []
            for idx, author in enumerate(chosen):
                chunks = random.sample(
                    eval_chunks[author],
                    MINM_CHUNKS_PER_AUTHOR
                )
                for c in chunks:
                    texts.append(c)
                    labels.append(idx)

            combined = list(zip(texts, labels))
            random.shuffle(combined)
            texts, labels = zip(*combined)

            cases.append({
                'num_authors': NUM_AUTHORS,
                'min_chunks_per_author': MINM_CHUNKS_PER_AUTHOR,
                'chunks': list(texts),
                '_ground_truth': list(labels)
            })

        return cases

    # Save test files
    def save_tests(self, task1, task2):
        os.makedirs(OUTPUT_DIR_TEST, exist_ok=True)

        with open(os.path.join(OUTPUT_DIR_TEST, TASK1_TEST_JSON), "w") as f:
            json.dump(task1, f, indent=2)

        with open(os.path.join(OUTPUT_DIR_TEST, TASK2_TEST_JSON), "w") as f:
            json.dump(task2, f, indent=2)

        print("Saved test files")


def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python data_utils.py <data_dir>")
        return

    splitter = DataSplitter(sys.argv[1])

    train_data, eval_chunks = splitter.load_and_split()
    splitter.save_train_data(train_data)

    task1 = splitter.generate_task1_test(eval_chunks)
    task2 = splitter.generate_task2_test(eval_chunks)

    splitter.save_tests(task1, task2)


if __name__ == "__main__":
    main()
