#!/usr/bin/env python3
"""
Utility to split author data into train/test sets
and generate synthetic test files for Task 1 and Task 2
"""
import os
import glob
import json
import random
import numpy as np
from collections import defaultdict

from config.hyper_parameters import (
    OVERLAP,
    SEED,
    MINM_CHUNK_SIZE,
    MAXM_CHUNK_SIZE,
    TRAIN_RATIO
)

# Output directories after spliting the data
OUTPUT_DIR_TRAIN = 'split_data/train'
OUTPUT_DIR_TEST = 'split_data/test'
TASK1_TEST_JSON = 'task1_test.json'
TASK2_TEST_JSON = 'task2_test.json'

# Inference tasks options
NUM_QUERIES = 20 
NUM_CANDIDATES = 10
NUM_AUTHORS = 5
MINM_CHUNKS_PER_AUTHOR = 5

class DataSplitter:
    """Split author texts into train/test and generate test files"""
    
    def __init__(self, data_dir, overlap=OVERLAP, seed=SEED):
        """
        Args:
            data_dir: directory containing author_XXX.txt files
            chunk_size: number of words per chunk
            overlap: overlapping words between chunks
            seed: random seed for reproducibility
        """
        self.data_dir = data_dir
        self.overlap = overlap
        random.seed(seed)
        np.random.seed(seed)
    
    def load_author_data(self):
        """Load all author files and create chunks"""
        author_chunks = defaultdict(list)
        
        file_paths = sorted(glob.glob(os.path.join(self.data_dir, 'author_*.txt')))
        print(f"Found {len(file_paths)} author files")
        
        for file_path in file_paths:
            author_id = os.path.basename(file_path).replace('.txt', '')
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Split into chunks
            chunks = self.text_to_chunks(text)
            author_chunks[author_id] = chunks
            print(f"{author_id}: {len(chunks)} chunks")
        
        return author_chunks
    
    def text_to_chunks(self, text, min_chunk_size=MINM_CHUNK_SIZE, max_chunk_size=MAXM_CHUNK_SIZE):
        """Split text into overlapping chunks of varying sizes"""
        words = text.split()
        chunks = []
        
        i = 0
        while i < len(words):
            # Variable chunk size between min and max
            current_chunk_size = random.randint(min_chunk_size, max_chunk_size)
            
            chunk_words = words[i:i + current_chunk_size]
            if len(chunk_words) >= min_chunk_size:
                chunks.append(' '.join(chunk_words))
            
            # Move forward with some overlap
            i += current_chunk_size - self.overlap
        
        return chunks

    def split_data(self, author_chunks, train_ratio=TRAIN_RATIO):
        """
        Split each author's chunks into train/val/test
        
        Returns:
            train_data, test_data: each is a dict {author_id: [chunks]}
        """
        train_data = {}
        test_data = {}
        
        for author_id, chunks in author_chunks.items():
            n = len(chunks)
            
            # Shuffle chunks
            shuffled_chunks = chunks.copy()
            random.shuffle(shuffled_chunks)
            
            # Split indices
            train_end = int(n * train_ratio)
            
            train_data[author_id] = shuffled_chunks[:train_end]
            test_data[author_id] = shuffled_chunks[train_end:]
            
            print(f"{author_id}: train={len(train_data[author_id])}, "
                  f"test={len(test_data[author_id])}")
        
        return train_data, test_data
    
    def save_train_data(self, train_data, output_dir):
        """Save training data as separate author files"""
        os.makedirs(output_dir, exist_ok=True)
        
        for author_id, chunks in train_data.items():
            output_path = os.path.join(output_dir, f"{author_id}.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(chunks))
            print(f"Saved {output_path}")

    def generate_task1_test(self, test_data, num_queries=NUM_QUERIES, num_candidates=NUM_CANDIDATES):
        """
        Generate Task 1 test file (author verification)
        
        For each query:
        - Pick a random chunk from a random author
        - Pick candidate chunks (1 from same author, rest from different authors)
        """
        test_cases = []
        
        author_ids = list(test_data.keys())
        
        for query_idx in range(num_queries):
            # Pick random author and chunk for query
            query_author = random.choice(author_ids)
            
            if len(test_data[query_author]) < 2:
                continue  # Need at least 2 chunks (1 for query, 1 for candidate)
            
            query_chunk = random.choice(test_data[query_author])
            
            # Remove query chunk temporarily
            remaining_chunks = [c for c in test_data[query_author] if c != query_chunk]
            
            # Pick 1 chunk from same author
            if remaining_chunks:
                same_author_chunk = random.choice(remaining_chunks)
            else:
                continue
            
            # Pick chunks from different authors
            different_author_chunks = []
            other_authors = [a for a in author_ids if a != query_author]
            
            for _ in range(num_candidates - 1):
                if not other_authors:
                    break
                other_author = random.choice(other_authors)
                if test_data[other_author]:
                    chunk = random.choice(test_data[other_author])
                    different_author_chunks.append(chunk)
            
            # Create candidates dict
            candidates = {}
            candidates['cand_0'] = same_author_chunk  # Correct answer
            
            for i, chunk in enumerate(different_author_chunks):
                candidates[f'cand_{i+1}'] = chunk
            
            # Shuffle candidate order (so correct answer isn't always first)
            items = list(candidates.items())
            random.shuffle(items)

            correct_key = None
            for k, v in items:
                if v == same_author_chunk:
                    correct_key = k
                    break  

            candidates = dict(items)
            
            test_case = {
                'query_id': f'query_{query_idx}',
                'query_text': query_chunk,
                'candidates': candidates,
                '_ground_truth': correct_key  # For evaluation (will be removed in actual test)
            }
            
            test_cases.append(test_case)
        
        return test_cases
    
    def generate_task2_test(self, test_data, num_authors = NUM_AUTHORS, min_chunks_per_author = MINM_CHUNKS_PER_AUTHOR):
        """
        Generate Task 2 test file (author chunks clustering)
        
        For each query:
        - Pick randomly some n number of authors given
        - For each author, pick minm m chunks of test data written by that author
        - Store the result for evaluation
        - Randomly suffle
        """
        test_cases = []
        
        author_ids = list(test_data.keys())

        valid_author_ids = [author_id for author_id in author_ids if len(test_data[author_id]) >= min_chunks_per_author]

        if(len(valid_author_ids) < num_authors):
            print(f"Number of authors requested are more than the available ones which counts to {len(valid_author_ids)}")
        
        random.shuffle(valid_author_ids)
        picked_author_ids = valid_author_ids[:num_authors]

        picked_chunks = []
        ground_truths = []

        for cluster_id, author_id in enumerate(picked_author_ids):
            # randomly check how many chunks can be picked 
            num_chunks_to_pick = random.randint(min_chunks_per_author, min_chunks_per_author*2)
            author_chunks = test_data[author_id].copy()
            random.shuffle(author_chunks)
            curr_author_chunks = author_chunks[:num_chunks_to_pick]  # Take first N chunks

            for chunk in curr_author_chunks:
                picked_chunks.append(chunk)
                ground_truths.append(cluster_id)
        
        # suffle while maintaining ground_truth allignments
        combined_data = list(zip(picked_chunks, ground_truths))
        random.shuffle(combined_data)
        picked_chunks, ground_truths = zip(*combined_data)

        test_case = {
            'num_authors': num_authors,
            'min_chunks_per_author': min_chunks_per_author,
            'chunks': list(picked_chunks),
            '_ground_truth': list(ground_truths)  # For evaluation (will be removed in actual test)
        }
        
        return test_case

    def save_test_files(self, task1_cases, task2_case, output_dir):
        """Save test files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save Task 1 test file (with ground truth for evaluation)
        task1_path = os.path.join(output_dir, TASK1_TEST_JSON)
        with open(task1_path, 'w', encoding='utf-8') as f:
            # Save all queries in a list
            json.dump(task1_cases, f, indent=2)
        print(f"Saved Task 1 test file: {task1_path} ({len(task1_cases)} queries)")
        
        # Save Task 2 test file (with ground truth for evaluation)
        task2_path = os.path.join(output_dir, TASK2_TEST_JSON)
        with open(task2_path, 'w', encoding='utf-8') as f:
            json.dump(task2_case, f, indent=2)
        print(f"Saved Task 2 test file: {task2_path} ({len(task2_case['chunks'])} chunks)")

def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python data_utils.py <data_dir>")
        sys.exit(1)

    data_dir = sys.argv[1]

    # Creating train-test split
    splitter = DataSplitter(data_dir)
    
    # Load author data
    print("\n[1/6] Loading author data...")
    author_chunks = splitter.load_author_data()

    # Split data
    print(f"\n[2/6] Splitting data into Train Ratio: {TRAIN_RATIO}")
    train_data, test_data = splitter.split_data(author_chunks)

    # Save training data
    print("\n[3/6] Saving training data...")
    splitter.save_train_data(train_data, OUTPUT_DIR_TRAIN)

    # Generate Task 1 test cases
    print("\n[4/6] Generating Task 1 test cases...")
    task1_cases = splitter.generate_task1_test(test_data)

    # Generate Task 2 test case
    print("\n[5/6] Generating Task 2 test case...")
    task2_case = splitter.generate_task2_test(test_data)

     # Save test files
    print("\n[6/6] Saving test files...")
    splitter.save_test_files(task1_cases, task2_case, OUTPUT_DIR_TEST)

if __name__ == '__main__':
    main()