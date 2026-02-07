#!/bin/bash

cd "$(dirname "$0")"
set -e

if [ "$1" = "test1" ]; then
    TEST_FILE=$2
    OUTPUT_DIR=$3
    
    if [ -z "$TEST_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
        echo "Error: Missing arguments"
        echo "Usage: ./run_model.sh test1 <test_file> <output_dir>"
        exit 1
    fi
    
    echo "Running Task 1 inference..."
    python3 -m src.inference_task1 "$TEST_FILE" "$OUTPUT_DIR"
    
elif [ "$1" = "test2" ]; then
    TEST_FILE=$2
    OUTPUT_DIR=$3
    
    if [ -z "$TEST_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
        echo "Error: Missing arguments"
        echo "Usage: ./run_model.sh test2 <test_file> <output_dir>"
        exit 1
    fi
    
    echo "Running Task 2 inference..."
    python3 -m src.inference_task2 "$TEST_FILE" "$OUTPUT_DIR"
    
else
    TRAIN_DIR=$1
    
    if [ -z "$TRAIN_DIR" ]; then
        echo "Error: Missing train directory"
        exit 1
    fi
    
    SPLIT_TRAIN_DIR='split_data/train'
    
    # Only split if needed
    if [ ! -d "$SPLIT_TRAIN_DIR" ]; then
        echo "Splitting data..."
        python3 -m src.data_utils "$TRAIN_DIR"
    fi
    
    echo "Training model..."
    python3 -m src.train "$SPLIT_TRAIN_DIR"
    echo "Training completed!"
fi