#!/bin/bash

# Set environment variables
export DATA_DIR="data/"
export MODEL_DIR="models/reader/"
export EPOCHS=10
export BATCH_SIZE=32

# Create model directory if it doesn't exist
mkdir -p $MODEL_DIR

# Run Python training script
python3 train_reader.py --data_dir $DATA_DIR --model_dir $MODEL_DIR --epochs $EPOCHS --batch_size $BATCH_SIZE
