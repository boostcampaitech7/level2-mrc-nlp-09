#!/bin/bash

# Set environment variables
export RETRIEVER_MODEL_DIR="models/retriever/"
export READER_MODEL_DIR="models/reader/"
export INFERENCE_DATA_DIR="data/inference/"

# Run Python inference script
python3 inference.py --retriever_model_dir $RETRIEVER_MODEL_DIR --reader_model_dir $READER_MODEL_DIR --inference_data_dir $INFERENCE_DATA_DIR
