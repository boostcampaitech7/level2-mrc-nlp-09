#!/bin/bash

# Set environment variables
export RETRIEVER_MODEL_DIR="models/retriever/"
export READER_MODEL_DIR="models/reader/"
export EVAL_DATA_DIR="data/eval/"

# Run Python evaluation script
python3 evaluate.py --retriever_model_dir $RETRIEVER_MODEL_DIR --reader_model_dir $READER_MODEL_DIR --eval_data_dir $EVAL_DATA_DIR
