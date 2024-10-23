#!/bin/bash

set +e

python src/retrieval_test_gteMultibase_pipeline.py --pooling max || true

python src/retrieval_test_bge-reranker_pipeline.py || true

python src/retrieval_test_ko-reranker_pipeline.py || true