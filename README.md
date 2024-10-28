# RAG based Open-Domain Question Answering (ODQA with RAG)
his project utilizes a three-stage Retrieval-Augmented Generation (RAG) pipeline to extract answers from large document collections effectively. The system combines Sparse and Dense Retrieval, Reranking, and Machine Reading Comprehension (MRC) to provide accurate answers from Wikipedia. Each stage is managed in a separate branch, with this README giving an overview and directing to each branch for in-depth setup and usage.

## Overview
### 1. Environment Setup
To replicate the environment used for each model, refer to the branch-specific environment.yml files. You can set up the environment by running:
```console
$ conda env create -f environment.yml
```

System requirements:
Ubuntu-20.04.6 LTS

Each branch (main-Retrieval, main-Reranker, main-MRC) details the specific Python and PyTorch versions for its corresponding model.


### 2. Data
에베베


### 3. Pipeline Stages

#### Retrieval
Branch: main-Retrieval

Filters 100 relevant documents per query from a large Wikipedia corpus.

#### Reranker
Branch: main-Reranker

Reranks the top 100 documents per query. The output can be analyzed for performance and is passed to the MRC stage.

#### MRC
Branch: main-MRC

Processes the final top-k reranked documents per query, extracting exact answers from the contexts using a machine reading comprehension model.

For additional information, refer to each branch’s README detailing installation, environment specifics, and model usage instructions.

