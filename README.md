# RAG based Open-Domain Question Answering (ODQA with RAG)
This project utilizes a three-stage Retrieval-Augmented Generation (RAG) pipeline to extract answers from large document collections effectively. The system combines Sparse and Dense Retrieval, Reranking, and Machine Reading Comprehension (MRC) to provide accurate answers from Wikipedia. Each stage is managed in a separate branch, with this README giving an overview and directing to each branch for in-depth setup and usage.

MORE : [ODQA with RAG](https://github.com/boostcampaitech7/level2-mrc-nlp-09/blob/main/RAG%20based%20Open-Domain%20Question%20Answering.pdf)

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
**The Wikipedia document corpus** used in the Retrieval stage is stored at ./data/raw/wikipedia_documents.json, comprising approximately 57,000 unique documents.

**The MRC dataset** used in the MRC stage is stored at ./data/raw/train_dataset and ./data/raw/test_dataset.  

Sample Data Structure Overview:  
![8acf9df5-ea23-4ad2-a24c-c9a847d24e59](https://github.com/user-attachments/assets/c3a69377-34e7-49d7-828c-a93977baa42d)

Sample Document Format:  
![00129d26-34fd-4313-9579-e088665239bf](https://github.com/user-attachments/assets/a812e35d-93e6-42c8-808f-37d683337e73)


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

### 4. Evaluation Metrics
On the Leader Board, EM would be the main metric for the task.

Exact Match (EM): A score is awarded only when the model's prediction exactly matches the true answer.(Each question is therefore scored as either 0 or 1)

F1 Score: Unlike EM, F1 Score gives partial score. For instance, if the correct answer is "Barack Obama" but the prediction is "Obama," the EM score would be 0, while the F1 Score considers overlapping words and would provide partial score.

### 5. Results

EM: 62.92%
F1 Score: 73.46%

*These metrics provide insight into both the accuracy and partial correctness of the model’s predictions across all stages of the pipeline.
