# RAG based Open-Domain Question Answering (ODQA with RAG)
This project uses a three-stage Retrieval-Augmented Generation (RAG) pipeline to effectively extract answers from large document collections. The system combines **Sparse and Dense Retrieval**, **Reranking**, and **Machine Reading Comprehension (MRC)** to provide accurate answers from Wikipedia. Each stage is managed in a separate branch, and this README provides an overview, directing to each branch for detailed setup and usage instructions.

Wrap-up Reports : [ODQA with RAG](https://github.com/boostcampaitech7/level2-mrc-nlp-09/blob/main/RAG%20based%20Open-Domain%20Question%20Answering.pdf)

## Overview
### 1. Environment Setup
To replicate the environment used for each model, refer to the `environment.yml` files in each branch. Set up the environment by running:  
```console
$ conda env create -f environment.yml
```

**System requirements**:
Ubuntu-20.04.6 LTS

Each branch (`main-Retrieval`, `main-Reranker`, `main-MRC`) specifies the exact Python and PyTorch versions used for its respective model.



### 2. Data
- **Wikipedia Document Corpus**: Used in the Retrieval stage, located at `./data/raw/wikipedia_documents.json`, comprising approximately 57,000 unique documents.

- **MRC Dataset**: Used in the MRC stage, stored at `./data/raw/train_dataset` and `./data/raw/test_dataset`.  

- **Sample Data Structure Overview**:  
![8acf9df5-ea23-4ad2-a24c-c9a847d24e59](https://github.com/user-attachments/assets/c3a69377-34e7-49d7-828c-a93977baa42d)

- **Sample Document Format**:  
![00129d26-34fd-4313-9579-e088665239bf](https://github.com/user-attachments/assets/a812e35d-93e6-42c8-808f-37d683337e73)



### 3. Pipeline Stages

- Retrieval
Branch: `main-Retrieval`

  Filters 100 relevant documents per query from a large Wikipedia corpus.

- Reranker
Branch: `main-Reranker`

  Reranks the top 100 documents per query. The output can be analyzed for performance and is passed to the MRC stage.

- MRC
Branch: `main-MRC`

  Processes the final top-k reranked documents per query, extracting exact answers from the contexts using a machine reading comprehension model.

For further information, refer to each branch’s README for installation, environment specifics, and model usage instructions.  



### 4. Evaluation Metrics
To assess model performance, the following metrics are used:

- Exact Match (EM): Awards a score only when the model’s prediction exactly matches the true answer. Each question is scored as either 0 or 1.  

- F1 Score: Unlike EM, F1 Score gives a partial score by considering word overlap between the prediction and the true answer. For instance, if the correct answer is "Barack Obama" but the prediction is "Obama," the EM score would be 0, while the F1 Score would award a partial score based on overlapping words.



### 5. Results

![image](https://github.com/user-attachments/assets/3235d172-82cc-4938-b1d0-1c996c09a4bb)

EM: 62.92%
F1 Score: 73.46%

*These metrics provide insight into both the accuracy and partial correctness of the model’s predictions across all stages of the pipeline.*
