## Rerank top-k documents from Retriever

### 1. Setup Environment
To ensure consistency, we've provided `environment.yml` to create the same environment we used in our experiments. Set up the environment by running:
```console
$ conda env create -f environment.yml
```

**Environment Requirements:**    
- Ubuntu 20.04.6 LTS    
- python=3.11.10    
- torch=2.4.1+cu121    


### 2. Directory Structure
./  
├── data/  
│   ├── pipeline/  
│   │   ├── BM25Ensemble_top100_original.csv  
│   │   └── reranker_final.csv  
│   ├── notebooks/  
│   │   └── dense_pipeline_result.ipynb  
│   ├── scripts/  
│   │   └── test.sh  
│   ├── src/  
│   │   └── retrieval_test_bge-reranker_pipeline.py  
├── .gitignore  
├── environment.yml  
└── README.md  


### 3. Data
The `BM25Ensemble_top100_original.csv` file contains data processed by Sparse Retrieval, which filters 100 Wikipedia documents from a set of 60,613 based on queries.  

- Format: .csv
- Rows: Queries
- Columns: `['id', 'question', 'top1_context', 'top2_context', ..., 'top100_context']`
- Shape: (600, 102)  
- Data Path: `data/pipeline/`

<br>

## Steps

### 0. Settings
Before starting, ensure the Conda environment is set up and that Sparse Retrieval data is available in `data/pipeline`.

### 1. Rerank
Run the following command to start the Reranker model:

```console
$ bash src/test.sh
```


### 2. Reranker Process
Once the process completes, the output file `reranker_final.csv` will be created. This file contains the top 45 reranked Wikipedia documents (out of the original 100), reordered based on similarity (Recall Accuracy: 97.5%).

- Format: .csv
- Rows: Queries
- Columns: `['id', 'question', 'top1_context', 'top2_context', ..., 'top45_context']`
- Shape: (600, 47)
- Data Path: `data/pipeline`


### 3. Result Analysis
You can analyze `reranker_final.csv` by calculating the Recall Score.



### 4. Output to MRC
Pass the output file to the MRC model for further processing.



