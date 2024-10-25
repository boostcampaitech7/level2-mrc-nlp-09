## Rerank top-k documents from Sparse Retriever

### Setup Environment
For our experiments we have added environment.yaml for creating the same environment that we have used. For setup of enviorment please run the following command:

```console
$ conda env create -f environment.yml
```

Please refer to these environments:
NVIDIA-SMI 535.161.08
Ubuntu-20.04.6 LTS
python=3.11.10
torch=2.4.1+cu121

### Directory Structure
./  
├── data/  
│   ├── pipeline/  
│   │   ├── BM25Ensemble_top100_original.csv        # input file  
│   │   └── reranker_final.csv                      # output file  
│   ├── notebooks/  
│   │   ├── dense_pipeline_result.ipynb             # result analysis  
│   ├── src/  
│   │   ├── retrieval_test_bge-reranker_pipeline.py # code  
├── environment.yml  
└── README.md  

### Data
We have processed the data on BM25Ensemble_top100_original.csv. This file is output of Sparse Retrieval, which filters 100 wikipedia documents from 60613 wikipedia documents based on queries.

Data Format: .csv
    Rows: Queries
    Cols: ['id', 'question', 'top1_context', 'top2_context', ..., 'top100_context']

Data Shape: (600, 102)

The path to the Data are:
"data/pipeline"



 ###### START ######

### 0. Settings
Before start, Make sure setting conda environment and check if data from Sparse Retrieval are in "data/pipeline".


### 1. Rerank
To start running Reranker model, please run the command below:

```console
$ bash src/test.sh
```


### 2. Reranker Process
After running process, Output file "reranker_final.csv" will be created. This file is output of Reranking, which cuts wikipedia documents from Sparse Retrieval from 100 to 45(Recall Accuracy: 97.5%) and rerank the order of 45 documents based on similarities.

Data Format: .csv
    Rows: Queries
    Cols: ['id', 'question', 'top1_context', 'top2_context', ..., 'top45_context']

Data Shape: (600, 47)

The path to the Data are:
"data/pipeline"


### 3. Result Analysis
With "reranker_final.csv", you can analyze it by Recall Score.


### 4. Output to MRC
You should give output file to MRC model to process MRC task.


