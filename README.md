## Reranker Experiments

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


### Model Candidate Selection

1. github → MTEB Korean Leaderboard
    
    https://github.com/su-park/mteb_ko_leaderboard?tab=readme-ov-file
    
2. Multilingual E5 Finetune (2024.10.15)
    
    https://yjoonjang.medium.com/koe5-최초의-한국어-임베딩-모델-multilingual-e5-finetune-22fa7e56d220
    
3. Autorag → Leaderboard
    
    https://velog.io/@autorag/어떤-한국어-임베딩-모델-성능이-가장-좋을까-직접-벤치마크-해보자
    
    https://github.com/Marker-Inc-Korea/AutoRAG-example-korean-embedding-benchmark

4. (Reference) SentenceTransformers Documents

   https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html


### Rerank Experiment

- Input: [240(query) x Top K] Wiki Docs .csv file
- Output: [240(query) x Top K] Wiki Docs .csv file (Similarity rank sorting results for top K)

| Model / Top K                                      | Top 1               | Top 2               | Top 3               | Top 5               | Top 10              | Top 15              | Top 20              |
|----------------------------------------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|
| dragonkue/bge-reranker-v2-m3-ko                    | 91.25% (219/240)    | 95.42% (229/240)    | 95.83% (230/240)    | 96.67% (232/240)    | 97.08% (233/240)    | 97.50% (234/240)    | 97.50% (234/240)    |
| Sparse only                                        | 71.67% (172/240)    | 82.08% (197/240)    | 85.42% (205/240)    | 86.67% (208/240)    | 92.08% (221/240)    | 94.58% (227/240)    | 95.00% (228/240)    |
| dragonkue/BGE-m3-ko                                | 66.67% (160/240)    | 79.17% (190/240)    | 83.33% (200/240)    | 87.50% (210/240)    | 94.58% (227/240)    | 95.83% (230/240)    | 96.25% (231/240)    |
| Ensemble (Sparse only, BGE-m3-ko, KoE5 (max))      | 73.75% (177/240)    | 82.50% (198/240)    | 84.58% (203/240)    | 85.83% (206/240)    | 90.00% (216/240)    | 92.50% (222/240)    | 92.92% (223/240)    |
| Ensemble (Sparse only, BGE-m3-ko, KoE5 (mean))     | 72.50% (174/240)    | 82.50% (198/240)    | 85.42% (205/240)    | 86.67% (208/240)    | 89.58% (215/240)    | 91.25% (219/240)    | 91.67% (220/240)    |
| nlpai-lab/KoE5 (max)                               | 60.42% (145/240)    | 76.25% (183/240)    | 79.17% (190/240)    | 85.42% (205/240)    | 92.92% (223/240)    | 94.58% (227/240)    | 95.00% (228/240)    |
| nlpai-lab/KoE5 (mean)                              | 52.08% (125/240)    | 67.50% (162/240)    | 74.58% (179/240)    | 81.67% (196/240)    | 89.17% (214/240)    | 90.83% (218/240)    | 92.92% (223/240)    |
| Alibaba-NLP/gte-Qwen2-1.5B-instruct                | 57.08% (137/240)    | 67.08% (161/240)    | 73.75% (177/240)    | 80.83% (194/240)    | 85.42% (205/240)    | 88.33% (212/240)    | 90.83% (218/240)    |
| upskyy/bge-m3-korean                               | 46.67% (112/240)    | 60.00% (144/240)    | 67.92% (163/240)    | 74.17% (178/240)    | 82.92% (199/240)    | 87.50% (210/240)    | 90.42% (217/240)    |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 32.08%  (77/240)   | 41.25%   (99/240)   | 45.00% (108/240)    | 50.83% (122/240)    | 59.58% (143/240)    | 65.42% (157/240)    | 70.83% (170/240)    |
| gte-multilingual-base                              | 55.00% (132/240)    | 67.92% (163/240)    | 75.00% (180/240)    | 81.67% (196/240)    | 86.67% (208/240)    | 88.33% (212/240)    | 90.42% (217/240)    |


-> 'dragonkue/bge-reranker-v2-m3-ko' model is the BEST for reranking process.


### Directory Structure
./  
├── code/  
├── data/  
│   ├── experiments/  
│   │   ├── (experiments data)  
│   ├── pipeline/  
│   │   ├── (experiments data)  
│   ├── preprocessed/  
│   │   ├── (preprocessed data)  
│   ├── raw/  
│   │   ├── (raw data)  
│   ├── notebooks/  
│   │   ├── (experiments codes)  
│   ├── scripts/  
│   │   ├── test.sh  
│   ├── src/  
│   │   ├── data_processing/  
│   │   │   ├── preprocess.py  
│   │   ├── (experiments codes)  
├── .gitignore  
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


