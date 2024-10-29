## Retrieval Experiments


### Setup Environment
For our experiments we have added `environment.yaml` for creating the same environment that we have used. For setup of enviorment please run the following command:

```console
$ conda env create -f environment.yml
```

Please refer to these environments:
- torch==1.13
- datasets==2.15.0
- transformers==4.25.1
- tqdm
- pandas
- scikit-learn
- fuzzywuzzy>=0.7.0,<=0.18.0
- numpy<2.0


### Data setting

Please place the dataset provided by the competition in the `input/data` directory:

```
input/
│
└── data
    ├── train_dataset
    │   ├── dataset_dict.json
    │   ├── train
    │   │   ├── dataset.arrow
    │   │   ├── dataset_info.json
    │   │   └── state.json
    │   └── validation
    │       ├── dataset.arrow
    │       ├── dataset_info.json
    │       └── state.json
    ├── test_dataset
    │   ├── dataset_dict.json
    │   └── validation
    │       ├── dataset.arrow
    │       ├── dataset_info.json
    │       └── state.json
    └── wikipedia_documents.json
```


### Directory Structure

```
exp-Retrieval/
│
├── config/
│   └── ST01.json
├── data/
│   └── pipeline/
├── input/
│   ├── data/
│   ├── embed/
│   └── checkpoint/
├── notebooks/
|   └── EDA.ipynb
├── scripts/
|   ├── run_retrieval.sh
|   ├── run_reader.sh
|   ├── run.sh
|   └── predict.sh
├── src/
|   ├── data_processing/
|   ├── arguments/
|   ├── retriever/
|   |   ├── sparse/
|   |   ├── dense/
|   |   └── hybrid/
|   ├── reader/
|   ├── utils/
|   ├── run_retriever.py
|   ├── run_retriever_csv.py
|   ├── run_reader.py
|   ├── run.py
|   ├── predict.py
|   ├── postprocessing.py
|   └── ensemble.py
├── .gitignore
├── README.md
├── requirements.txt
└── environment.yml
```



###### START ######

### 0. Settings
Before start, Make sure setting conda environment and check the dataset are in `input/data`.


### 1. Retrieval
To start running Retrieval model, please run the command below:

```console
$ bash scripts/run_retrieval.sh
```


### 2. Retrieval Process
After running process, Output file `BM25Ensemble_topk_100_original.csv` will be created. This file is output of Retrieval, which cuts wikipedia documents from 60,613 to 100(Recall Accuracy: 97.92%) and ranks these 100 documents based on similarities.

Data Format: .csv
    Rows: Queries
    Cols: `['id', 'question', 'top1_context', 'top2_context', ..., 'top100_context']`

Data Shape: (600, 102)

The path to the Data are: `data/pipeline`


### 3. Result Analysis
With `BM25Ensemble_topk_100_original.csv`, you can analyze it by Recall Score.


### 4. Output to Rerank
You should give output file to Rerank model to process Rerank task.

