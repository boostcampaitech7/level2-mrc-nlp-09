## A reader that trains on the training dataset and performs evaluation and inference based on a CSV file of retrieval results

### Setup Environment

For our experiments we have added conda-requirements.txt for creating the same environment that we have used. For setup of enviorment please run the following command:

```console
$ while read requirement; do conda install --yes $requirement || pip install $requirement; done < conda-requirements.txt
```

Please refer to these environments:
Ubuntu-20.04.6 LTS
python=3.9.20

### Directory Structure

./  
|-- data/
| |-- raw/
| | |-- wikipedia.json
| | |-- train_dataset
| | | |-- train
| | | |-- validation
| |-- external/
| |-- preprocessed/
| | |-- retrievalResults.csv
|-- models/
|-- notebooks/
| |-- EDA.ipynb
|-- outputs/
|-- src/
| |-- nbest_files/
| | |-- nbest_prediction1.json
| |-- arguments.py
| |-- ensemble.py
| |-- inference_csv.py
| |-- train_csv.py
| |-- train_wandb.py
| |-- trainer_qa.py
| |-- utils_qa.py
|-- eval.sh
|-- inference.sh
|-- train.sh
|-- .gitignore
|-- README.md
|-- conda-requirements.txt

### Data

The structure of the CSV file consists of id, question, top1_context, top2_context, and so on. The answer is retrieved from the training or validation dataset using the id value.
The id matches those in the provided validation and training datasets.

###### START

### 0. Settings

Before start, Make sure setting conda environment and check if the CSV file with retrieval results is in the "data/preprocessed" directory.

### 1. Train

Refer to arguments.py and execute the following code:

```console
$ source train.sh
```

### 2. Evaluate

Modify the paths in the main function of train_csv.py to match your environment, and then execute the following code:

```console
$ source eval.sh
```

The output is located in output/train_dataset/validation.

### 3. Inference

Modify the paths in the main function of train_csv.py to match your environment, and then execute the following code:

```console
$ source eval.sh
```

The output is located in output/test_dataset/validation.

### 4. Ensemble

To prepare for ensembling, place the output result file nbest_prediction.json in the src/nbest_files directory, and then enter the following code:

```console
$ python src/ensemble.py
```
