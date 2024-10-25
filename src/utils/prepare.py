import os.path as p

from transformers import AutoConfig, AutoModel, AutoTokenizer
from datasets import load_from_disk, load_dataset, concatenate_datasets, Dataset

from reader import CustomHeadReader
from retriever.hybrid import BM25EnsembleDprBert, LogisticBM25EnsembleDprBert
from retriever.sparse import (
    TfidfRetrieval,
    BM25LRetrieval,
    BM25PlusRetrieval,
    ATIREBM25Retrieval,
    BM25EnsembleRetrieval,
)
from retriever.dense import DprBert, BaseTrainMixin, Bm25TrainMixin, DprElectra


RETRIEVER = {
    # Sparse
    "TFIDF": TfidfRetrieval,
    "BM25L": BM25LRetrieval,
    "BM25Plus": BM25PlusRetrieval,
    "ATIREBM25": ATIREBM25Retrieval,
    "BM25Ensemble": BM25EnsembleRetrieval,
    # Dense
    "DPRBERT": DprBert,
    "DPRELECTRA": DprElectra,
    # Hybrid
    "BM25Ensemble_DPRBERT": BM25EnsembleDprBert,
    "LOG_BM25Ensemble_DPRBERT": LogisticBM25EnsembleDprBert,
}


def retriever_mixin_factory(name, base, mixin):
    """ mixin class의 method를 overwriting."""
    return base.__class__(name, (mixin, base), {})


def get_retriever(args):
    """
    Get appropriate retriever.

    AVAILABLE OPTIONS(2021.05.05)
    - Term-based
        - TF-IDF : use konlpy-Mecab for word tokenization.
        - BM-25
    - Vector Embedding
        - Sparse
        - Dense
    Need more retriever and retriever options.

    :param args
        - model.retriever_name : [TFIDF, DPR, BM25]
    :return: Retriever which contains embedded vector(+indexer if faiss is built).
    """

    retriever_class = RETRIEVER[args.model.retriever_name]

    # Dataset에 따라서 학습 방법이 달라진다. # retriever/dense/dense_train_mixin.py
    if args.retriever.dense_train_dataset.startswith("bm25"):
        retriever_class = retriever_mixin_factory("bm25_mixin_class", retriever_class, Bm25TrainMixin)
    elif args.retriever.dense_train_dataset == "train_dataset" and args.model.retriever_name != "COLBERT":
        retriever_class = retriever_mixin_factory("base_mixin_class", retriever_class, BaseTrainMixin)

    retriever = retriever_class(args)
    retriever.get_embedding()

    return retriever


def get_reader(args, eval_answers):
    """
    Get pretrained MRC-Reader model and tokenizer.
    If model setting is KoBERT, then load tokenizer from public KoBERT Tokenizer.
    Else, transformers library autosets appropriate tokenizer from model(when tokenizer is not specified).

    :param args:
        - model.model_name_or_path(required) : model repository name that registered in huggingface library.
            (ex - 'monologg/koelectra-small-v3-discriminator')
        - model.model_path : saved checkpoint in server disk.
            (ex - '/input/checkpoint/ST01_0_temp/checkpoint-500')
        - model.config_name
        - model_tokenizer_name
    :return: pretrained model and tokenizer.
    """
    config = AutoConfig.from_pretrained(
        args.model.config_name if args.model.config_name else args.model.model_name_or_path
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model.model_name_or_path, use_fast=True)

    if "bert" in args.model.model_name_or_path:
        model = AutoModel.from_pretrained(
            args.model.model_name_or_path,
            from_tf=bool(".ckpt" in args.model.model_name_or_path),
            config=config,
            add_pooling_layer=False,
        )
    elif "electra" in args.model.model_name_or_path:
        model = AutoModel.from_pretrained(
            args.model.model_name_or_path, from_tf=bool(".ckpt" in args.model.model_name_or_path), config=config
        )
    else:
        raise ValueError("BERT/ELECTRA 외 모델에 대한 Custom head model 미구현")

    reader = CustomHeadReader(args=args, model=model, tokenizer=tokenizer, eval_answers=eval_answers)
    return reader


def get_dataset(args, is_train=True):
    """
    Load dataset from dataset path in disk.

    :param args
        - data.dataset_name : [train_dataset, test_dataset, squad_kor_v1]
        - debug : True expressions. If this setting is true, epoch and dataset will be restricted for quick testing.
    :param is_train: True for training, False for validation.
    :return: Loaded dataset.
    """
    datasets = None

    if args.data.dataset_name == "train_dataset":
        if is_train:
            datasets = load_from_disk(p.join(args.path.train_data_dir, args.data.dataset_name))
        else:
            datasets = load_from_disk(p.join(args.path.train_data_dir, "test_dataset"))
    elif args.data.dataset_name == "squad_kor_v1":
        datasets = load_dataset(args.data.dataset_name)
        datasets = datasets.map(lambda examples: {"document_id": examples["id"]})

    # Add more dataset option here.

    if datasets is None:
        raise KeyError(f"{args.data.dataset_name}데이터는 존재하지 않습니다.")

    if args.data.sub_datasets != "" and is_train:
        datasets["train"] = concatenate_datasets_with_ratio(args, datasets["train"])

    if args.debug:
        args.train.num_train_epochs = 1.0
        datasets["train"] = datasets["train"].select(range(100))

    if is_train:
        print(f"TRAIN DATASET 길이: {len(datasets['train'])}")
    print(f"VALID DATASET 길이: {len(datasets['validation'])}")

    return datasets


def concatenate_datasets_with_ratio(args, train_dataset):
    concatenate_list = []

    for sub_dataset_name, ratio in zip(args.data.sub_datasets.split(","), args.data.sub_datasets_ratio.split(",")):
        ratio = float(ratio)
        sub_dataset_path = p.join(args.path.train_data_dir, sub_dataset_name)
        assert p.exists(sub_dataset_path), f"{sub_dataset_name}이 존재하지 않습니다."

        sub_dataset = load_from_disk(sub_dataset_path)
        sub_dataset_len = int(len(sub_dataset["train"]) * ratio)

        print(f"ADD SUB DATASET {sub_dataset_name}, LENGTH: {sub_dataset_len}")

        # sub dataset must have same features: ['id', 'title', 'context', 'question', 'answers']
        features = sub_dataset["train"].features

        new_sub_dataset = sub_dataset["train"].select(range(sub_dataset_len))
        new_sub_dataset = Dataset.from_pandas(new_sub_dataset.to_pandas(), features=features)

        concatenate_list.append(new_sub_dataset.flatten_indices())

    train_dataset = Dataset.from_pandas(train_dataset.to_pandas(), features=features)
    train_dataset = concatenate_datasets([train_dataset.flatten_indices()] + concatenate_list)

    return train_dataset