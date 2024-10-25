from typing import Optional
from dataclasses import dataclass, field


@dataclass
class RetrievalTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dense_train_dataset: Optional[str] = field(
        default="squad_kor_v1", metadata={"help": "The name of the dataset to use."}
    )
    topk: Optional[int] = field(default=3)
    # Whether retrain embeddings
    retrain: Optional[bool] = field(default=False, metadata={"help": "Whether retrain&overwrite embedding files"})

    # Parameters for bm25
    b: Optional[float] = field(
        default=0.01, metadata={"help": "0일 수록 문서 길이의 중요도가 낮아진다. 일반적으로 0.75 사용, 우리 모델에서 최적 0.01로 나옴"}
    )
    k1: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "TF의 saturation을 결정하는 요소. 어떤 토큰이 한 번 더 등장했을 때 이전에 비해 점수를 얼마나 높여주어야 하는가를 결정. (1.2~2.0을 사용하는 것이 일반적)"
        },
    )

    learning_rate: Optional[float] = field(default=3e-5, metadata={"help": "Learning Rate"})
    per_device_train_batch_size: Optional[float] = field(
        default=1, metadata={"help": "bm25 데이터 셋을 사용한다면 batch_size를 1로 사용하셔야 합니다!"}
    )
    per_device_eval_batch_size: Optional[float] = field(default=4, metadata={"help": "학습시에 evaluation을 사용하지는 않습니다..."})
    num_train_epochs: Optional[int] = field(default=2, metadata={"help": "Num Epoch"})
    weight_decay: Optional[float] = field(default=0.01, metadata={"help": "Model Weight Decay"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "덴스 리트리버에서 사용되는 Gradient Accumulation 스텝입니다."}
    )

    # Parameters for hybrid-retriever
    alpha: Optional[float] = field(default=0.1, metadata={"help": "Set weight for sparse retriever"})