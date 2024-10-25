import os
import json
import argparse
import os.path as p

from arguments.train_args import TrainArguments
from arguments.model_args import ModelArguments
from arguments.data_args import DataTrainingArguments
from arguments.retriever_args import RetrievalTrainingArguments

from transformers import HfArgumentParser, TrainingArguments


SEEDS = [95, 12, 0, 7, 63, 3, 2, 61, 4, 32, 40, 94, 2033, 2314]


def str2bool(v):
    """
    Transform user input(argument) to be boolean expression.

    :param v: (string) user input
    :return: Bool(True, False)
    """
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def str2list(v):
    """
    Transform user input list(arguments) to be list of string.
    Multiple options needs comma(,) between each options.
    ex)
        --strategies ST01
        --strategies ST01,ST02,ST03

    :param v: (string) user input
    :return: list of option string. (ex - ['ST01', 'ST02', 'ST03')
    """
    if isinstance(v, list):
        return v

    return v.strip().split(",")


def str2intlist(v):
    """
    Transform user input list(arguments) to be list of integer.
    Multiple options needs comma(,) between each options.
    ex)
        --seeds 42
        --seeds 42, 84, 126

    :param v: (string) user input
    :return: list of option interger. (ex - [42, 84, 126])
    """
    if isinstance(v, list):
        return v

    return list(map(int, v.strip().split(",")))


def update_args(args, strategy):
    """
    Setup args with strategy setting file(ex-ST01.json) in config.

    :param args: args to setup with.
    :param strategy: strategy file name in config directory(input/config/).
    :return: updated args.
    """
    json_path = os.path.join(args.path.config, f"{strategy}.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path}, JSON 파일이 보이지 않습니다.")

    with open(json_path, "r") as f:
        temp = json.load(f)

    args.alias = temp["alias"]
    for arg_type in ["model", "data", "train", "retriever"]:
        temp_type = getattr(args, arg_type)
        for k, v in temp[arg_type].items():
            setattr(temp_type, k, v)

    return args


def get_args(is_notebook=False):
    """
    Parse arguments and handle exceptions.

    :return: processed arg(arguments)
    """
    arg_parser = argparse.ArgumentParser(description="mrc-stage-openqa-vumblebot")

    arg_parser.add_argument("--strategies", type=str2list)
    arg_parser.add_argument("--run_cnt", type=int, default=1)
    arg_parser.add_argument("--seeds", type=str2intlist, default=SEEDS)
    arg_parser.add_argument("--data_path", type=str, default="input/")
    arg_parser.add_argument("--debug", type=str2bool, default=False)
    arg_parser.add_argument("--report", type=str2bool, default=False)

    # use for predict
    arg_parser.add_argument("--model_path", type=str, default="")

    # data_path + 'info', 시각화를 위한 정보 저장
    # data_path + 'checkpoint', 모델 가중치 저장
    # data_path + 'config', 모델 하이퍼파라미터
    # data_path + 'embed', 임베딩 데이터
    # data_path + 'train_data', MRC 데이터

    if is_notebook:
        args = arg_parser.parse_args([])
    else:
        args = arg_parser.parse_args()

    args.path = argparse.Namespace()
    args.path.info = p.join(args.data_path, "info")
    args.path.embed = p.join(args.data_path, "embed")
    #  args.path.config = p.join(args.data_path, "config")
    args.path.config = "config"
    args.path.checkpoint = p.join(args.data_path, "checkpoint")

    if not p.exists(args.data_path):
        os.mkdir(args.data_path)

    for k in ["info", "embed", "config", "checkpoint"]:
        path = getattr(args.path, k)
        if not p.exists(path):
            os.mkdir(path)

    args.path.train_data_dir = p.join(args.data_path, "data")

    if not p.exists(args.path.train_data_dir):
        os.mkdir(args.path.train_data_dir)

    if not os.path.exists(args.path.train_data_dir):
        raise FileNotFoundError(
            f"{p.abspath(args.path.train_data_dir)} \
위치가 보이지 않습니다. args.path값을 절대 경로로 수정하거나 \
input과 같은 폴더에 위치해주세요."
        )

    if args.run_cnt > len(SEEDS):
        raise ValueError("SEEDS를 직접 입력하거나 SEEDS Default 값을 늘려주세요. ")

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainArguments, RetrievalTrainingArguments))
    model_args, data_args, train_args, retriever_args = parser.parse_args_into_dataclasses(args=[])

    # train_args : TrainingArgument Datasetdict인데, pororo_prediction key가 없어서 update_args에서 오류가 뜰 수 있어 key값 미리 initialize
    # training_args : pororo_prediction와 output_dir를 추가반영하여 만든 custom argument(config의 train argument에 가까운)
    training_args = TrainingArguments(output_dir=args.path.checkpoint)

    args.data = data_args
    args.model = model_args
    args.retriever = retriever_args

    args.train = training_args
    args.train.eval_step = train_args.eval_step
    args.train.do_ensemble = train_args.do_ensemble
    args.train.masking_ratio = train_args.masking_ratio
    args.train.freeze_backbone = train_args.freeze_backbone
    args.train.do_eval_during_training = train_args.do_eval_during_training

    return args


def run_test(tcls):
    import unittest

    """
    Runs unit tests from a test class
    :param tcls: A class, derived from unittest.TestCase
    """
    suite = unittest.TestLoader().loadTestsFromTestCase(tcls)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)