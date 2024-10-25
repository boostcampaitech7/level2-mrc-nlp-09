from glob import glob

import os.path as p
from itertools import product
from argparse import Namespace
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from utils.tools import update_args
from utils.prepare import get_dataset, get_reader, get_retriever
from utils.evaluation import evaluation


def train_reader(args):
    strategies = args.strategies
    seeds = args.seeds[: args.run_cnt]

    for idx, (seed, strategy) in enumerate(product(seeds, strategies)):
        args = update_args(args, strategy)  # auto add args.save_path, args.base_path
        args.strategy, args.seed = strategy, seed
        args.info = Namespace()
        set_seed(seed)

        checkpoint_dir = glob(p.join(args.path.checkpoint, f"{strategy}*"))
        if not checkpoint_dir:
            raise FileNotFoundError(f"{strategy} 전략에 대한 checkpoint가 존재하지 않습니다.")

        args.model.model_path = get_last_checkpoint(checkpoint_dir[0])
        if args.model.model_path is None:
            raise FileNotFoundError(f"{checkpoint_dir[0]} 경로에 체크포인트가 존재하지 않습니다.")

        # run_name: strategy + alias + seed
        args.train.run_name = "_".join([strategy, args.alias, str(seed)])
        args.train.output_dir = p.join(args.path.checkpoint, args.train.run_name)

        print("checkpoint_dir: ", args.train.output_dir)

        datasets = get_dataset(args, is_train=True)
        reader = get_reader(args, eval_answers=datasets["validation"])
        retriever = get_retriever(args)

        datasets["validation"] = retriever.retrieve(datasets["validation"], topk=args.retriever.topk)["validation"]
        reader.set_dataset(eval_dataset=datasets["validation"])

        trainer = reader.get_trainer()

        if args.train.do_eval:
            metric_results = trainer.evaluate()
            results = evaluation(args)

            metric_results["predictions"]["exact_match"] = results["EM"]["value"]
            metric_results["predictions"]["f1"] = results["F1"]["value"]

            print("EVAL RESULT")
            print(metric_results["predictions"])


if __name__ == "__main__":
    from utils.tools import get_args

    args = get_args()
    train_reader(args)