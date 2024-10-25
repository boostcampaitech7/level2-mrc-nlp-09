import os.path as p
from itertools import product
from argparse import Namespace
from transformers import set_seed

from utils.tools import update_args
from utils.evaluation import evaluation
from utils.prepare import get_dataset, get_reader


def train_reader(args):
    strategies = args.strategies
    seeds = args.seeds[: args.run_cnt]

    for idx, (seed, strategy) in enumerate(product(seeds, strategies)):
        args = update_args(args, strategy)  # auto add args.save_path, args.base_path
        args.strategy, args.seed = strategy, seed
        args.info = Namespace()
        set_seed(seed)

        # below codes must run before 'reader.get_trainer()'
        # run_name: strategy + alias + seed
        args.train.run_name = "_".join([strategy, args.alias, str(seed)])
        args.train.output_dir = p.join(args.path.checkpoint, args.train.run_name)

        print("checkpoint_dir: ", args.train.output_dir)

        # retrieve 과정이 없어 top-k를 반환할 수 없음. 무조건 top-1만 반환
        # run_mrc.py DOES NOT execute retrieve, so args.retriever.topk cannot be n(>1).
        # If topk > 1, post processing function returns mis-bundled predictions.
        args.retriever.topk = 1

        datasets = get_dataset(args, is_train=True)
        reader = get_reader(args, eval_answers=datasets["validation"])

        reader.set_dataset(train_dataset=datasets["train"], eval_dataset=datasets["validation"])

        trainer = reader.get_trainer()

        if args.train.do_train:
            train_results = trainer.train()
            print(train_results)

        metric_results = None

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