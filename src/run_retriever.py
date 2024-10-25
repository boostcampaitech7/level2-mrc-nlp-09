import random
from itertools import product
from collections import defaultdict

import numpy as np
from transformers import set_seed

from fuzzywuzzy import fuzz
from utils.tools import get_args, update_args
from utils.prepare import get_dataset, get_retriever



def train_retriever(args):
    strategies = args.strategies
    seeds = args.seeds[: args.run_cnt]

    topk_result = defaultdict(list)

    for idx, (seed, strategy) in enumerate(product(seeds, strategies)):
        args = update_args(args, strategy)
        set_seed(seed)

        datasets = get_dataset(args, is_train=True)

        retriever = get_retriever(args)
        valid_datasets = retriever.retrieve(datasets["validation"], topk=args.retriever.topk)

        print(f"전략: {strategy} RETRIEVER: {args.model.retriever_name}")
        legend_name = "_".join([strategy, args.model.retriever_name])
        topk = args.retriever.topk

        cur_cnt, tot_cnt = 0, len(datasets["validation"])

        indexes = np.array(range(tot_cnt * topk))
        print("total_cnt:", tot_cnt)
        print("valid_datasets:", valid_datasets)

        qc_dict = defaultdict(bool)
        for idx, fancy_index in enumerate(zip([indexes[i::topk] for i in range(topk)])):
            topk_dataset = valid_datasets["validation"][fancy_index[0]]

            for question, real, pred in zip(
                topk_dataset["question"], topk_dataset["original_context"], topk_dataset["context"]
            ):
                # if two texts overlaps more than 65%,
                if fuzz.ratio(real, pred) > 85 and not qc_dict[question]:
                    qc_dict[question] = True
                    cur_cnt += 1

            topk_acc = cur_cnt / tot_cnt
            topk_result[legend_name].append(topk_acc)
            print(f"TOPK: {idx + 1} ACC: {topk_acc * 100:.2f}")


if __name__ == "__main__":
    args = get_args()
    train_retriever(args)
