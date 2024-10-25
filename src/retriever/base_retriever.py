import os
import json

import pandas as pd
from tqdm.auto import tqdm
from fuzzywuzzy import fuzz

from datasets import Dataset
from datasets import Sequence, Value, Features, DatasetDict


class Retrieval:
    def __init__(self, args):
        self.args = args
        self.encoder = None
        self.p_embedding = None

        with open(os.path.join(self.args.data_path, "data", "wikipedia_documents.json"), "r") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))

        # BM25 단독으로 사용하시는 경우, title을 추가해주시면 성능이 더 올라갑니다.
        # self.contexts = list(dict.fromkeys([v["title"] + ": " + v["text"] for v in wiki.values()]))

        self.context_ids = list(dict.fromkeys([v["document_id"] for v in wiki.values()]))

    def _exec_embedding(self):
        raise NotImplementedError

    def get_embedding(self):
        raise NotImplementedError

    def get_relevant_doc_bulk(self, queries, topk):
        """전체 doc scores, doc indices를 반환합니다."""
        raise NotImplementedError

    def retrieve(self, query_or_dataset, topk=1):
        assert self.p_embedding is not None, "get_embedding()을 먼저 수행한 후에 retrieve()를 작동시켜 주세요. "

        total = []
        # 중복을 걸러내기 위해 40 + topk (확인된 최대 중복 개수 40 + topk개)으로 최소값을 설정하고, topk의 alpha 배수로 뽑습니다.
        alpha = 2
        doc_scores, doc_indices = self.get_relevant_doc_bulk(
            query_or_dataset["question"], topk=max(40 + topk, alpha * topk)
        )

        for idx, example in enumerate(tqdm(query_or_dataset, desc="Retrieval: ")):

            doc_scores_topk = [doc_scores[idx][0]]
            doc_indices_topk = [doc_indices[idx][0]]

            pointer = 1

            while len(doc_indices_topk) != topk:
                is_non_duplicate = True
                new_text_idx = doc_indices[idx][pointer]
                new_text = self.contexts[new_text_idx]

                for d_id in doc_indices_topk:
                    if fuzz.ratio(self.contexts[d_id], new_text) > 65:
                        is_non_duplicate = False
                        break

                if is_non_duplicate:
                    if idx < len(doc_scores) and pointer < len(doc_scores[idx]):
                        doc_scores_topk.append(doc_scores[idx][pointer])
                    doc_indices_topk.append(new_text_idx)

                pointer += 1

                if pointer == max(40 + topk, alpha * topk):
                    break

            assert len(doc_indices_topk) == topk, "중복 없는 topk 추출을 위해 alpha 값을 증가시켜 주세요."

            for doc_id in range(topk):
                doc_idx = doc_indices_topk[doc_id]
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context_id": self.context_ids[doc_idx],  # retrieved id
                    "context": self.contexts[doc_idx],  # retrieved document
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]  # original document
                    tmp["answers"] = example["answers"]  # original answer
                total.append(tmp)

        df = pd.DataFrame(total)

        if self.args.train.do_predict is True:
            f = Features(
                {
                    "context": Value(dtype="string"),
                    "id": Value(dtype="string"),
                    "question": Value(dtype="string"),
                    "context_id": Value(dtype="int64"),
                }
            )
        else:
            f = Features(
                {
                    "answers": Sequence(
                        feature={"text": Value(dtype="string"), "answer_start": Value(dtype="int64")},
                        length=-1,
                    ),
                    "context": Value(dtype="string"),
                    "id": Value(dtype="string"),
                    "question": Value(dtype="string"),
                    "original_context": Value(dtype="string"),
                    "context_id": Value(dtype="int64"),
                }
            )

        datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
        return datasets