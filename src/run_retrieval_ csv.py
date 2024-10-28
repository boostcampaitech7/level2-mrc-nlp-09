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
        alpha = 2
        doc_scores, doc_indices = self.get_relevant_doc_bulk(
            query_or_dataset["question"], topk=max(40 + topk, alpha * topk)
        )

        for idx, example in enumerate(tqdm(query_or_dataset, desc="Retrieval: ")):

            doc_scores_topk = []
            doc_indices_topk = []

            pointer = 0  # 초기 포인터 설정

            while len(doc_indices_topk) < topk and pointer < len(doc_indices[idx]):
                new_text_idx = doc_indices[idx][pointer]
                new_text = self.contexts[new_text_idx]

                is_non_duplicate = True
                for d_id in doc_indices_topk:
                    # fuzz ratio를 사용해 중복 제거
                    if fuzz.ratio(self.contexts[d_id], new_text) > 65:
                        is_non_duplicate = False
                        break

                if is_non_duplicate:
                    doc_scores_topk.append(doc_scores[idx][pointer])
                    doc_indices_topk.append(new_text_idx)

                pointer += 1

                # 충분한 개수를 찾지 못한 경우, alpha 값을 증가시켜야 함
                if pointer >= max(40 + topk, alpha * topk):
                    print(f"Warning: Query {idx}에서 중복 없는 topk 문서를 추출하는데 실패하였습니다. alpha 값을 늘려주세요.")
                    break

            # 최종 topk 개수 보장
            if len(doc_indices_topk) < topk:
                print(f"Query {idx}: 중복을 걸러낸 후 topk 문서가 부족합니다.")

            # 하나의 tmp에 모든 topk context 저장
            tmp = {
                "question_id": example["id"],
                "question": example["question"],
            }

            for doc_id in range(len(doc_indices_topk)):
                doc_idx = doc_indices_topk[doc_id]
                tmp[f"top{doc_id + 1}_context"] = self.contexts[doc_idx]

            total.append(tmp)

        # 결과를 DataFrame으로 저장
        df = pd.DataFrame(total)
        df.to_csv('data/pipeline/BM25Ensemble_top100_original.csv', index=False)
