import numpy as np

from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

from retriever.sparse import SparseRetrieval


class TfidfRetrieval(SparseRetrieval):
    def __init__(self, args):
        super().__init__(args)

        print("Using AutoTokenizer: ", args.model.tokenizer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model.tokenizer_name, use_fast=True).tokenize

        self.encoder = TfidfVectorizer(tokenizer=self.tokenizer, ngram_range=(1, 2))
        self.p_embedding = None

    def _exec_embedding(self):
        p_embedding = self.encoder.fit_transform(self.contexts)
        return p_embedding, self.encoder

    def get_relevant_doc_bulk(self, queries, topk):
        query_vec = self.encoder.transform(queries)
        assert np.sum(query_vec) != 0, "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = query_vec * self.p_embedding.T

        if not isinstance(result, np.ndarray):
            result = result.toarray()

        doc_scores, doc_indices = [], []

        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:topk])
            doc_indices.append(sorted_result.tolist()[:topk])

        return doc_scores, doc_indices