import numpy as np
from retriever.sparse import BM25Retrieval

class ATIREBM25Retrieval(BM25Retrieval):
    def __init__(self, args):
        super().__init__(args)

    def calculate_score(self, p_embedding, query_vec):
        b, k1, avdl = self.b, self.k1, self.avdl
        len_p = self.dls[:p_embedding.shape[0]]  # Ensure len_p matches the number of embeddings

        p_emb_for_q = p_embedding[:, query_vec.indices].toarray()
        denom = p_emb_for_q + (k1 * (1 - b + b * len_p / avdl))[:, None]

        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it needs to be converted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.idf[None, query_vec.indices] - 1.0
        idf_broadcasted = np.broadcast_to(idf, p_emb_for_q.shape)

        numer = p_emb_for_q * (idf_broadcasted * (k1 + 1))

        result = (numer / denom).sum(1)

        return result
