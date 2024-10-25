import numpy as np
from retriever.sparse import BM25Retrieval

class BM25LRetrieval(BM25Retrieval):
    def __init__(self, args):
        super().__init__(args)

        self.delta = 0.6

    def calculate_score(self, p_embedding, query_vec):
        b, k1, avdl, delta = self.b, self.k1, self.avdl, self.delta
        len_p = self.dls[:p_embedding.shape[0]]  # Slice len_p to match the number of embeddings

        p_emb_for_q = p_embedding[:, query_vec.indices].toarray()

        # Compute ctd, adjusting len_p for broadcast compatibility
        ctd = p_emb_for_q / (1 - b + b * len_p / avdl)[:, None]
        denom = k1 + (ctd + delta)

        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it needs to be converted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.idf[None, query_vec.indices] - 1.0

        # Compute numer using the adjusted idf and ctd
        numer = np.multiply((ctd + delta), np.broadcast_to(idf, p_emb_for_q.shape)) * (k1 + 1)

        result = (numer / denom).sum(1)

        return result
