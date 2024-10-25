import numpy as np
from retriever.sparse import BM25Retrieval

class BM25PlusRetrieval(BM25Retrieval):
    def __init__(self, args):
        super().__init__(args)
        self.delta = 0.7

    def calculate_idf(self):
        idf = self.idf_encoder.idf_
        idf = idf - np.log(len(self.contexts)) + np.log(len(self.contexts) + 1.0)
        return idf

    def calculate_score(self, p_embedding, query_vec):
        b, k1, avdl, delta = self.b, self.k1, self.avdl, self.delta
        len_p = self.dls[:p_embedding.shape[0]]  # Adjust len_p to match the number of embeddings

        p_emb_for_q = p_embedding[:, query_vec.indices].toarray()

        # Adjusting for passage lengths in the denominator
        denom = p_emb_for_q + (k1 * (1 - b + b * len_p / avdl))[:, None]

        # Adjust the idf calculation
        idf = self.idf[None, query_vec.indices] - 1.0
        idf_broadcasted = np.broadcast_to(idf, p_emb_for_q.shape)

        # Numerator calculation
        numer = p_emb_for_q * (k1 + 1)

        # Final score computation with delta and idf
        result = (np.multiply((numer / denom) + delta, idf_broadcasted)).sum(1)

        return result
