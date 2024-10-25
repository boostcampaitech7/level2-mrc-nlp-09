from retriever.dense import DprBert
from retriever.hybrid import HybridRetrieval, HybridLogisticRetrieval
from retriever.sparse import BM25EnsembleRetrieval


class BM25EnsembleDprBert(HybridRetrieval):
    def __init__(self, args):
        super().__init__(args)
        temp = args.model.retriever_name

        args.model.retriever_name = "BM25Ensemble"
        self.sparse_retriever = BM25EnsembleRetrieval(args)
        args.model.retriever_name = "DPRBERT"
        self.dense_retriever = DprBert(args)

        args.model.retriever_name = temp


class LogisticBM25EnsembleDprBert(HybridLogisticRetrieval):
    def __init__(self, args):
        super().__init__(args)
        temp = args.model.retriever_name

        args.model.retriever_name = "BM25Ensemble"
        self.sparse_retriever = BM25EnsembleRetrieval(args)
        args.model.retriever_name = "DPRBERT"
        self.dense_retriever = DprBert(args)

        args.model.retriever_name = temp