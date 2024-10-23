from __future__ import annotations
import json
import os
import tiktoken
import numpy as np
#from polyglot.text import Text

from transformers import AutoTokenizer
from langchain.storage import LocalFileStore
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
#from langchain.document_loaders import BaseLoader
from langchain_core.pydantic_v1 import Field
from langchain.embeddings import CacheBackedEmbeddings
from langchain_upstage import UpstageEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from operator import itemgetter
from typing import List, Optional, Tuple, Any, Callable, Dict, Iterable


try:
    from kiwipiepy import Kiwi
except ImportError:
    raise ImportError(
        "Could not import kiwipiepy, please install with `pip install " "kiwipiepy`."
    )


class EmbeddingHelper:
    def __init__(self, path="./cache/"):
        self.path = path
        os.makedirs(self.path, exist_ok=True)
        self.store = LocalFileStore(self.path)

    def cache_embedding(self, embedding):
        embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=embedding,
            document_embedding_cache=self.store,
            namespace=embedding,
            #namespace=embedding.model,
        )
        return embeddings

    @staticmethod
    def count_tokens(docs: List[Document], model_name: str) -> Tuple[List[int], int]:

        tokenizer = tiktoken.encoding_for_model(model_name)

        # Calculate the token count for each document's content
        tokens_per_text = [len(tokenizer.encode(doc.page_content)) for doc in docs]

        # Calculate the total number of tokens
        total_tokens = sum(tokens_per_text)

        return tokens_per_text, total_tokens

    def show_metadata(self, docs: Optional[List]) -> None:
        if docs:
            print("[metadata]")
            print(list(docs[0].metadata.keys()))
            print("\n[examples]")
            max_key_length = max(len(k) for k in docs[0].metadata.keys())
            for k, v in docs[0].metadata.items():
                print(f"{k:<{max_key_length}} : {v}")
        else:
            print("No documents available.")


kiwi_tokenizer = Kiwi()


def kiwi_preprocessing_func(text: str) -> List[str]:
    return [token.form for token in kiwi_tokenizer.tokenize(text)]

def default_preprocessing_func(text: str) -> List[str]:
    return text.split()

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def bert_preprocessing_func(text: str) -> List[str]:
    return [token for token in bert_tokenizer.tokenize(text)]

# def bert_preprocessing_func(text: str) -> List[str]:
#     return [token for token in bert_tokenizer.encode(text, add_special_tokens=True)]
    



# def polyglot_tokenzier(text: str):
#     polyglot_text =Text(text)
#     return polyglot_text.words

# def poly_preprocessing_func(text: str) -> List[str]:
#     return [token.form for token in polyglot_tokenzier(text)]


class KiwiBM25Retriever(BaseRetriever):
    """`BM25` retriever without Elasticsearch."""

    vectorizer: Any
    """ BM25 vectorizer."""
    docs: List[Document] = Field(repr=False)
    """ List of documents."""
    k: int = 10
    """ Number of documents to return."""
    preprocess_func: Callable[[str], List[str]] = bert_preprocessing_func
    """ Preprocessing function to use on the text before BM25 vectorization."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        metadatas: Optional[Iterable[dict]] = None,
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]] = bert_preprocessing_func,
        **kwargs: Any,
    ) -> KiwiBM25Retriever:
        """
        Create a KiwiBM25Retriever from a list of texts.
        Args:
            texts: A list of texts to vectorize.
            metadatas: A list of metadata dicts to associate with each text.
            bm25_params: Parameters to pass to the BM25 vectorizer.
            preprocess_func: A function to preprocess each text before vectorization.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A KiwiBM25Retriever instance.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "Could not import rank_bm25, please install with `pip install "
                "rank_bm25`."
            )

        texts_processed = [preprocess_func(t) for t in texts]
        bm25_params = bm25_params or {}
        vectorizer = BM25Okapi(texts_processed, **bm25_params)
        metadatas = metadatas or ({} for _ in texts)
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        return cls(
            vectorizer=vectorizer, docs=docs, preprocess_func=preprocess_func, **kwargs
        )

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        *,
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]] = bert_preprocessing_func,
        **kwargs: Any,
    ) -> KiwiBM25Retriever:
        """
        Create a KiwiBM25Retriever from a list of Documents.
        Args:
            documents: A list of Documents to vectorize.
            bm25_params: Parameters to pass to the BM25 vectorizer.
            preprocess_func: A function to preprocess each text before vectorization.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A KiwiBM25Retriever instance.
        """
        texts, metadatas = zip(*((d.page_content, d.metadata) for d in documents))
        return cls.from_texts(
            texts=texts,
            bm25_params=bm25_params,
            metadatas=metadatas,
            preprocess_func=preprocess_func,
            **kwargs,
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        processed_query = self.preprocess_func(query)
        return_docs = self.vectorizer.get_top_n(processed_query, self.docs, n=self.k)
        return return_docs

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    @staticmethod
    def argsort(seq, reverse):
        # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
        return sorted(range(len(seq)), key=seq.__getitem__, reverse=reverse)

    def search_with_score(self, query: str, top_k=None):
        normalized_score = KiwiBM25Retriever.softmax(
            self.vectorizer.get_scores(self.preprocess_func(query))
        )

        if top_k is None:
            top_k = self.k

        score_indexes = KiwiBM25Retriever.argsort(normalized_score, True)

        docs_with_scores = []
        for i, doc in enumerate(self.docs):
            document = Document(
                page_content=doc.page_content, metadata={"score": normalized_score[i]}
            )
            docs_with_scores.append(document)

        score_indexes = score_indexes[:top_k]

        # Creating an itemgetter object
        getter = itemgetter(*score_indexes)

        # Using itemgetter to get items
        selected_elements = getter(docs_with_scores)
        return selected_elements


def retrieve_cleaner(docs):
    rt_doc = {}

    for i, doc in enumerate(docs):
        name = f"retrieved_document{i} :"
        rt_doc[name] = doc.page_content

    return rt_doc


# class CustomRetriever(BaseRetriever):

#     documents: List[Document]
#     k: int = 4

#     def __init__(self, query_embedding_model, document_embedding_model, vectorstore):
#         self.query_embedding_model = query_embedding_model
#         self.document_embedding_model = document_embedding_model
#         self.vectorstore = vectorstore

#     def _get_relevant_documents(
#         self, query: str, *, run_manager: CallbackManagerForRetrieverRun
#     ) -> List[Document]:

#==================================================================
# class Custom_JSONLoader(BaseLoader):
#     """
#     Load a `JSON` file using a `jq` schema.

#     Setup:
#         .. code-block:: bash

#             pip install -U jq

#     Instantiate:
#         .. code-block:: python

#             from langchain_community.document_loaders import JSONLoader
#             import json
#             from pathlib import Path

#             file_path='./sample_quiz.json'
#             data = json.loads(Path(file_path).read_text())
#             loader = JSONLoader(
#                      file_path=file_path,
#                      jq_schema='.quiz',
#                      text_content=False)

#     Load:
#         .. code-block:: python

#             docs = loader.load()
#             print(docs[0].page_content[:100])
#             print(docs[0].metadata)

#         .. code-block:: python

#             {"sport": {"q1": {"question": "Which one is correct team name in
#             NBA?", "options": ["New York Bulls"
#             {'source': '/sample_quiz
#             .json', 'seq_num': 1}

#     Async load:
#         .. code-block:: python

#             docs = await loader.aload()
#             print(docs[0].page_content[:100])
#             print(docs[0].metadata)

#         .. code-block:: python

#             {"sport": {"q1": {"question": "Which one is correct team name in
#             NBA?", "options": ["New York Bulls"
#             {'source': '/sample_quizg
#             .json', 'seq_num': 1}

#     Lazy load:
#         .. code-block:: python

#             docs = []
#             docs_lazy = loader.lazy_load()

#             # async variant:
#             # docs_lazy = await loader.alazy_load()

#             for doc in docs_lazy:
#                 docs.append(doc)
#             print(docs[0].page_content[:100])
#             print(docs[0].metadata)

#         .. code-block:: python

#             {"sport": {"q1": {"question": "Which one is correct team name in
#             NBA?", "options": ["New York Bulls"
#             {'source': '/sample_quiz
#             .json', 'seq_num': 1}
#     """

#     def __init__(
#         self,
#         file_path: Union[str, Path],
#         jq_schema: str,
#         content_key: Optional[str] = None,
#         is_content_key_jq_parsable: Optional[bool] = False,
#         metadata_func: Optional[Callable[[Dict, Dict], Dict]] = None,
#         text_content: bool = True,
#         json_lines: bool = False,
#         encoding: str = "utf-8-sig",
#     ):
#         """Initialize the JSONLoader.

#         Args:
#             file_path (Union[str, Path]): The path to the JSON or JSON Lines file.
#             jq_schema (str): The jq schema to use to extract the data or text from
#                 the JSON.
#             content_key (str): The key to use to extract the content from
#                 the JSON if the jq_schema results to a list of objects (dict).
#                 If is_content_key_jq_parsable is True, this has to be a jq compatible
#                 schema. If is_content_key_jq_parsable is False, this should be a simple
#                 string key.
#             is_content_key_jq_parsable (bool): A flag to determine if
#                 content_key is parsable by jq or not. If True, content_key is
#                 treated as a jq schema and compiled accordingly. If False or if
#                 content_key is None, content_key is used as a simple string.
#                 Default is False.
#             metadata_func (Callable[Dict, Dict]): A function that takes in the JSON
#                 object extracted by the jq_schema and the default metadata and returns
#                 a dict of the updated metadata.
#             text_content (bool): Boolean flag to indicate whether the content is in
#                 string format, default to True.
#             json_lines (bool): Boolean flag to indicate whether the input is in
#                 JSON Lines format.
#         """
#         try:
#             import jq

#             self.jq = jq
#         except ImportError:
#             raise ImportError(
#                 "jq package not found, please install it with `pip install jq`"
#             )

#         self.file_path = Path(file_path).resolve()
#         self._jq_schema = jq.compile(jq_schema)
#         self._is_content_key_jq_parsable = is_content_key_jq_parsable
#         self._content_key = content_key
#         self._metadata_func = metadata_func
#         self._text_content = text_content
#         self._json_lines = json_lines

#     def lazy_load(self) -> Iterator[Document]:
#         """Load and return documents from the JSON file."""
#         index = 0
#         if self._json_lines:
#             with self.file_path.open(encoding=self.encdoing) as f:
#                 for line in f:
#                     line = line.strip()
#                     if line:
#                         for doc in self._parse(line, index):
#                             yield doc
#                             index += 1
#         else:
#             for doc in self._parse(self.file_path.read_text(encoding=self.encoding), index):
#                 yield doc
#                 index += 1

#     def _parse(self, content: str, index: int) -> Iterator[Document]:
#         """Convert given content to documents."""
#         data = self._jq_schema.input(json.loads(content))

#         # Perform some validation
#         # This is not a perfect validation, but it should catch most cases
#         # and prevent the user from getting a cryptic error later on.
#         if self._content_key is not None:
#             self._validate_content_key(data)
#         if self._metadata_func is not None:
#             self._validate_metadata_func(data)

#         for i, sample in enumerate(data, index + 1):
#             text = self._get_text(sample=sample)
#             metadata = self._get_metadata(
#                 sample=sample, source=str(self.file_path), seq_num=i
#             )
#             yield Document(page_content=text, metadata=metadata)

#     def _get_text(self, sample: Any) -> str:
#         """Convert sample to string format"""
#         if self._content_key is not None:
#             if self._is_content_key_jq_parsable:
#                 compiled_content_key = self.jq.compile(self._content_key)
#                 content = compiled_content_key.input(sample).first()
#             else:
#                 content = sample[self._content_key]
#         else:
#             content = sample

#         if self._text_content and not isinstance(content, str):
#             raise ValueError(
#                 f"Expected page_content is string, got {type(content)} instead. \
#                     Set `text_content=False` if the desired input for \
#                     `page_content` is not a string"
#             )

#         # In case the text is None, set it to an empty string
#         elif isinstance(content, str):
#             return content
#         elif isinstance(content, dict):
#             return json.dumps(content) if content else ""
#         else:
#             return str(content) if content is not None else ""

#     def _get_metadata(
#         self, sample: Dict[str, Any], **additional_fields: Any
#     ) -> Dict[str, Any]:
#         """
#         Return a metadata dictionary base on the existence of metadata_func
#         :param sample: single data payload
#         :param additional_fields: key-word arguments to be added as metadata values
#         :return:
#         """
#         if self._metadata_func is not None:
#             return self._metadata_func(sample, additional_fields)
#         else:
#             return additional_fields

#     def _validate_content_key(self, data: Any) -> None:
#         """Check if a content key is valid"""

#         sample = data.first()
#         if not isinstance(sample, dict):
#             raise ValueError(
#                 f"Expected the jq schema to result in a list of objects (dict), \
#                     so sample must be a dict but got `{type(sample)}`"
#             )

#         if (
#             not self._is_content_key_jq_parsable
#             and sample.get(self._content_key) is None
#         ):
#             raise ValueError(
#                 f"Expected the jq schema to result in a list of objects (dict) \
#                     with the key `{self._content_key}`"
#             )
#         if (
#             self._is_content_key_jq_parsable
#             and self.jq.compile(self._content_key).input(sample).text() is None
#         ):
#             raise ValueError(
#                 f"Expected the jq schema to result in a list of objects (dict) \
#                     with the key `{self._content_key}` which should be parsable by jq"
#             )

#     def _validate_metadata_func(self, data: Any) -> None:
#         """Check if the metadata_func output is valid"""

#         sample = data.first()
#         if self._metadata_func is not None:
#             sample_metadata = self._metadata_func(sample, {})
#             if not isinstance(sample_metadata, dict):
#                 raise ValueError(
#                     f"Expected the metadata_func to return a dict but got \
#                         `{type(sample_metadata)}`"
#                 )