import json
import pandas as pd
from datasets import load_from_disk

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

import torch
from tqdm import tqdm
import numpy as np
import random

import utils_retrieval as utils


def sliding_window_tokenize(text, tokenizer, max_length, stride):
    tokens = tokenizer(
        text, truncation=True, max_length=max_length, stride=stride, return_overflowing_tokens=True
    )
    
    # input_ids와 overflow된 부분을 모두 하나의 리스트로 반환
    chunks = [tokens["input_ids"]]  # 첫 번째 청크
    while "overflowing_tokens" in tokens:
        # 초과된 토큰들을 처리해 다음 청크 생성
        tokens = tokenizer(
            tokenizer.decode(tokens["overflowing_tokens"]), 
            truncation=True, max_length=max_length, stride=stride, return_overflowing_tokens=True
        )
        chunks.append(tokens["input_ids"])  # 각 청크 추가
    
    # 각 청크를 문자열로 변환해서 반환
    return [tokenizer.decode(chunk) for chunk in chunks]



random.seed(42)

'''data'''
data = json.load(open('data/raw/wikipedia_documents.json'))
wiki = pd.DataFrame(data).T
dataset = load_from_disk("data/raw/train_dataset/")
train_df = pd.DataFrame(dataset['train'])
valid_df = pd.DataFrame(dataset['validation'])
mrc = pd.concat([train_df, valid_df])

queries = mrc['question'].tolist()[:2]
correct_contexts = mrc['context'].tolist()[:2]
wiki_list = wiki['text'].tolist()[:10]
random.shuffle(wiki_list)

top_k = 100

'''model'''
prompts = {
    "query": "query: ",        # 검색 쿼리 프롬프트
    "passage": "passage: "     # 문서 패시지 프롬프트
}
model_name = "nlpai-lab/KoE5"
model = SentenceTransformer(
    model_name_or_path=model_name, 
    similarity_fn_name='dot',
    prompts=prompts,
    )
# model.set_pooling_include_prompt(True) # KoE5는 instructor아닌듯
model.eval()


'''Description'''
print()
print(f"Query number: {len(queries)}")
print(f"Wiki number: {len(wiki_list)}")
print(f"Model: {model_name}")
print(f"Top k: {top_k}")
print()


# '''Sliding window'''
max_length = min(model.max_seq_length, model[0].auto_model.config.max_position_embeddings)  # 최대 시퀀스 길이
stride = int(max_length * 0.5)  # 50% 겹침

split_passages = []
split_passage_info = []  # 청크의 원본 문서 정보를 저장할 리스트

# 문서를 슬라이딩 윈도우로 나누고, 토큰 ID를 다시 문자열로 변환
for idx, passage in tqdm(enumerate(wiki_list), total=len(wiki_list), desc="Processing wiki passages"):
    chunked = sliding_window_tokenize(passage, model.tokenizer, max_length, stride)
    split_passages.extend(chunked)
    
    # 각 청크에 원본 문서의 인덱스와 시작, 끝 토큰 정보를 추가
    for chunk in chunked:
        split_passage_info.append({
            "doc_id": idx,  # 원본 문서의 인덱스
            "chunk": chunk,  # 청크
            "original_text": passage  # 나중에 reader로 넘길 원본 문서
        })

with torch.no_grad():
    query_embeddings = model.encode(
        sentences=queries, 
        prompt_name="query", 
        batch_size=320,
        show_progress_bar=True,
        )
    wiki_embeddings = model.encode(
        sentences=[info['chunk'] for info in split_passage_info],  # 청크들로부터 임베딩을 구함
        prompt_name="passage", 
        batch_size=320,
        show_progress_bar=True,
        )
    
    print("Calculating similarities...")
    similarities = model.similarity(query_embeddings, wiki_embeddings)

# top_chunk_idx = np.argmax(similarities, axis=1)
# retrieved_original_docs = [split_passage_info[i]['original_text'] for i in top_chunk_idx]



# 함수 호출
output_file = "data/experiments/KoE5_sliding_windows_all_queries.csv"
utils.save_results(results, queries, correct_contexts, similarities, split_passage_info, top_k, correct_count, output_file)


