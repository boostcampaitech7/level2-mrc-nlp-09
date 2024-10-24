import json
import pandas as pd
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

import torch
from tqdm import tqdm
import numpy as np
import random
import os

random.seed(42)
tqdm.pandas()


def safe_encode(model, texts, batch_size=32, prompt_name=None):
    if prompt_name is not None:
        while batch_size > 1:
            try:
                return model.encode(texts, batch_size=batch_size, prompt_name=prompt_name, show_progress_bar=True)
            except torch.cuda.OutOfMemoryError:
                batch_size = batch_size // 2
                print(f"Out of memory, reducing batch size to {batch_size}")
        raise RuntimeError("Batch size too small, still out of memory.")
    else:
        while batch_size > 1:
            try:
                return model.encode(texts, batch_size=batch_size, show_progress_bar=True)
            except torch.cuda.OutOfMemoryError:
                batch_size = batch_size // 2
                print(f"Out of memory, reducing batch size to {batch_size}")
        raise RuntimeError("Batch size too small, still out of memory.")
        


def calculate_and_save_similarities(sparse_retrieval_results, model, topn=100, pooling='max', analysis=False, file_name="output.csv"):
    similarities = []
    reordered_data = []

    # 결과 저장 경로가 없으면 생성
    os.makedirs("data/pipeline", exist_ok=True)
    
    for idx, row in tqdm(sparse_retrieval_results.iterrows(), desc="Calculating and sorting similarities", total=len(sparse_retrieval_results)):
        # print(f"\nProcessing row {idx+1}/{len(sparse_retrieval_results)}: {row['id']}")
        
        # 쿼리 임베딩 생성
        query_embedding = safe_encode(model, row['question'], batch_size=32, prompt_name="query")

        # print(f"Query embedding created for question: {row['question']}")

        # 각 쿼리별로 top1_context부터 topn_context까지 모두 가져와서 유사도 계산
        contexts = [row[f'top{i+1}_context'] for i in range(topn) if f'top{i+1}_context' in row]
        # print(f"Retrieved {len(contexts)} contexts for this query.")
        
        max_length = min(model.max_seq_length, model[0].auto_model.config.max_position_embeddings)
        # print(f"Model max sequence length: {max_length}")

        context_chunked = []
        context_index_map = {}  # 원본 인덱스를 저장할 맵

        # 문서를 chunk로 나눔
        for i, context in enumerate(contexts):
            tokenized_context = model.tokenizer(context, truncation=False)['input_ids']
            # print(f"Context {i+1} token length: {len(tokenized_context)}")

            if len(tokenized_context) > max_length:
                # 토큰 길이가 max_length를 넘으면 잘라서 처리
                chunks = [tokenized_context[j:j+max_length] for j in range(0, len(tokenized_context), max_length)]
                # print(f"Context {i+1} split into {len(chunks)} chunks.")
                
                for chunk in chunks:
                    chunk_text = model.tokenizer.decode(chunk, skip_special_tokens=True)
                    context_chunked.append(chunk_text)
                    if i in context_index_map:
                        context_index_map[i].append(len(context_chunked) - 1)
                    else:
                        context_index_map[i] = [len(context_chunked) - 1]
            else:
                # 길이가 max_length 이하이면 그대로 처리
                context_chunked.append(context)
                context_index_map[i] = [len(context_chunked) - 1]

        # print(f"Total number of chunks to be encoded: {len(context_chunked)}")

        # 모든 chunk에 대해 유사도를 한꺼번에 계산
        # context_embeddings = model.encode(context_chunked)
        context_embeddings = safe_encode(model, context_chunked, batch_size=32)

        # print(f"Context embeddings shape: {context_embeddings.shape}")
        similarity_scores = (query_embedding @ context_embeddings.T).squeeze().tolist()

        # print(f"Calculated similarity scores for {len(similarity_scores)} chunks.")

        # pooling 방법에 따라 chunk들을 하나로 합침
        final_similarity_scores = []
        for i in range(len(contexts)):
            if len(context_index_map[i]) > 1:  # chunk로 나눠졌다면
                if pooling == 'max':
                    score = max([similarity_scores[j] for j in context_index_map[i]])
                elif pooling == 'mean':
                    score = sum([similarity_scores[j] for j in context_index_map[i]]) / len(context_index_map[i])
                else:
                    raise ValueError("Invalid pooling method. Choose 'max' or 'mean'.")
                final_similarity_scores.append(score)
            else:
                final_similarity_scores.append(similarity_scores[context_index_map[i][0]])


        # 유사도 기준으로 context와 score를 묶어 정렬 (높은 순)
        sorted_contexts_scores = sorted(zip(final_similarity_scores, contexts), reverse=True)
        # print(f"Top similarity score for this query: {sorted_contexts_scores[0][0]:.2f}")
        # print(f"Length of sorted contexts: {len(sorted_contexts_scores)}")

        # 정렬된 context와 유사도를 다시 row로 저장
        new_row = {
            'id': row['id'],
            'question': row['question'],
        }
        
        for i, (score, context) in enumerate(sorted_contexts_scores[:topn]):  # topn개의 context만 저장
            new_row[f'top{i+1}_context'] = context
            if analysis:
                new_row[f'top{i+1}_score'] = score * 100

        reordered_data.append(new_row)

    # 정렬된 데이터를 DataFrame으로 변환
    reordered_df = pd.DataFrame(reordered_data)

    # csv 파일로 저장
    reordered_df.to_csv(f"data/pipeline/{file_name}_{pooling}", index=False)
    print(f"Saved reordered similarities to data/pipeline/{file_name}_{pooling}")

    return reordered_df




# 모델
model_name = "Alibaba-NLP/gte-Qwen2-7B-instruct"
model = SentenceTransformer(
    model_name_or_path=model_name, 
    # device='cuda', 
    # similarity_fn_name='dot',
    # truncate_dim=512,
    # model_kwargs={"torch_dtype": torch.bfloat16},
    # prompts=prompts,
    trust_remote_code=True
    )
model.max_seq_length = 512

max_length = min(model.max_seq_length, model[0].auto_model.config.max_position_embeddings)

print(f'Model: {model_name}')
print(f'Max sequence length: {max_length}')


# 데이터
'''Sparse retrieval로 받아온 set으로 테스트'''
input_file = "BM25Ensemble_top100_original.csv"
output_file = "BM25Ensemble_top100_Qwen.csv"
sparse_retrieval_results = pd.read_csv(f'data/pipeline/{input_file}')
sparse_retrieval_results = sparse_retrieval_results[:10]
print(f"Input Shape: {sparse_retrieval_results.shape}")

sorted_df = calculate_and_save_similarities(
    sparse_retrieval_results=sparse_retrieval_results, 
    model=model, 
    topn=sum(1 for col in sparse_retrieval_results.columns if 'context' in col),
    pooling='max',
    analysis=False,
    file_name=f"{output_file}"
)

sorted_df.head()