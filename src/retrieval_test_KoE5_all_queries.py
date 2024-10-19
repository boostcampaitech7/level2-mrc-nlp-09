import json
import pandas as pd
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

import torch
from tqdm import tqdm
import numpy as np
import random


def embed_long_document(model, document, max_length=512):
    # 문서를 512 토큰 단위로 나눔
    tokens = tokenizer(document, return_tensors='pt', truncation=False)['input_ids']
    if tokens.size(1) <= max_length:
        # 문서가 512 토큰 이하면 그대로 임베딩
        return model.encode(document, convert_to_tensor=True, normalize_embeddings=True)
    
    # 문서를 512 토큰씩 나눠서 임베딩
    chunks = torch.split(tokens, max_length, dim=1)
    embeddings = []
    for chunk in chunks:
        chunk_text = tokenizer.decode(chunk.squeeze(), skip_special_tokens=True)
        chunk_embedding = model.encode(chunk_text, convert_to_tensor=True, normalize_embeddings=True)
        embeddings.append(chunk_embedding)

    # 각 청크의 임베딩을 결합
    if mode == 'mean':
        return torch.mean(torch.stack(embeddings), dim=0)
    elif mode == 'max':
        return torch.max(torch.stack(embeddings), dim=0)[0]
    else:
        raise ValueError(f"Invalid pooling mode: {mode}")


random.seed(42)

'''data'''
data = json.load(open('data/raw/wikipedia_documents.json'))
wiki = pd.DataFrame(data).T
dataset = load_from_disk("data/raw/train_dataset/")
train_df = pd.DataFrame(dataset['train'])
valid_df = pd.DataFrame(dataset['validation'])
mrc = pd.concat([train_df, valid_df])


'''inference'''
model_name = "nlpai-lab/KoE5"
model = SentenceTransformer(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

query_n = mrc.shape[0] # query number
wiki_n = wiki.shape[0] # wiki number
top_k = 100
mode = 'mean'  # mean or max pooling

model.max_seq_length = 512

print(f"Query number: {query_n}")
print(f"Wiki number: {wiki_n}")
print(f"Model: {model_name}")
print(f"Max sequence length: {model.max_seq_length}")
print(f"Top k: {top_k}")
print(f"Pooling mode: {mode}")
print()

queries = mrc['question'].tolist()
correct_contexts = mrc['context'].tolist()
wiki_list = wiki['text'].tolist()
random.shuffle(wiki_list)

query_embeddings = model.encode(queries, convert_to_tensor=True, normalize_embeddings=True)
wiki_embeddings = torch.stack([embed_long_document(model, doc) for doc in tqdm(wiki_list, desc="Embedding all wiki documents")])

print("Calculating similarities...")
similarities = model.similarity(query_embeddings, wiki_embeddings) * 100  # 모든 쿼리와 모든 위키에 대해 유사도 계산
similarities = similarities.cpu().numpy()


# 결과 저장
results = []
correct_count = 0
for i in range(len(queries)):
    query_scores = similarities[i]

    # 상위 top_k 문서 찾기
    top_indices = np.argsort(query_scores)[::-1][:top_k]

    # 정답확인
    correct = False
    for idx in top_indices:
        if correct_contexts[i] == wiki_list[idx]:
            correct = True
            correct_count += 1
            break

    result = {
        'query': queries[i],
        'answer_text': correct_contexts[i],
        'correct': correct
    }


    print('\n' + '*' * 80)
    print(f"Query    : {queries[i]}")
    print(f"Answer   : {correct_contexts[i][:50]}...")
    print(f"Correct  : {correct}")
    print()

    for rank, idx in enumerate(top_indices, 1):
        result[f'top{rank}_text'] = wiki_list[idx]
        result[f'top{rank}_score'] = query_scores[idx]
        if rank <= 5:
            print(f"Top {rank} (Document {idx}), Score: {query_scores[idx]:.2f}")
            print(f"Documents: {wiki_list[idx][:50]}...")
            print()

    results.append(result)
    print('*' * 80 + '\n')

# DataFrame 생성
df = pd.DataFrame(results)

# 정답률(accuracy) 계산 및 출력
accuracy = correct_count / len(queries)
print(f"\nAccuracy: {accuracy:.2%} ({correct_count}/{len(queries)})")

# CSV로 저장
if mode == 'mean':
    df.to_csv("data/experiments/KoE5_mean_pooling_all_queries.csv", index=False)
elif mode == 'max':
    df.to_csv("data/experiments/KoE5_max_pooling_all_queries.csv", index=False)