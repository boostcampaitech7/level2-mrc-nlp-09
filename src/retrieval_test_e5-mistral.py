import json
import pandas as pd
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import numpy as np
import random

random.seed(42)

'''data'''
data = json.load(open('data/preprocessed/wikipedia_model.json'))
wiki = pd.DataFrame(data).T
dataset = load_from_disk("data/raw/train_dataset/")
train_df = pd.DataFrame(dataset['train'])
valid_df = pd.DataFrame(dataset['validation'])
mrc = pd.concat([train_df, valid_df])


'''inference'''
model_name = "intfloat/e5-mistral-7b-instruct"
model = SentenceTransformer(model_name)

query_n = 50 # query number
wiki_n = 100 # wiki number
top_k = 100

# 모델 input 최대 길이, 모델 Mem에 따라 조절
batch_size = 5  # 한번에 처리할 문서 수 설정
model.max_seq_length = 2048

print(f"Query number: {query_n}")
print(f"Wiki number: {wiki_n}")
print(f"Batch size: {batch_size}")
print(f'Model: {model_name}')
print(f"Max sequence length: {model.max_seq_length}")
print(f"Top k: {top_k}")


# wiki = random wiki + correct answer included
sampled_indices = random.sample(range(len(mrc)), query_n)
queries = [mrc['question'].iloc[idx] for idx in sampled_indices]
correct_contexts = [mrc['context'].iloc[idx] for idx in sampled_indices]
wiki_list = random.sample(wiki['text'].tolist(), wiki_n - query_n) + correct_contexts
random.shuffle(wiki_list)

results = []
correct_count = 0
for i in range(len(queries)):
    query_embedding = model.encode(queries[i], prompt_name="web_search_query")

    # 배치로 document embedding을 처리
    all_scores = []
    for start_idx in tqdm(range(0, len(wiki_list), batch_size), desc=f"Processing query {i+1}/{len(queries)}"):
        batch_wiki_list = wiki_list[start_idx:start_idx + batch_size]
        document_embeddings = model.encode(batch_wiki_list)
        scores = (query_embedding @ document_embeddings.T) * 100
        all_scores.append(scores)

    # 점수 결합
    all_scores = np.concatenate(all_scores, axis=-1)
    
    # 상위 top_k 문서 찾기
    top_indices = np.argsort(all_scores)[::-1][:top_k]
    
    
    # 정답확인
    correct = False
    for idx in top_indices:
        if correct_contexts[i] == wiki_list[idx]:  # 정답이 포함되어 있는지 확인
            correct = True
            correct_count += 1  # 정답이 포함되면 카운트 증가
            break

    # 결과 딕셔너리 생성
    result = {
        'query': queries[i],
        'answer_text': correct_contexts[i],  # 예시로 mrc 데이터의 context를 answer_text로 사용
        'correct': correct  # 정답 여부 추가
    }
    
    print('\n'+'*'*80)
    print(f"Query    : {queries[i]}")
    print(f"Answer   : {correct_contexts[i][:50]}...")
    print(f"Correct  : {correct}")
    print()
    # top k 문서와 그 점수를 딕셔너리에 저장
    for rank, idx in enumerate(top_indices, 1):
        result[f'top{rank}_text'] = wiki_list[idx]
        result[f'top{rank}_score'] = all_scores[idx]
        if rank <= 5:
            print(f"Top {rank} (Document {idx}), Score: {all_scores[idx]:.2f}")
            print(f"Documents: {wiki_list[idx][:50]}...")
            print()
    
    # 결과 리스트에 추가
    results.append(result)
    print('*'*80 + '\n')

# DataFrame 생성
df = pd.DataFrame(results)

# 정답률(accuracy) 계산 및 출력
accuracy = correct_count / len(queries)
print(f"\nAccuracy: {accuracy:.2%} ({correct_count}/{len(queries)})")

# CSV로 저장
df.to_csv("data/experiments/e5-mistral.csv", index=False)