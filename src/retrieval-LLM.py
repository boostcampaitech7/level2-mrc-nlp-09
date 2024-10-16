import json
import pandas as pd
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import numpy as np

'''data'''
data = json.load(open('data/raw/wikipedia_documents.json'))
wiki = pd.DataFrame(data).T

dataset = load_from_disk("data/raw/train_dataset/")
train_df = pd.DataFrame(dataset['train'])
valid_df = pd.DataFrame(dataset['validation'])
mrc = pd.concat([train_df, valid_df])



'''inference'''
model = SentenceTransformer("intfloat/e5-mistral-7b-instruct")

# 문장 길이: max query length: 78, The 75th percentile length of the wiki: 857 ==> 1024
model.max_seq_length = 1024

query_n = 2 # query number
wiki_n = 30 - query_n # wiki number

# wiki = random wiki + correct answer included
queries = []
wiki_list = wiki['text'].tolist()[:wiki_n]
for i in range(query_n):
    queries.append(mrc['question'].iloc[i])
    wiki_list.append(mrc['context'].iloc[i])


query_embeddings = model.encode(queries, prompt_name="web_search_query")

batch_size = 10  # 한번에 처리할 문서 수 설정
top_k = 3
results = []

for i in range(len(queries)):
    
    # 배치로 document embedding을 처리
    all_scores = []
    for start_idx in tqdm(range(0, len(wiki_list), batch_size), desc=f"Processing query {i+1}/{len(queries)}"):
        batch_wiki_list = wiki_list[start_idx:start_idx + batch_size]
        document_embeddings = model.encode(batch_wiki_list)
        scores = (query_embeddings[i] @ document_embeddings.T) * 100
        all_scores.append(scores)
        
        del document_embeddings
        torch.cuda.empty_cache()

    # 점수 결합
    all_scores = np.concatenate(all_scores, axis=-1)
    
    # 상위 top_k 문서 찾기
    top_indices = np.argsort(all_scores)[::-1][:top_k]
    
    
    # 점수 후처리
    result = {
        'query': queries[i],
        'answer_text': mrc['context'].iloc[i]  # 예시로 mrc 데이터의 context를 answer_text로 사용
    }
    
    print('\n'+'*'*80)
    print(f"Query    : {queries[i]}")
    print(f"Answer   : {mrc['context'].iloc[i][:50]}...")
    print()
    # top k 문서와 그 점수를 딕셔너리에 저장
    for rank, idx in enumerate(top_indices, 1):
        result[f'top{rank}_text'] = wiki_list[idx]
        result[f'top{rank}_score'] = all_scores[idx]
        print(f"Top {rank} (Document {idx}), Score: {all_scores[idx]:.2f}")
        print(f"Documents: {wiki_list[idx][:50]}...")
        print()
    
    # 결과 리스트에 추가
    results.append(result)
    print('*'*80 + '\n')

# DataFrame 생성
df = pd.DataFrame(results)

# CSV로 저장
df.to_csv("data/experiments/query_topk_results.csv", index=False)