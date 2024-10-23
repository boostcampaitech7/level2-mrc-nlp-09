import json
import pandas as pd
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

import torch
from tqdm import tqdm
import numpy as np
import random

random.seed(42)


def create_topn_dataframe(mrc_samples, model_wiki_samples, raw_wiki, similarities, topn=100, file_name="wiki_sp-topn_dn-topn.csv"):
    # 각 query에 대해 topn 유사도를 가지는 context와 점수를 담을 리스트 준비
    topn_contexts = []

    for i, query_id in tqdm(enumerate(mrc_samples['document_id']), desc="Creating topn DataFrame"):
        # 유사도 기준으로 상위 n개의 index를 가져옴
        topn_scores, topn_indices = torch.topk(similarities[i], k=topn, largest=True)
        if topn > 1:
            topn_scores, topn_indices = topn_scores.tolist(), topn_indices.tolist()
        else:
            topn_scores, topn_indices = top_scores.item(), top_indices.item()
        
        # 새로운 row 생성
        row = {
            'id': mrc_samples.iloc[i]['id'],
            'question': mrc_samples.iloc[i]['question'],
            'answer': mrc_samples.iloc[i]['context']
        }
        
        # top1, top2, ..., topn context와 score 추가
        correct = False
        for j, idx in enumerate(topn_indices):
            doc_id = model_wiki_samples.iloc[idx]['document_id']
            
            # 해당 doc_id의 text를 raw_wiki에서 가져옴
            text = raw_wiki[raw_wiki['document_id'] == doc_id]['text'].values
            
            # 값이 존재하면 context 추가, 없으면 None
            row[f'top{j+1}_context'] = text[0] if len(text) > 0 else None
            row[f'top{j+1}_score'] = topn_scores[j]
            
            # 정답 여부 확인
            if doc_id == query_id:
                correct = True
        
        # correct 여부 추가
        row['correct'] = correct
        topn_contexts.append(row)

    # 새로운 DataFrame 생성
    df_topn = pd.DataFrame(topn_contexts)
    # save
    df_topn.to_csv(f"data/experiments/{file_name}", index=False)
    print("Saved topn DataFrame to data/experiments/{file_name}")

    return df_topn


def create_mrc_input_csv(mrc_samples, model_wiki_samples, similarities, file_name="wiki_sp-topn_dn-topn.csv"):
    # N개의 top contexts를 담을 리스트 준비
    top_n = len(model_wiki_samples)

    # 각 query에 대해 topn 유사도를 가지는 context를 담을 새로운 DataFrame 생성
    topn_contexts = []

    for i, query_id in tqdm(enumerate(mrc_samples['document_id']), desc="Creating MRC input CSV"):
        # 유사도 기준으로 상위 n개의 index를 가져옴
        topn_values, topn_indices = torch.topk(similarities[i], k=top_n, largest=True)
        topn_context = model_wiki_samples.iloc[topn_indices.cpu().numpy()]['text'].tolist()


        # 새로운 row 생성
        row = {
            'id': mrc_samples.iloc[i]['id'],
            'question': mrc_samples.iloc[i]['question'],
        }

        # top1, top2, ..., topn context 추가
        for j in range(top_n):
            row[f'top{j+1}_context'] = topn_context[j]

        topn_contexts.append(row)

    # 새로운 DataFrame 생성
    result_to_mrc = pd.DataFrame(topn_contexts)

    # DataFrame을 .csv 파일로 저장
    result_to_mrc.to_csv(f"data/pipeline/{file_name}", index=False)
    print(f"Saved MRC input to data/pipeline/{file_name}")

    # 결과 반환
    return result_to_mrc


def calculate_topn_percentage(df_topn, topn_list):
    # 각 topn에 대한 정답 비율을 저장할 딕셔너리
    topn_correct_percentage = {}

    # topn 리스트에 있는 각 값을 처리
    for topn in tqdm(topn_list, desc="Calculating topn percentage"):
        correct_count = 0
        total_count = len(df_topn)

        # 각 row에 대해 정답이 있는지 확인
        for index, row in df_topn.iterrows():
            # topn에 해당하는 열을 검사 (top1, top2, ..., topn)
            correct_found = False
            for n in range(1, topn + 1):
                if row[f'top{n}_context'] in row['answer']:  # correct가 True이면 정답
                    correct_found = True
                    break

            if correct_found:
                correct_count += 1

        # 정답 비율 계산
        percentage = (correct_count / total_count) * 100
        topn_correct_percentage[f'top {topn}'] = percentage

    return topn_correct_percentage



'''data'''
# data = json.load(open('data/raw/wikipedia_documents.json'))
model_data = json.load(open('data/preprocessed/wikipedia_model.json'))
model_wiki = pd.DataFrame(model_data).T
raw_data = json.load(open('data/raw/wikipedia_documents.json'))
raw_wiki = pd.DataFrame(raw_data).T

dataset = load_from_disk("data/raw/train_dataset/")
train_df = pd.DataFrame(dataset['train'])
valid_df = pd.DataFrame(dataset['validation'])
mrc = pd.concat([train_df, valid_df])

query_n = 100 # query number
wiki_n = 100 # wiki number
topn = 100
save_file_name = f"wiki_sp-top-{wiki_n}_dn-top-all.csv"


assert query_n <= wiki_n, 'query_n should be smaller than total wiki_n'

sampled_indices = random.sample(range(len(mrc)), query_n)
print(f'Sampled indices (Randomly Sampled 0~len(mrc)): {sampled_indices}')

mrc_samples = mrc.iloc[sampled_indices]
raw_wiki_document_ids = mrc_samples['document_id'].tolist()
model_wiki_samples = model_wiki[model_wiki['document_id'].isin(raw_wiki_document_ids)]
available_document_ids = [i for i in range(model_wiki['document_id'].max()) if i not in raw_wiki_document_ids]
extra_raw_wiki_document_ids = random.sample(available_document_ids, wiki_n - query_n)
print(f'Raw wiki document ids in MRC samples df: {raw_wiki_document_ids}')
print(f'Extra Raw wiki document ids: {extra_raw_wiki_document_ids}')
print(f'MRC samples df added: {mrc_samples.shape[0]}')
print(f'Model-Wiki samples df added (Answer): {model_wiki_samples.shape[0]}')

extra_model_wiki_samples = model_wiki[model_wiki['document_id'].isin(extra_raw_wiki_document_ids)]
model_wiki_samples = pd.concat([model_wiki_samples, extra_model_wiki_samples])
print(f'Model-Wiki samples df added (Extra): {extra_model_wiki_samples.shape[0]}')
print(f'Model-Wiki samples df added (Answer + Extra): {model_wiki_samples.shape[0]}')


'''model'''
prompts = {
    "query": "query: ",        # 검색 쿼리 프롬프트
    "passage": "passage: "     # 문서 패시지 프롬프트
}

model_name = "nlpai-lab/KoE5"
model = SentenceTransformer(
    model_name_or_path=model_name, 
    device='cuda', 
    similarity_fn_name='dot',
    # truncate_dim=512,
    # model_kwargs={"torch_dtype": torch.bfloat16},
    prompts=prompts,
    )

print(f"Queries n: {mrc_samples.shape[0]}")
print(f"From wiki n: {model_wiki_samples.shape[0]}")
print(f'Model: {model_name}')


query_embeddings = model.encode(mrc_samples['question'].tolist(), prompt_name="query", show_progress_bar=True)
document_embeddings = model.encode(model_wiki_samples['text'].tolist(), prompt_name="passage", show_progress_bar=True)
similarities = model.similarity(query_embeddings, document_embeddings) * 100


topn_list = [i for i in range(topn+1)][1:]  # 원하는 topn 설정
df_topn = create_topn_dataframe(mrc_samples, model_wiki_samples, raw_wiki, similarities, topn=topn, file_name=save_file_name)

topn_percentage = calculate_topn_percentage(df_topn, topn_list)
for topn, percentage in topn_percentage.items():
    print(f"{topn}: {percentage:.2f}% correct")
    if percentage == 100:
        break

create_mrc_input_csv(mrc_samples, model_wiki_samples, similarities, file_name=save_file_name)