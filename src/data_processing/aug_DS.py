'''

train 데이터의 질문과 답을 기반으로 위키피디아 문서 내에서 context를 새로 뽑아보기 (→ Distant supervision)

가설 → 다양한 context에서 질문과 답변이 나오게 되면, 이를 기반으로 새로운 질문에서 retriever가 관련 문서를 더 잘 가져올 수 있지 않을까

'''

import json
import pandas as pd
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from tqdm import tqdm

# 1. Train 데이터 로드
from datasets import load_from_disk

dataset = load_from_disk("data/raw/train_dataset")
train_dataset = dataset["train"]
train_data = pd.DataFrame(train_dataset)

# 2. DPR 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base').to(device)
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base').to(device)
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

# 3. 위키피디아 문서 로드
wikipedia_data_path = 'data/raw/wikipedia_documents.json'
with open(wikipedia_data_path, 'r') as f:
    wikipedia_data = json.load(f)

# Helper function to encode questions and contexts
def encode_question(questions, max_length=512):
    inputs = question_tokenizer(questions, return_tensors='pt', truncation=True, padding=True, max_length=max_length).to(device)
    question_emb = question_encoder(**inputs).pooler_output
    return question_emb

def encode_context(contexts, max_length=512):
    inputs = context_tokenizer(contexts, return_tensors='pt', truncation=True, padding=True, max_length=max_length).to(device)
    context_emb = context_encoder(**inputs).pooler_output
    return context_emb

# 4. 유사도 높은 문서 찾기
def find_similar_documents(questions, wikipedia_data):
    question_embs = encode_question(questions)
    
    similarities = []
    doc_ids = []
    
    # 각 문서에 대해 첫 문단의 embedding을 계산하고 유사도 비교
    for doc_id, document in wikipedia_data.items():
        first_paragraph = document['text'].split('\n')[0]  # 첫 문단 가져오기
        context_emb = encode_context(first_paragraph)
        similarity = cosine_similarity(question_embs.detach().cpu().numpy(), context_emb.detach().cpu().numpy())
        similarities.append(similarity[0])
        doc_ids.append(doc_id)
    
    # 유사도가 가장 높은 문서 선택
    best_match_indices = [similarity.argmax() for similarity in similarities]
    best_matches = [(doc_ids[idx], wikipedia_data[doc_ids[idx]]) for idx in best_match_indices]
    
    return best_matches

# 5. 증강된 데이터셋 생성
batch_size = 256
augmented_data = []

for idx in tqdm(range(0, train_data.shape[0], batch_size), desc="Augmenting Data"):
    
    batch = train_data.iloc[idx:idx + batch_size]
    questions = batch['question'].tolist()
    current_doc_ids = batch['document_id'].tolist()
    
    # DPR로 유사도가 높은 문서 찾기
    best_matches = find_similar_documents(questions, wikipedia_data)
    
    for i in range(len(batch)):
        new_doc_id, similar_doc = best_matches[i]
        current_doc_id = current_doc_ids[i]
        answer = batch.iloc[i]['answers']['text']
        
        # 만약 다른 문서라면 해당 문서 전체를 context로 사용
        if new_doc_id != current_doc_id:
            # 답이 전체 문서에 포함되는지 확인
            if answer in similar_doc['text']:
                # 기존 train 데이터에 행 추가 (새로운 문서와 전체 문서 사용)
                new_row = batch.iloc[i].copy()
                new_row['document_id'] = new_doc_id
                new_row['context'] = similar_doc['text']
                augmented_data.append(new_row)

# 6. 증강된 데이터 저장
from datasets import Dataset, DatasetDict

augmented_data = pd.DataFrame(augmented_data)
augmented_data = pd.concat([train_data, augmented_data], ignore_index=True)
train_dataset = Dataset.from_pandas(augmented_data)

dataset_dict = DatasetDict({
    'train': train_dataset
})

dataset_dict.save_to_disk('data/preprocessed/train_dataset_aug_DS')
