import pandas as pd
import numpy as np
import re
import torch
from tqdm import tqdm


import re
import torch
from tqdm import tqdm

def sentence_split(text):
    # 간단한 문장 단위 분리 함수 (. ! ? 기준으로 문장을 분리)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return sentences


def chunking_with_overlap(sentences, max_length, overlap, include_position=False):
    chunks = []
    positions = []  # 순서 정보 저장 리스트

    current_chunk = []
    total_length = 0
    start_index = 0

    for i, sentence in enumerate(sentences):
        sentence_length = len(sentence.split())
        total_length += sentence_length

        # 현재 문장이 추가되어도 최대 길이를 넘지 않는 경우
        if total_length <= max_length:
            current_chunk.append(sentence)
        else:
            # chunk 생성 후 초기화
            chunks.append(" ".join(current_chunk))
            if include_position:
                positions.append((start_index, i-1))  # 시작, 끝 문장 인덱스 저장
            # 다음 chunk 생성 준비 (overlap 문장 포함)
            current_chunk = sentences[max(0, i-overlap):i]
            total_length = sum(len(s.split()) for s in current_chunk)
            start_index = i - overlap

        # 마지막 남은 chunk 처리
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        if include_position:
            positions.append((start_index, len(sentences)-1))

    return chunks, positions if include_position else chunks

# 적용 함수
def process_wiki_documents(wiki, max_length, overlap, include_position=False):
    processed_docs = []
    doc_positions = []  # 순서 정보를 저장할 리스트

    for doc_id, text in tqdm(enumerate(wiki), total=len(wiki), desc="Processing wiki documents"):
        sentences = sentence_split(text)  # 문장 단위로 문서 분리
        if len(text.split()) > max_length:
            chunks, positions = chunking_with_overlap(sentences, max_length, overlap, include_position)
            processed_docs.extend(chunks)  # 자른 청크들 추가
            if include_position:
                doc_positions.extend(positions)  # 순서 정보 추가
        else:
            processed_docs.append(text)  # 길이가 짧으면 그냥 추가

    return (processed_docs, doc_positions) if include_position else processed_docs

# 파라미터 설정
max_length = 512  # 모델의 최대 입력 길이
overlap = 2  # 겹치는 문장 수
include_position = True  # 순서 정보 포함 여부

# 예시 위키 문서 리스트 (30000개를 wiki에 담고 있다고 가정)
wiki_list = ["This is a long document example. It consists of multiple sentences. Here is another sentence."] * 30000

# 문서 처리
processed_wiki_docs, doc_positions = process_wiki_documents(wiki_list, max_length, overlap, include_position)

print(f"Processed {len(processed_wiki_docs)} wiki chunks.")
if include_position:
    print(f"Document positions: {doc_positions[:5]}")  # 순서 정보 예시 출력





def save_results(queries, correct_contexts, similarities, split_passage_info, top_k, output_file):
    results = []
    correct_count = 0
    
    for i in range(len(queries)):
        query_scores = similarities[i]

        # 상위 top_k 문서 찾기 (유사도 내림차순으로 정렬)
        top_indices = np.argsort(query_scores)[::-1][:top_k]

        # 정답확인
        correct = False
        for idx in top_indices:
            if correct_contexts[i] == split_passage_info[idx]['original_text']:
                correct = True
                correct_count += 1
                break

        # 결과 기록
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

        # 상위 top_k 문서 및 점수 기록
        for rank, idx in enumerate(top_indices, 1):
            original_doc = split_passage_info[idx]['original_text']
            result[f'top{rank}_text'] = split_passage_info[idx]['original_text']
            result[f'top{rank}_score'] = query_scores[idx]
            if rank <= 3:  # 상위 3개의 문서만 출력
                print(f"Top {rank} (Document {split_passage_info[idx]['doc_id']}), Score: {query_scores[idx]:.2f}")
                print(f"Document Excerpt: {original_doc[:50]}...")
                print()

        results.append(result)
        print('*' * 80 + '\n')

    # DataFrame 생성
    df = pd.DataFrame(results)

    # 정답률(accuracy) 계산 및 출력
    accuracy = correct_count / len(queries)
    print(f"\nAccuracy: {accuracy:.2%} ({correct_count}/{len(queries)})")

    # DataFrame을 CSV 파일로 저장
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")