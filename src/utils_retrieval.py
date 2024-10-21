import pandas as pd
import numpy as np
import re
import torch
from tqdm import tqdm


import re
import torch
from tqdm import tqdm


def process_sentences(df, tokenizer, max_seq_length):
    """
    DataFrame의 각 row에 있는 문장들을 토큰화하고, 주어진 최대 시퀀스 길이에 맞게 문장을 분할하여 처리하는 함수입니다.

    각 row에 대해:
    - 각 문장을 토큰화합니다.
    - 토큰화된 문장들을 합쳐서 최대 시퀀스 길이를 넘기지 않는 범위에서 처리합니다.
    - 토큰 길이가 max_seq_length를 넘으면, 새로운 row를 생성하고 합쳐진 문장들을 저장한 후 다시 길이를 초기화합니다.
    - 모든 문장을 처리할 때까지 반복합니다.

    반환:
    - 처리된 결과가 담긴 새로운 DataFrame을 반환합니다.
    """
    new_rows = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking wiki documents to fit model input size"):
        sentences = row['context_sentences']
        token_length_sum = 0
        current_text = []
        
        for sentence in sentences:
            token_length = len(tokenizer.tokenize(sentence))
            
            if token_length_sum + token_length > max_seq_length:
                new_row = row.copy()
                new_row['text_processed'] = " ".join(current_text) # text_processed 대신 text로 나중에 바꾸기
                new_rows.append(new_row)
                
                token_length_sum = 0
                current_text = []
            
            token_length_sum += token_length
            current_text.append(sentence)
        
        if current_text:
            new_row = row.copy()
            new_row['text_processed'] = " ".join(current_text)
            new_rows.append(new_row)
    
    return pd.DataFrame(new_rows)




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