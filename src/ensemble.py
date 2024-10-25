import json
import numpy as np
from collections import Counter

# JSON 파일 경로
file_paths = [
    'input/checkpoint/ST_Ensemble_02/nbest_predictions_test.json', 
    'input/checkpoint/ST_Ensemble_03/nbest_predictions_test.json', # HANTAEK
    'nbest_monologg.json',
    # 'nbest_HANTAEK.json',
    'nbest_base.json',
]

# JSON 파일 읽기 및 데이터 합치기
def load_data(file_paths):
    all_data = {}
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for key, value in data.items():
                if key not in all_data:
                    all_data[key] = []
                all_data[key].extend(value)
    return all_data

# 하드 보팅 구현
def hard_voting(predictions):
    final_predictions = {}
    for key, preds in predictions.items():
        # 각 예측에서 가장 많이 나타나는 클래스 선택
        votes = [pred['text'] for pred in preds]
        most_common = Counter(votes).most_common(1)[0][0]
        final_predictions[key] = most_common
    return final_predictions

# 소프트 보팅 구현
# def soft_voting(predictions):
#    final_predictions = {}
#    for key, preds in predictions.items():
#        # 각 클래스에 대한 확률 계산
#        total_scores = {}
#        for pred in preds:
#            text = pred['text']
#            score = pred['score']
#            if text not in total_scores:
#                total_scores[text] = []
#            total_scores[text].append(score)
#
#        # 각 클래스의 평균 점수 계산
#        avg_scores = {text: np.mean(scores) for text, scores in total_scores.items()}
#        # 가장 높은 평균 점수를 가진 클래스 선택
#        best_prediction = max(avg_scores, key=avg_scores.get)
#        final_predictions[key] = best_prediction
#    return final_predictions

# 데이터 로드
data = load_data(file_paths)

# 하드 보팅 및 소프트 보팅 결과
hard_voting_results = hard_voting(data)
# soft_voting_results = soft_voting(data)

# 결과 저장
with open('hard_voting_predictions_v6.json', 'w', encoding='utf-8') as hv_file:
    json.dump(hard_voting_results, hv_file, ensure_ascii=False, indent=4)

# with open('soft_voting_predictions_v2.json', 'w', encoding='utf-8') as sv_file:
#     json.dump(soft_voting_results, sv_file, ensure_ascii=False, indent=4)