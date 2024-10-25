import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# Llama 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("allganize/Llama-3-Alpha-Ko-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("allganize/Llama-3-Alpha-Ko-8B-Instruct")

def generate_answer(context, question):
    # 입력 텍스트 생성
    input_text = f"질문: {question}\n\n문맥: {context}\n\n답변(단어로만):"
    inputs = tokenizer.encode(input_text, return_tensors='pt')

    # 모델을 통해 답변 생성
    outputs = model.generate(inputs, max_new_tokens=50, num_return_sequences=1)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 'Answer:' 이후의 실제 답변만 반환
    return answer.split("Answer:")[-1].strip()

# JSON 데이터에서 질문과 컨텍스트 추출
file_path = 'extracted_data.json'

with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 새 답변을 추가할 데이터 저장소
new_answers = {}

# 데이터의 절반만 처리
half_data = list(data.items())[:len(data)//2]  # 데이터의 절반을 선택

# 각 항목에 대해 답변 생성
for key, value in half_data:
    context = value['context']
    question = value['question']
    answer = generate_answer(context, question)
    new_answers[key] = answer

# 결과를 JSON 파일로 저장
with open('new_answers.json', 'w', encoding='utf-8') as f:
    json.dump(new_answers, f, ensure_ascii=False, indent=4)
