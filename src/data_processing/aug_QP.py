'''

알고리즘을 통해 가상으로 데이터 생성하기 (→ Question Paraphrasing)

가설 → train 데이터 기반에서 다양한 형태의 질문을 추가 학습시키면, 각 모델이 새로운 질문을 더 잘 이해하고 task를 수행할 수 있지 않을까

'''

import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm

dataset = load_from_disk("data/raw/train_dataset")
train_dataset = dataset["train"]
train_df = pd.DataFrame(train_dataset)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "allganize/Llama-3-Alpha-Ko-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def question2_batch(questions):
    prompts = [f"질문: {q}\n이 질문을 다른 표현으로 바꿔 주세요:\n새로운 질문:" for q in questions]
    
    input_ids = tokenizer.batch_encode_plus(prompts, return_tensors='pt', padding=True, truncation=True).to(device)

    output = model.generate(input_ids['input_ids'], max_length=128, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)

    paraphrased_questions = []
    for decoded in output:
        paraphrased_question = tokenizer.decode(decoded, skip_special_tokens=True)
        question_part = paraphrased_question.split("새로운 질문:")[1].split('\n')[0].strip()
        paraphrased_questions.append(question_part)

    return paraphrased_questions

tqdm.pandas(desc="Paraphrasing Questions")

batch_size = 4
new_rows = []

for i in tqdm(range(0, len(train_df), batch_size)):
    batch_questions = train_df['question'][i:i + batch_size].tolist()
    new_rows.extend(question2_batch(batch_questions))

new_df = train_df.copy()
new_df['question'] = new_rows
train_df = pd.concat([train_df, new_df], ignore_index=True)


from datasets import Dataset, DatasetDict

train_dataset = Dataset.from_pandas(train_df)

dataset_dict = DatasetDict({
    'train': train_dataset
})

dataset_dict.save_to_disk('data/preprocessed/train_dataset_aug_QP')
