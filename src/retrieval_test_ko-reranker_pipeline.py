import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()
    
# Load CSV file
df = pd.read_csv('data/pipeline/BM25Ensemble_top100_original.csv')

# # Initialize CrossEncoder model
model_path = 'Dongjin-kr/ko-reranker'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()


# Function to rank and sort contexts based on model's score
def rank_contexts(row):
    question = row['question']
    
    # Collect all top-k contexts (in this case, top-100)
    contexts = [row[f'top{i}_context'] for i in range(1, 101)]
    
    # Prepare pairs of question and contexts
    input_pairs = [[question, context] for context in contexts]
    
    # Predict scores for each pair
    with torch.no_grad():
        inputs = tokenizer(input_pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        scores = exp_normalize(scores.numpy())

    
    # Create a list of tuples (context, score) and sort them by score in descending order
    sorted_contexts = sorted(zip(contexts, scores), key=lambda x: x[1], reverse=True)
    
    # Return sorted contexts (only the contexts, not the scores)
    return [context for context, score in sorted_contexts]

# Apply the ranking function to each row in the dataframe
for index, row in tqdm(df.iterrows(), total=len(df), desc="Ranking contexts"):
    sorted_contexts = rank_contexts(row)
    
    # Save the sorted contexts back into the dataframe
    for i in range(1, 101):
        df.at[index, f'top{i}_context'] = sorted_contexts[i - 1]

# Save the updated dataframe back to a new CSV file
output_path = 'data/pipeline/BM25Ensemble_top100_ko-reranker.csv'  # 정렬된 결과를 저장할 경로
df.to_csv(output_path, index=False)

print(f"Sorted CSV saved to {output_path}")
