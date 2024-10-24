import pandas as pd
from sentence_transformers import CrossEncoder
import torch
from tqdm import tqdm
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, required=True, help="Path to the input CSV file")
parser.add_argument('--output_file', type=str, required=True, help="Path to save the output CSV file")
args = parser.parse_args()

print(f"Input CSV: {args.input_file}")

# Load CSV file
df = pd.read_csv(f'data/pipeline/{args.input_file}')
print(f'df shape: {df.shape}')

# # Initialize CrossEncoder model
model = CrossEncoder('dragonkue/bge-reranker-v2-m3-ko', default_activation_function=torch.nn.Sigmoid())

# Function to rank and sort contexts based on model's score
def rank_contexts(row):
    question = row['question']
    
    # Collect all top-k contexts (in this case, top-100)
    contexts = [row[f'top{i}_context'] for i in range(1, 46)]
    
    # Prepare pairs of question and contexts
    input_pairs = [[question, context] for context in contexts]
    
    # Predict scores for each pair
    scores = model.predict(input_pairs)
    
    # Create a list of tuples (context, score) and sort them by score in descending order
    sorted_contexts = sorted(zip(contexts, scores), key=lambda x: x[1], reverse=True)
    
    # Return sorted contexts (only the contexts, not the scores)
    return [context for context, score in sorted_contexts]

# Apply the ranking function to each row in the dataframe
for index, row in tqdm(df.iterrows(), total=len(df), desc="Ranking contexts"):
    with torch.no_grad():  # 그래디언트 저장 방지
        sorted_contexts = rank_contexts(row)
    
    # Save the sorted contexts back into the dataframe
    for i in range(1, 46):
        df.at[index, f'top{i}_context'] = sorted_contexts[i - 1]
        
    torch.cuda.empty_cache()  # GPU 메모리 캐시 해제


# Save the updated dataframe back to a new CSV file
output_path = f'data/pipeline/{args.output_file}'  # 정렬된 결과를 저장할 경로
df.to_csv(output_path, index=False)

print(f"Sorted CSV saved to {output_path}")
