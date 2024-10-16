import json
import pandas as pd
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
import torch


'''data'''
# data = json.load(open('../data/raw/wikipedia_documents.json'))
# wiki = pd.DataFrame(data).T

dataset = load_from_disk("../data/raw/train_dataset/")
train_df = pd.DataFrame(dataset['train'])
valid_df = pd.DataFrame(dataset['validation'])
mrc = pd.concat([train_df, valid_df])



'''inference'''
model = SentenceTransformer("intfloat/e5-mistral-7b-instruct")

# cache the model on the GPU to reduce GPU memory usage
torch.cuda.empty_cache()

# In case you want to reduce the maximum sequence length:
model.max_seq_length = 4096

# queries = [
#     "how much protein should a female eat",
#     "summit define",
# ]
# documents = [
#     "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
#     "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
# ]
queries = []
documents = []

x = 5

for i in range(x):
    queries.append(mrc['question'].iloc[i])
    documents.append(mrc['context'].iloc[i])

query_embeddings = model.encode(queries, prompt_name="web_search_query")
document_embeddings = model.encode(documents)

scores = (query_embeddings @ document_embeddings.T) * 100

for i in range(x):
    print(f"Query: {queries[i]}")
    for j in range(x):
        print(f"Document {j}: {documents[j]}")
        print(f"Score: {scores[i][j]:.2f}")
        print()
