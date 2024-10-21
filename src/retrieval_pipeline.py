import argparse
import sys
import json
from datasets import load_from_disk


def re_rank_pipeline(query, sparse_top_k=100, dense_top_k=3):
    # Step 1: Perform Sparse Retrieval
    sparse_output = sparse_retrieval(query, sparse_top_k)
    
    # Step 2: Re-rank using Dense Retrieval
    dense_output = dense_retrieval(query, dense_top_k, sparse_output)
    
    return dense_output


def weighted_retrieval_pipeline(query, top_k=5, sparse_weight=0.5, dense_weight=0.5):
    # Step 1: Perform Sparse Retrieval
    sparse_output = sparse_retrieval(query, top_k)
    
    # Step 2: Perform Dense Retrieval
    dense_output = dense_retrieval(query, top_k)
    
    # Step 3: Combine scores with weighted sum
    combined_scores = (sparse_weight * sparse_output["sparse_scores"]) + (dense_weight * dense_output["dense_scores"])
    
    # Step 4: Rank documents based on combined scores
    ranked_indices = np.argsort(combined_scores)[::-1]  # Sort by score (descending)
    ranked_docs = [sparse_output["document_ids"][i] for i in ranked_indices]
    
    return {"id": query["id"], "question": query["question"], "document_ids": ranked_docs}


if __name__ == "__main__":
    # set arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--re_rank", type=bool, default=False, help="Whether to perform re-ranking")
    parser.add_argument("--weighted", type=bool, default=False, help="Whether to perform weighted retrieval")

    parser.add_argument("--sparse_top_k", type=int, default=100, help="Number of documents to retrieve in sparse retrieval")
    parser.add_argument("--dense_top_k", type=int, default=3, help="Number of documents to retrieve in dense retrieval")

    parser.add_argument("--sparse_weight", type=float, default=0.5, help="Weight for sparse retrieval scores")
    parser.add_argument("--dense_weight", type=float, default=0.5, help="Weight for dense retrieval scores")
    parser.add_argument("--top_k", type=int, default=5, help="Number of documents to return in the final output")
    args = parser.parse_args()
    
    dataset = load_from_disk("data/raw/test_dataset/")

    
    if args.re_rank:
        output = re_rank_pipeline(dataset, args.sparse_top_k, args.dense_top_k)
    elif args.weighted:
        output = weighted_retrieval_pipeline(dataset, args.top_k, args.sparse_weight, args.dense_weight)
    else:
        print()
        print("***Please specify either re_rank or weighted as True***")
        print()
        sys.exit(1)

    # 어디에 하는거임?    
    # # Save output
    # output_path = "data/output.json"
    # with open(output_path, "w") as f:
    #     json.dump(output, f)