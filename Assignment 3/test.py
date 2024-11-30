import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from preprocess import Preprocessor

def calculate_precision_recall(retrieved_docs, relevant_docs, k):
    retrieved_top_k = retrieved_docs[:k]  # Top-k retrieved documents
    relevant_retrieved = set(retrieved_top_k).intersection(set(relevant_docs))  # Intersection with relevant docs

    precision_at_k = len(relevant_retrieved) / k if k > 0 else 0
    recall_at_k = len(relevant_retrieved) / len(relevant_docs) if len(relevant_docs) > 0 else 0
    
    return precision_at_k, recall_at_k

def main():
    top_k = [1, 3, 5, 10]
    directory_path = "../full_docs_small"
    documents = []
    file_names = []
    preprocessor = Preprocessor()
    # Read documents from the directory
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".txt"):  # Ensure you only read text files
            file_path = os.path.join(directory_path, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # content = preprocessor.preprocess(content)
                documents.append(content)
                file_names.append(file_name)  # Use filenames as identifiers

    # Check if documents were loaded correctly
    print(f"Loaded {len(documents)} documents.")

    # Load the SentenceTransformer model
    model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
    document_embeddings = model.encode(documents, convert_to_tensor=True).cpu().numpy()

    # Read queries and their numbers
    file = pd.read_excel('dev_small_queries.xlsx')
    query_numbers = file['Query number'].tolist()
    # queries = [preprocessor.preprocess(query) for query in file['Query'].tolist()]]
    queries = file['Query'].tolist()
    
    # Load ground truth and map queries to relevant documents
    ground_truth_df = pd.read_csv('dev_query_results_small.csv')
    relevant_docs = {query_number: group['doc_number'].tolist() for query_number, group in ground_truth_df.groupby('Query_number')}

    # Validate the mapping
    print(f"Ground truth contains {len(relevant_docs)} queries.")

    # Encode query embeddings
    query_embeddings = model.encode(queries, convert_to_tensor=True).cpu().numpy()

    # Initialize precision and recall metrics
    precision_at_k = {k: [] for k in top_k}
    recall_at_k = {k: [] for k in top_k}

    # Compute cosine similarities for each query
    similarities = cosine_similarity(query_embeddings, document_embeddings)

    # Normalize file names to match ground truth identifiers
    file_ids = [file_name.split('_')[-1].replace('.txt', '') for file_name in file_names]

    # Convert relevant_doc_ids to strings
    relevant_docs = {query_number: list(map(str, group['doc_number'].tolist())) 
                    for query_number, group in ground_truth_df.groupby('Query_number')}

    # Process each query
    for query_number, query_similarities in zip(query_numbers, similarities):
        # Rank documents based on similarity (high to low)
        sorted_doc_indices = np.argsort(-query_similarities)  # Sort indices by descending similarity
        ranked_docs = [file_ids[idx] for idx in sorted_doc_indices]  # Map indices to numeric identifiers (strings)

        # Get relevant documents for this query
        relevant_doc_ids = relevant_docs.get(query_number, [])  # Default to empty list if no ground truth

        # Debugging outputs
        print(f"Query {query_number}:")
        print(f"  Relevant docs: {relevant_doc_ids}")
        print(f"  Top-10 retrieved docs: {ranked_docs[:10]}")

        # Debug intersection
        for k in top_k:
            print(f"Top-{k} Retrieved Docs: {ranked_docs[:k]}")
            print(f"Relevant Docs: {relevant_doc_ids}")
            print(f"Intersection: {set(ranked_docs[:k]).intersection(set(relevant_doc_ids))}")

            precision, recall = calculate_precision_recall(ranked_docs, relevant_doc_ids, k)
            precision_at_k[k].append(precision)
            recall_at_k[k].append(recall)


    # Compute mean Precision@k and Recall@k across all queries
    mean_precision_at_k = {k: np.mean(precision_at_k[k]) for k in top_k}
    mean_recall_at_k = {k: np.mean(recall_at_k[k]) for k in top_k}

    # Print results
    print("\nMean Precision@k:")
    for k in top_k:
        print(f"  Precision@{k}: {mean_precision_at_k[k]:.4f}")
    print("\nMean Recall@k:")
    for k in top_k:
        print(f"  Recall@{k}: {mean_recall_at_k[k]:.4f}")

if __name__ == "__main__":
    main()
