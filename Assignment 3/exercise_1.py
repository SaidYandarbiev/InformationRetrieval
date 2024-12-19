import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from torch.utils.data import DataLoader


def calculate_precision_recall(retrieved_docs, relevant_docs, k):
    retrieved_top_k = retrieved_docs[:k]
    relevant_retrieved = set(retrieved_top_k).intersection(set(relevant_docs))

    precision_at_k = len(relevant_retrieved) / k if k > 0 else 0
    recall_at_k = len(relevant_retrieved) / len(relevant_docs) if len(relevant_docs) > 0 else 0

    return precision_at_k, recall_at_k


def load_data(directory_path, query_file, ground_truth_file):
    # Load documents
    documents = []
    file_names = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(directory_path, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append(f.read())
                file_names.append(file_name)

    # Load queries
    query_df = pd.read_excel(query_file)
    queries = query_df['Query'].tolist()
    query_numbers = query_df['Query number'].tolist()

    # Load ground truth
    ground_truth_df = pd.read_csv(ground_truth_file)
    relevant_docs = {
        query_number: group['doc_number'].astype(str).tolist()
        for query_number, group in ground_truth_df.groupby('Query_number')
    }

    return documents, file_names, queries, query_numbers, relevant_docs


def compute_embeddings(documents, model, embedding_file, batch_size=512):
    # Load or compute document embeddings
    if os.path.exists(embedding_file):
        return np.load(embedding_file)

    document_loader = DataLoader(documents, batch_size=batch_size, shuffle=False)
    embeddings = []
    for batch in document_loader:
        batch_embeddings = model.encode(batch, convert_to_tensor=True).cpu().numpy()
        embeddings.extend(batch_embeddings)

    embeddings = np.array(embeddings)
    np.save(embedding_file, embeddings)
    return embeddings

def main():
    # File paths
    directory_path = "../full_docs_small"
    query_file = 'dev_small_queries.xlsx'
    ground_truth_file = 'dev_query_results_small.csv'
    embedding_file = "document_embeddings.npy"

    # Parameters
    top_k = [1, 3, 5, 10]

    # Load data
    documents, file_names, queries, query_numbers, relevant_docs = load_data(
        directory_path, query_file, ground_truth_file
    )

    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Compute document embeddings
    document_embeddings = compute_embeddings(documents, model, embedding_file)

    # Compute query embeddings
    query_embeddings = model.encode(queries, convert_to_tensor=True).cpu().numpy()

    # Compute cosine similarities
    similarities = cosine_similarity(query_embeddings, document_embeddings)

    # Normalize file names to match ground truth identifiers
    file_ids = [file_name.split('_')[-1].replace('.txt', '') for file_name in file_names]

    # Precision and Recall Metrics
    precision_at_k = {k: [] for k in top_k}
    recall_at_k = {k: [] for k in top_k}

    # Process each query
    for query_number, query_similarities in zip(query_numbers, similarities):
        # Rank documents based on similarity
        sorted_doc_indices = np.argsort(-query_similarities)
        ranked_docs = [file_ids[idx] for idx in sorted_doc_indices]

        # Relevant documents for this query
        relevant_doc_ids = relevant_docs.get(query_number, [])

        # Compute precision and recall
        for k in top_k:
            precision, recall = calculate_precision_recall(ranked_docs, relevant_doc_ids, k)
            precision_at_k[k].append(precision)
            recall_at_k[k].append(recall)

    # Compute Mean Precision and Recall
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
