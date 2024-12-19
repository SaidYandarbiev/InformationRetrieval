import csv
import os
import pandas as pd
import numpy as np
import torch

from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm


# def load_data(directory_path, query_file, ground_truth_file, max_queries=1000):
def load_data(directory_path, query_file):

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
    # query_df = pd.read_excel(query_file, names=["QueryNumber", "Query"])
    query_df = pd.read_csv(query_file, names=["QueryNumber", "Query"], header=0, sep="\t")
    queries = query_df['Query'].tolist()
    query_numbers = query_df['QueryNumber'].tolist()

    # Load ground truth
    # ground_truth_df = pd.read_csv(ground_truth_file)
    # relevant_docs = {
    #     query_number: group['doc_number'].astype(str).tolist()
    #     for query_number, group in ground_truth_df.groupby('Query_number')
    # }

    # return documents, file_names, queries, query_numbers, relevant_docs
    # We don't want to make use of the ground truth file for test set of queries since this is not provided
    return documents, file_names, queries, query_numbers


# Compute embeddings for documents, handling documents longer than the model's max sequence length.
def compute_embeddings_with_chunks(documents, model, embedding_file, max_seq_length=512, overlap=50, batch_size=512):
    if os.path.exists(embedding_file):
        document_embeddings = np.load(embedding_file)
        # Reshape to ensure 2D array
        if document_embeddings.ndim == 3 and document_embeddings.shape[1] == 1:
            document_embeddings = document_embeddings.squeeze(1)
        print("Loaded precomputed embeddings:", document_embeddings.shape)
        return document_embeddings

    def chunk_document(doc, max_seq_length, overlap):
        tokens = doc.split()  # Tokenize document by whitespace
        chunks = []
        for i in range(0, len(tokens), max_seq_length - overlap):
            chunk = tokens[i:i + max_seq_length]
            chunks.append(" ".join(chunk))
        return chunks

    # Prepare chunks
    all_chunks = []
    doc_chunk_mapping = []  # To keep track of which chunks belong to which documents
    print("Preparing document chunks...")
    for doc_idx, doc in enumerate(tqdm(documents, desc="Chunking documents")):
        chunks = chunk_document(doc, max_seq_length, overlap)
        all_chunks.extend(chunks)
        doc_chunk_mapping.append(len(chunks))  # Number of chunks per document

    # Encode chunks in batches
    print("Total number of chunks to encode:", len(all_chunks))
    chunk_loader = DataLoader(all_chunks, batch_size=batch_size, shuffle=False)
    chunk_embeddings = []
    for batch in tqdm(chunk_loader, desc="Encoding chunks"):
        batch_embeddings = model.encode(batch, convert_to_tensor=True).cpu().numpy()
        chunk_embeddings.extend(batch_embeddings)

    # Aggregate embeddings for each document
    document_embeddings = []
    start_idx = 0
    print("Aggregating chunk embeddings into document embeddings...")
    for num_chunks in tqdm(doc_chunk_mapping, desc="Aggregating embeddings"):
        if num_chunks == 0:  # Handle empty documents
            aggregated_embedding = np.zeros(model.get_sentence_embedding_dimension())
        else:
            doc_chunks = chunk_embeddings[start_idx:start_idx + num_chunks]
            aggregated_embedding = np.mean(doc_chunks, axis=0) if len(doc_chunks) > 0 else np.zeros(model.get_sentence_embedding_dimension())
        document_embeddings.append(aggregated_embedding)
        start_idx += num_chunks

    # Ensure proper shape and save
    document_embeddings = np.array(document_embeddings)
    print("Shape of document embeddings after aggregation:", document_embeddings.shape)
    assert document_embeddings.ndim == 2, "Final document embeddings must be a 2D array (num_docs, embedding_size)"
    np.save(embedding_file, document_embeddings)
    return document_embeddings


def build_inverted_index(embeddings, num_clusters):
    # Validate that embeddings are 2D
    assert embeddings.ndim == 2, "Embeddings must be a 2D array with shape (num_docs, embedding_size)"
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_
    return labels, centroids, kmeans


# Optimized function to search queries using GPU for cosine similarity.
def search_with_inverted_index(
    query_embeddings: np.ndarray, 
    centroids: np.ndarray, 
    kmeans, 
    doc_embeddings: np.ndarray, 
    file_names: list, 
    top_k_clusters: int, 
    top_docs: int
):
    # Move centroids and document embeddings to GPU
    centroids_gpu = torch.tensor(centroids, dtype=torch.float32).cuda()
    doc_embeddings_gpu = torch.tensor(doc_embeddings, dtype=torch.float32).cuda()

    results = {}
    print("Searching queries using inverted index...")
    for idx, query_vec in enumerate(tqdm(query_embeddings, desc="Processing queries")):
        query_vec_gpu = torch.tensor(query_vec, dtype=torch.float32).unsqueeze(0).cuda()

        # Compute distances to centroids
        cluster_distances = torch.nn.functional.cosine_similarity(query_vec_gpu, centroids_gpu, dim=1)
        top_clusters = torch.topk(cluster_distances, top_k_clusters, largest=True).indices

        # Collect documents from relevant clusters
        relevant_docs = []
        for cluster in top_clusters:
            cluster_docs = torch.where(torch.tensor(kmeans.labels_ == cluster.item()))[0].cpu().numpy()
            cluster_doc_embeddings = doc_embeddings_gpu[cluster_docs]
            similarities = torch.nn.functional.cosine_similarity(query_vec_gpu, cluster_doc_embeddings, dim=1)
            relevant_docs.extend(
                [(file_names[doc_idx], sim.item()) for doc_idx, sim in zip(cluster_docs, similarities)]
            )

        # Rank documents and take top_docs
        ranked_docs = sorted(relevant_docs, key=lambda x: x[1], reverse=True)[:top_docs]
        results[idx] = ranked_docs

    return results


def evaluate(results, relevant_docs, query_numbers, k_values):
    precision_at_k = {k: [] for k in k_values}
    recall_at_k = {k: [] for k in k_values}
    for query_idx, retrieved_docs in results.items():
        query_number = query_numbers[query_idx]  # Map index to query number
        retrieved_doc_ids = [doc[0].split('_')[-1].replace('.txt', '') for doc in retrieved_docs]
        true_relevant = relevant_docs.get(query_number, [])  # Use actual query number for lookup
        
        for k in k_values:
            top_k_docs = retrieved_doc_ids[:k]
            true_positive = len(set(top_k_docs) & set(true_relevant))
            precision = true_positive / k if k > 0 else 0
            recall = true_positive / len(true_relevant) if true_relevant else 0

            precision_at_k[k].append(precision)
            recall_at_k[k].append(recall)

    mean_precision = {k: np.mean(precision_at_k[k]) for k in k_values if precision_at_k[k]}
    mean_recall = {k: np.mean(recall_at_k[k]) for k in k_values if recall_at_k[k]}
    return mean_precision, mean_recall


def main():
    # File paths

    # directory_path = "../full_docs_small"
    # query_file = "dev_small_queries.xlsx"
    # ground_truth_file = "dev_query_results_small.csv"
    # embedding_file = "document_embeddings_inverted_klein.npy"

    # directory_path = "../full_docs_medium"

    directory_path = "../full_docs"
    query_file = "dev_queries.tsv"
    # ground_truth_file = "dev_query_results.csv"
    embedding_file = "document_embeddings_inverted.npy"

    # Parameters
    num_clusters = 1000
    top_k_clusters = 10
    top_docs = 10
    k_values = [1, 3, 5, 10]

    # Load data
    # documents, file_names, queries, query_numbers, relevant_docs = load_data(
    #     directory_path, query_file, ground_truth_file, max_queries=1000
    # )

    # Load data without max queries (for test set of queries)
    documents, file_names, queries, query_numbers = load_data(
        directory_path, query_file
    )

    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Compute document embeddings with sequence length extension
    document_embeddings = compute_embeddings_with_chunks(documents, model, embedding_file)

    # Build inverted index
    labels, centroids, kmeans = build_inverted_index(document_embeddings, num_clusters)

    # Compute query embeddings
    query_embeddings = model.encode(queries, convert_to_tensor=True).cpu().numpy()

    # Perform search
    results = search_with_inverted_index(
        query_embeddings, centroids, kmeans, document_embeddings, file_names, top_k_clusters, top_docs
    )

    # Evaluate results
    # mean_precision, mean_recall = evaluate(results, relevant_docs, query_numbers, k_values)

    # Save top results to CSV
    with open("result.csv", mode="w", newline="") as file:
            fieldnames = ["Query_number", "Doc_number"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for query_idx, docs in results.items():
                # Correct mapping starts from the second query number
                query_number = query_numbers[query_idx]
                for doc_id in docs:
                    writer.writerow({"Query_number": query_number, "Doc_number": doc_id})

    # Print results
    # print("\nMean Precision@k:")
    # for k in k_values:
    #     print(f"  Precision@{k}: {mean_precision.get(k, 0):.4f}")
    # print("\nMean Recall@k:")
    # for k in k_values:
    #     print(f"  Recall@{k}: {mean_recall.get(k, 0):.4f}")


if __name__ == "__main__":
    main()
