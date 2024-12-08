import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_data(directory_path, query_file, ground_truth_file, max_queries=1000):
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
    query_df = pd.read_excel(query_file, header=None, names=["QueryNumber", "Query"])
    queries = query_df['Query'].tolist()[:max_queries]
    query_numbers = query_df['QueryNumber'].tolist()[:max_queries]

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


def build_inverted_index(embeddings, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_
    return labels, centroids, kmeans


def search_with_inverted_index(query_embeddings, centroids, kmeans, doc_embeddings, file_names, top_k_clusters, top_docs):
    results = {}
    for idx, query_vec in enumerate(tqdm(query_embeddings)):
        # Compute distances to centroids
        cluster_distances = cdist([query_vec], centroids, metric="cosine")[0]
        top_clusters = np.argsort(cluster_distances)[:top_k_clusters]

        # Collect documents from relevant clusters
        relevant_docs = []
        for cluster in top_clusters:
            cluster_docs = np.where(kmeans.labels_ == cluster)[0]
            for doc_idx in cluster_docs:
                sim = cosine_similarity([query_vec], [doc_embeddings[doc_idx]])[0, 0]
                relevant_docs.append((file_names[doc_idx], sim))

        # Rank documents and take top_docs
        ranked_docs = sorted(relevant_docs, key=lambda x: x[1], reverse=True)[:top_docs]
        results[idx] = ranked_docs
    return results


def evaluate(results, relevant_docs, k_values):
    precision_at_k = {k: [] for k in k_values}
    recall_at_k = {k: [] for k in k_values}

    for query_idx, retrieved_docs in results.items():
        retrieved_doc_ids = [doc[0].split('_')[-1].replace('.txt', '') for doc in retrieved_docs]
        true_relevant = relevant_docs.get(query_idx + 1, [])  # query_idx is 0-based

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
    directory_path = "../full_docs_small"
    query_file = "dev_small_queries.xlsx"
    ground_truth_file = "dev_query_results_small.csv"
    embedding_file = "document_embeddings_inverted.npy"

    # Parameters
    num_clusters = 10
    top_k_clusters = 3
    top_docs = 10
    k_values = [1, 3, 5, 10]

    # Load data
    documents, file_names, queries, query_numbers, relevant_docs = load_data(
        directory_path, query_file, ground_truth_file, max_queries=1000
    )

    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Compute document embeddings in batches
    document_embeddings = compute_embeddings(documents, model, embedding_file)

    # Build inverted index
    labels, centroids, kmeans = build_inverted_index(document_embeddings, num_clusters)

    # Compute query embeddings
    query_embeddings = model.encode(queries, convert_to_tensor=True).cpu().numpy()

    # Perform search
    results = search_with_inverted_index(
        query_embeddings, centroids, kmeans, document_embeddings, file_names, top_k_clusters, top_docs
    )

    # Evaluate results
    mean_precision, mean_recall = evaluate(results, relevant_docs, k_values)

    # Print results
    print("\nMean Precision@k:")
    for k in k_values:
        print(f"  Precision@{k}: {mean_precision.get(k, 0):.4f}")
    print("\nMean Recall@k:")
    for k in k_values:
        print(f"  Recall@{k}: {mean_recall.get(k, 0):.4f}")


if __name__ == "__main__":
    main()
