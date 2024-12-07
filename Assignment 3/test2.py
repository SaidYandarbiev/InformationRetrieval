import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
import numpy as np
from tqdm import tqdm

# Load embeddings model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Constants
DOC_DIR = "../full_docs_small"
QUERY_FILE = "dev_small_queries.xlsx"
GROUND_TRUTH_FILE = "dev_query_results_small.csv"
NUM_CLUSTERS = 10  # Adjust based on experimentation
TOP_K_CLUSTERS = 3
TOP_DOCS = 10
MAX_QUERIES = 1000  # Limit to the first 1000 queries

# Step 1: Process documents
def load_documents(doc_dir):
    doc_embeddings = []
    doc_ids = []
    for file in tqdm(os.listdir(doc_dir)):
        if file.endswith(".txt"):
            with open(os.path.join(doc_dir, file), 'r', encoding='utf-8') as f:
                text = f.read()
                # Encode and move to CPU for compatibility with NumPy
                embedding = model.encode(text, convert_to_tensor=True).cpu().numpy()
                doc_embeddings.append(embedding)
                doc_ids.append(file)
    return np.vstack(doc_embeddings), doc_ids

# Step 2: Build inverted vector index
def build_index(doc_embeddings, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(doc_embeddings)
    centroids = kmeans.cluster_centers_
    return labels, centroids, kmeans

# Step 3: Process queries
def load_queries(query_file, max_queries):
    queries = pd.read_excel(query_file, header=None, names=["QueryNumber", "Query"])
    # queries = pd.read_csv(query_file, sep="\t", header=None, names=["QueryNumber", "Query"])
    queries = queries.head(max_queries)  # Limit to the first 1000 queries
    query_embeddings = [
        model.encode(q, convert_to_tensor=True).cpu().numpy() for q in queries["Query"]
    ]
    return queries, np.vstack(query_embeddings)

# Step 4: Search
def search_query(query_vector, centroids, kmeans, doc_embeddings, doc_ids, top_k_clusters, top_docs):
    cluster_distances = cdist([query_vector], centroids, metric="cosine")[0]
    top_clusters = np.argsort(cluster_distances)[:top_k_clusters]
    
    relevant_docs = []
    for cluster in top_clusters:
        cluster_docs = np.where(kmeans.labels_ == cluster)[0]
        for doc_idx in cluster_docs:
            sim = cosine_similarity([query_vector], [doc_embeddings[doc_idx]])[0, 0]
            relevant_docs.append((doc_ids[doc_idx], sim))
    
    # Sort by similarity and take top_docs
    ranked_docs = sorted(relevant_docs, key=lambda x: x[1], reverse=True)[:top_docs]
    return ranked_docs

# Step 5: Evaluate
def evaluate(results, ground_truth_path, k_values):
    ground_truth = pd.read_csv(ground_truth_path)
    precision_at_k = {k: [] for k in k_values}
    recall_at_k = {k: [] for k in k_values}

    for query_id, retrieved_docs in results.items():
        # Retrieve relevant docs for this query
        relevant_docs = ground_truth[ground_truth["Query_number"] == query_id]["doc_number"].tolist()
        # Normalize ground truth IDs to match results format
        normalized_relevant_docs = [f"output_{doc}.txt" for doc in relevant_docs]

        print(f"Query ID: {query_id}, Normalized Relevant Docs: {normalized_relevant_docs}")

        for k in k_values:
            # Take top-k retrieved documents
            retrieved_k = [doc[0] for doc in retrieved_docs[:k]]
            true_positive = len(set(retrieved_k) & set(normalized_relevant_docs))
            precision = true_positive / k
            recall = true_positive / len(normalized_relevant_docs) if normalized_relevant_docs else 0
            precision_at_k[k].append(precision)
            recall_at_k[k].append(recall)

    # Compute mean precision and recall
    mean_precision = {k: np.mean(precision_at_k[k]) for k in k_values if precision_at_k[k]}
    mean_recall = {k: np.mean(recall_at_k[k]) for k in k_values if recall_at_k[k]}
    return mean_precision, mean_recall


# Main pipeline
if __name__ == "__main__":
    print("Loading documents...")
    doc_embeddings, doc_ids = load_documents(DOC_DIR)
    
    print("Building inverted index...")
    labels, centroids, kmeans = build_index(doc_embeddings, NUM_CLUSTERS)
    
    print("Loading queries...")
    queries, query_embeddings = load_queries(QUERY_FILE, MAX_QUERIES)
    
    print("Performing search...")
    results = {}
    for idx, query_vec in enumerate(tqdm(query_embeddings)):
        query_id = queries["QueryNumber"][idx]
        results[query_id] = search_query(query_vec, centroids, kmeans, doc_embeddings, doc_ids, TOP_K_CLUSTERS, TOP_DOCS)
    print("Evaluating...")
    precision, recall = evaluate(results, GROUND_TRUTH_FILE, [1, 3, 5, 10])
    print(f"Precision@k: {precision}")
    print(f"Recall@k: {recall}")