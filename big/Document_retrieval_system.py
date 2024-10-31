import os
import pickle
from collections import defaultdict
from math import log
import pandas as pd
import csv
from preprocess import Preprocessor
import time
from scipy.sparse import csr_matrix, save_npz, load_npz, vstack
from sklearn.preprocessing import normalize as sparse_normalize
from sklearn.metrics.pairwise import cosine_similarity as sparse_cosine_similarity

# Directory, file extension, and configuration
directory = './full_docs'
extension = '.txt'
k = 10
BATCH_SIZE = 100  # Number of files per batch

def save_checkpoint(data, filename, mode="ab"):
    with open(filename, mode) as f:
        pickle.dump(data, f)

def load_checkpoint(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

def main():
    start_time = time.time()
    print("Program started")
    preprocessor = Preprocessor()
    file_count = len([file for file in os.listdir(directory) if file.endswith(extension)])
    
    # Document Map Checkpointing
    document_map = load_checkpoint('document_map.pkl') or {}
    processed_docs = set(document_map.keys())
    all_docs = [file for file in os.listdir(directory) if file.endswith(extension) and file not in processed_docs]
    print("Processing document map in batches...")
    for i in range(0, len(all_docs), BATCH_SIZE):
        batch_docs = all_docs[i:i + BATCH_SIZE]
        for file in batch_docs:
            with open(os.path.join(directory, file), 'r', encoding='utf-8') as doc:
                text = doc.read()
                document_map[file] = preprocessor.preprocess(text)
        save_checkpoint(document_map, 'document_map.pkl', mode="wb")  # Overwrite with each batch
        document_map.clear()  # Free memory after each batch
    document_map = load_checkpoint('document_map.pkl')  # Reload entire map if needed later
    
    # Inverted Index Checkpointing
    inverted_index_docs = load_checkpoint('inverted_index_docs.pkl') or defaultdict(list)
    print("Building inverted index in batches...")
    for file, wordlist in document_map.items():
        for word in wordlist:
            inverted_index_docs[word].append(file)
    save_checkpoint(inverted_index_docs, 'inverted_index_docs.pkl', mode="wb")
    
    # Document TF-IDF Sparse Matrix Checkpointing
    doc_terms, row, col, data = [], [], [], []
    doc_index = {doc_id: idx for idx, doc_id in enumerate(document_map)}
    tf_idf_filename = "document_tf_idf.npz"
    if os.path.exists(tf_idf_filename):
        doc_tf_idf_matrix = load_npz(tf_idf_filename)
    else:
        for doc_id, tokens in document_map.items():
            doc_id_idx = doc_index[doc_id]
            tf = defaultdict(int)
            for word in tokens:
                tf[word] += 1
            for word, freq in tf.items():
                tf_value = 1 + log(freq)
                df = len(inverted_index_docs[word])
                idf = log(file_count / df)
                term_weight = tf_value * idf
                row.append(doc_id_idx)
                if word not in doc_terms:
                    doc_terms.append(word)
                col.append(doc_terms.index(word))
                data.append(term_weight)
            # Save batch matrix
            if (doc_id_idx + 1) % BATCH_SIZE == 0:
                batch_matrix = csr_matrix((data, (row, col)), shape=(BATCH_SIZE, len(doc_terms)))
                batch_matrix = sparse_normalize(batch_matrix, norm='l2', axis=1)
                if 'doc_tf_idf_matrix' in locals():
                    doc_tf_idf_matrix = vstack([doc_tf_idf_matrix, batch_matrix])
                else:
                    doc_tf_idf_matrix = batch_matrix
                row, col, data = [], [], []  # Clear data after each batch
                save_npz(tf_idf_filename, doc_tf_idf_matrix)
    
    # Load or preprocess queries
    queries = load_checkpoint('queries.pkl')
    if queries is None:
        print("Loading and preprocessing queries...")
        file = pd.read_csv('dev_queries.tsv', sep="\t")
        query_numbers = file['Query number'].tolist()
        queries = [preprocessor.preprocess(query) for query in file['Query'].tolist()]
        save_checkpoint((query_numbers, queries), 'queries.pkl')
    else:
        query_numbers, queries = queries

    # Build Query TF-IDF Sparse Matrix
    query_terms, q_row, q_col, q_data = doc_terms, [], [], []
    query_index = {query_num: idx for idx, query_num in enumerate(query_numbers)}
    query_tf_idf_filename = "query_tf_idf.npz"
    if os.path.exists(query_tf_idf_filename):
        query_tf_idf_matrix = load_npz(query_tf_idf_filename)
    else:
        for query_num, query in zip(query_numbers, queries):
            q_id_idx = query_index[query_num]
            tf = defaultdict(int)
            for word in query:
                tf[word] += 1
            for word, freq in tf.items():
                tf_value = 1 + log(freq)
                if word in inverted_index_docs:
                    df = len(inverted_index_docs[word])
                    idf = log(file_count / df)
                    term_weight = tf_value * idf
                    q_row.append(q_id_idx)
                    q_col.append(query_terms.index(word))
                    q_data.append(term_weight)
        query_tf_idf_matrix = csr_matrix((q_data, (q_row, q_col)), shape=(len(query_numbers), len(query_terms)))
        query_tf_idf_matrix = sparse_normalize(query_tf_idf_matrix, norm='l2', axis=1)
        save_npz(query_tf_idf_filename, query_tf_idf_matrix)
    
    # Similarity Calculation and Results Generation
    similarity_matrix = sparse_cosine_similarity(query_tf_idf_matrix, doc_tf_idf_matrix)
    results = {query_numbers[i]: sorted(
        [(list(document_map.keys())[j], similarity_matrix[i, j]) for j in range(similarity_matrix.shape[1])],
        key=lambda x: x[1], reverse=True
    )[:10] for i in range(similarity_matrix.shape[0])}
    
    # Save Top Results to CSV
    with open("results.csv", mode="w", newline="") as file:
        fieldnames = ["Query_number", "Doc_number"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for query_num, docs in results.items():
            for doc_id, _ in docs:
                writer.writerow({"Query_number": query_num, "Doc_number": doc_id})

    # MAP@K and MAR@K Calculations
    relevant_docs = {}
    with open("dev_query_results.csv", newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            relevant_docs[row[0]] = row[1]

    total_average_precision = 0
    total_average_recall = 0
    for query in results.keys():
        APK = 0
        ARK = 0
        for i in range(min(k, len(results[query]))):  # Limit to top k results
            doc_id = results[query][i][0].split('.')[0]
            doc_id = doc_id.split('_')[1]
            if str(doc_id) == relevant_docs[str(query)]:
                APK = 1.0 / (i + 1)
                ARK = 1  # Recall is 1 if we find relevant doc within top k
                break  # Stop after finding the first relevant document
        total_average_precision += APK
        total_average_recall += ARK

    MAPK = total_average_precision / len(query_numbers)
    MARK = total_average_recall / len(query_numbers)

    print(f"MAP@{k}: {MAPK}")
    print(f"MAR@{k}: {MARK}")
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Program completed. Total time taken: {duration:.2f} seconds.")

if __name__ == "__main__":
    main()
