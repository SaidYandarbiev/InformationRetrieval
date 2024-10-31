import os
import pickle
from collections import defaultdict
from math import log
import pandas as pd
import csv
from preprocess import Preprocessor
import time
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.preprocessing import normalize as sparse_normalize
from sklearn.metrics.pairwise import cosine_similarity as sparse_cosine_similarity

directory = './full_docs_small'
extension = '.txt'
k = 10

def load_preprocessed_data(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

def save_preprocessed_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def main():
    start_time = time.time()
    print(f"Program started")
    preprocessor = Preprocessor()

    # Load or preprocess documents
    document_map = load_preprocessed_data('document_map.pkl')
    if document_map is None:
        print("No document map?")
        document_map = {}
        for file in os.listdir(directory):
            if file.endswith(extension):
                with open(os.path.join(directory, file), 'r', encoding='utf-8') as doc:
                    text = doc.read()
                    document_map[file] = preprocessor.preprocess(text)
        save_preprocessed_data(document_map, 'document_map.pkl')

    file_count = len([file for file in os.listdir(directory) if file.endswith(extension)])
    inverted_index_docs = load_preprocessed_data('inverted_index_docs.pkl')
    if inverted_index_docs is None:
        print("No inverted index?")
        inverted_index_docs = defaultdict(list)
        for file, wordlist in document_map.items():
            for word in wordlist:
                if file not in inverted_index_docs[word]:
                    inverted_index_docs[word].append(file)
        save_preprocessed_data(inverted_index_docs, 'inverted_index_docs.pkl')

    # Load or preprocess queries
    queries = load_preprocessed_data('queries.pkl')
    if queries is None:
        print("No queries?")
        file = pd.read_excel('dev_small_queries.xlsx')
        query_numbers = file['Query number'].tolist()
        queries = [preprocessor.preprocess(query) for query in file['Query'].tolist()]
        save_preprocessed_data((query_numbers, queries), 'queries.pkl')
    else:
        query_numbers, queries = queries

    # Build document TF-IDF sparse matrix
    doc_terms, row, col, data = [], [], [], []
    doc_index = {doc_id: idx for idx, doc_id in enumerate(document_map)}
    if os.path.exists("document_tf_idf.npz"):
        doc_tf_idf_matrix = load_npz("document_tf_idf.npz")
    else:
        for doc_id, tokens in document_map.items():
            doc_id_idx = doc_index[doc_id]
            tf = {}
            for word in tokens:
                tf[word] = tf.get(word, 0) + 1
            for word, freq in tf.items():
                tf_value = 1 + log(freq)
                df = len(inverted_index_docs[word])
                idf = log(file_count / df)
                term_weight = tf_value * idf
                row.append(doc_id_idx)
                col.append(doc_terms.index(word) if word in doc_terms else len(doc_terms))
                if word not in doc_terms:
                    doc_terms.append(word)
                data.append(term_weight)
        doc_tf_idf_matrix = csr_matrix((data, (row, col)), shape=(len(document_map), len(doc_terms)))
        doc_tf_idf_matrix = sparse_normalize(doc_tf_idf_matrix, norm='l2', axis=1)
        save_npz("document_tf_idf.npz", doc_tf_idf_matrix)

    # Build query TF-IDF sparse matrix
    query_terms, q_row, q_col, q_data = doc_terms, [], [], []
    query_index = {query_num: idx for idx, query_num in enumerate(query_numbers)}
    if os.path.exists("query_tf_idf.npz"):
        query_tf_idf_matrix = load_npz("query_tf_idf.npz")
    else:
        for query_num, query in zip(query_numbers, queries):
            q_id_idx = query_index[query_num]
            tf = {}
            for word in query:
                tf[word] = tf.get(word, 0) + 1
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
        save_npz("query_tf_idf.npz", query_tf_idf_matrix)

    similarity_matrix = sparse_cosine_similarity(query_tf_idf_matrix, doc_tf_idf_matrix)

    # Generate sorted results dictionary
    results = {query_numbers[i]: sorted(
        [(list(document_map.keys())[j], similarity_matrix[i, j]) for j in range(similarity_matrix.shape[1])],
        key=lambda x: x[1], reverse=True
    )[:10] for i in range(similarity_matrix.shape[0])}

    # Save top results to CSV
    with open("results.csv", mode="w", newline="") as file:
        fieldnames = ["Query_number", "Doc_number"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for query_num, docs in results.items():
            for doc_id, _ in docs:
                writer.writerow({"Query_number": query_num, "Doc_number": doc_id})

    # MAPK and MARK calculation
    relevant_docs = {}
    with open("dev_query_results_small.csv", newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            relevant_docs[row[0]] = row[1]

    # Scoring calculations adjusted for MAP@K and MAR@K using k
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
    return 0

if __name__ == "__main__":
    main()
