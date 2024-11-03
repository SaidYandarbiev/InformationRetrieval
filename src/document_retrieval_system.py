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
# Import tqdm for progress bars
from tqdm import tqdm  


directory = './full_docs'
# directory = './full_docs_small'

extension = '.txt'
# Process documents in batches
batch_size = 1000  


def load_preprocessed_data(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None


def save_preprocessed_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def iterate_files_in_batches(directory, batch_size, extension):
    # Generator function to yield batches of preprocessed documents
    batch = {}
    count = 0
    preprocessor = Preprocessor()
    total_files = len([file for file in os.listdir(directory) if file.endswith(extension)])

    for file in tqdm(os.listdir(directory), desc="Processing files in batches", total=total_files):
        if file.endswith(extension):
            with open(os.path.join(directory, file), 'r', encoding='utf-8') as doc:
                text = doc.read()
                batch[file] = preprocessor.preprocess(text)
            count += 1
            if count >= batch_size:
                yield batch
                batch = {}
                count = 0
    if batch:
        # Yield any remaining files in the last batch
        yield batch  


def strip_output_id(output_name):
    # Extract the numeric document ID from the output file name
    return output_name[0].split('_')[1].split('.')[0]


def apk(actual, predicted, k=10):
    # Calculate Average Precision at k (AP@K) for a single query
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k) if actual else 0.0


def ark(actual, predicted, k=10):
    # Calculate Average Recall at k (AR@K) for a single query
    if len(predicted) > k:
        predicted = predicted[:k]

    num_hits = 0
    recall_scores = []

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1
            # Recall calculation
            recall = num_hits / len(actual)  
            recall_scores.append(recall)

    return sum(recall_scores) / len(recall_scores) if recall_scores else 0.0


def mapk(actual, predicted, k=10):
    # Calculate Mean Average Precision at k (MAP@K) across all queries 
    return sum(apk(a, p, k) for a, p in zip(actual, predicted)) / len(actual)


def mark(actual, predicted, k=10):
    # Calculate Mean Average Recall at k (MAR@K) across all queries 
    return sum(ark(a, p, k) for a, p in zip(actual, predicted)) / len(actual)


def inverted_indexing(preprocessor):
    inverted_index_docs = load_preprocessed_data('inverted_index_docs.pkl')
    if inverted_index_docs is None:
        print("No inverted index found, creating a new one.")
        inverted_index_docs = defaultdict(list)
        for file in tqdm(os.listdir(directory), desc="Creating inverted index"):
            if file.endswith(extension):
                with open(os.path.join(directory, file), 'r', encoding='utf-8') as doc:
                    text = doc.read()
                    words = preprocessor.preprocess(text)
                    for word in words:
                        if file not in inverted_index_docs[word]:
                            inverted_index_docs[word].append(file)
        save_preprocessed_data(inverted_index_docs, 'inverted_index_docs.pkl')
    return inverted_index_docs


def query_processing(preprocessor):
    queries = load_preprocessed_data('queries.pkl')
    if queries is None:
        print("No queries found, preprocessing queries.")

        # Uncomment the necessary file you want to use for queries
        # file = pd.read_excel('dev_query_results_small.csv')
        # file = pd.read_csv('dev_queries.tsv', sep="\t")
        file = pd.read_csv('queries.csv', sep="\t")
        query_numbers = file['Query number'].tolist()
        queries = [preprocessor.preprocess(query) for query in file['Query'].tolist()]
        save_preprocessed_data((query_numbers, queries), 'queries.pkl')
    else:
        query_numbers, queries = queries
    return queries, query_numbers


def tf_idf_calculation(inverted_index_docs):
    file_count = len([file for file in os.listdir(directory) if file.endswith(extension)])
    doc_terms = list(inverted_index_docs.keys())
    if os.path.exists("document_tf_idf.npz") and os.path.exists("doc_ids.pkl"):
        doc_tf_idf_matrix = load_npz("document_tf_idf.npz")
        with open("doc_ids.pkl", 'rb') as f:
            # Load doc_ids if it exists
            doc_ids = pickle.load(f)  
    else:
        # Initialize an empty matrix
        doc_tf_idf_matrix = csr_matrix((0, len(doc_terms)))  
        doc_ids = []
        for batch_files in tqdm(iterate_files_in_batches(directory, batch_size, extension),
                                desc="Building TF-IDF matrix"):
            batch_data, batch_row, batch_col = [], [], []

            for local_idx, (doc_id, tokens) in enumerate(batch_files.items()):
                # Append to the global list of doc IDs
                doc_ids.append(doc_id)  
                tf = defaultdict(int)
                for word in tokens:
                    tf[word] += 1
                for word, freq in tf.items():
                    if word in inverted_index_docs:
                        tf_value = 1 + log(freq)
                        df = len(inverted_index_docs[word])
                        idf = log(file_count / df)
                        term_weight = tf_value * idf
                        # Index within this batch only
                        batch_row.append(local_idx)  
                        batch_col.append(doc_terms.index(word))
                        batch_data.append(term_weight)

            # Create batch-specific sparse matrix
            batch_matrix = csr_matrix((batch_data, (batch_row, batch_col)), shape=(len(batch_files), len(doc_terms)))
            # Stack batches vertically
            doc_tf_idf_matrix = vstack([doc_tf_idf_matrix, batch_matrix])  

        # Normalize and save the full matrix and doc_ids
        doc_tf_idf_matrix = sparse_normalize(doc_tf_idf_matrix, norm='l2', axis=1)
        save_npz("document_tf_idf.npz", doc_tf_idf_matrix)
        with open("doc_ids.pkl", 'wb') as f:
            # Save doc_ids for future use
            pickle.dump(doc_ids, f)  

    return file_count, doc_terms, doc_ids, doc_tf_idf_matrix


def query_tf_idf_calculation(inverted_index_docs, doc_terms, query_numbers, queries, file_count):
    query_terms, q_row, q_col, q_data = doc_terms, [], [], []
    query_index = {query_num: idx for idx, query_num in enumerate(query_numbers)}
    if os.path.exists("query_tf_idf.npz"):
        query_tf_idf_matrix = load_npz("query_tf_idf.npz")
    else:
        for query_num, query in tqdm(zip(query_numbers, queries), desc="Building query TF-IDF matrix",
                                     total=len(queries)):
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
        save_npz("query_tf_idf.npz", query_tf_idf_matrix)
    return query_tf_idf_matrix


def sort_results(query_numbers, similarity_matrix, doc_ids):
    results = {query_numbers[i]: sorted(
        [(doc_ids[j], similarity_matrix[i, j]) for j in range(similarity_matrix.shape[1])],
        key=lambda x: x[1], reverse=True
    )[:10] for i in range(similarity_matrix.shape[0])}
    return results


def calculate_MAPK_MARK(relevant_docs, results, query_numbers):
    for k in [3, 10]:
    # Format relevant documents to match the predicted docs structure
        actual_docs = [relevant_docs.get(str(query_id), []) for query_id in query_numbers]
        predicted_docs_list = list(results.values())

        # Calculate MAP@K and MAR@K
        map_at_k = mapk(actual_docs, predicted_docs_list, k)
        mar_at_k = mark(actual_docs, predicted_docs_list, k)

        print(f"MAP@{k}: {map_at_k}")
        print(f"MAR@{k}: {mar_at_k}")


def main():
    start_time = time.time()
    print("Program started")
    preprocessor = Preprocessor()

    # Load or preprocess inverted index
    inverted_index_docs = inverted_indexing(preprocessor)


    # Load or preprocess queries
    queries, query_numbers = query_processing(preprocessor)


    # Batch Processing for TF-IDF Matrix
    file_count, doc_terms, doc_ids, doc_tf_idf_matrix = tf_idf_calculation(inverted_index_docs)


    # Build query TF-IDF sparse matrix
    query_tf_idf_matrix = query_tf_idf_calculation(inverted_index_docs, doc_terms, query_numbers, queries, file_count)


    # Calculate cosine similarity
    similarity_matrix = sparse_cosine_similarity(query_tf_idf_matrix, doc_tf_idf_matrix)


    # Generate sorted results dictionary using `doc_ids` for indexing
    results = sort_results(query_numbers, similarity_matrix, doc_ids)


    # Save top results to CSV
    with open("result.csv", mode="w", newline="") as file:
        fieldnames = ["Query_number", "Doc_number"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for query_num, docs in results.items():
            for doc_id, _ in docs:
                writer.writerow({"Query_number": query_num, "Doc_number": doc_id})


    # MAPK and MARK calculation
    # Use defaultdict to automatically handle lists
    # relevant_docs = defaultdict(list)  
    # with open("dev_query_results_small.csv", newline="") as csvfile:
    #     reader = csv.reader(csvfile)
    #     next(reader)
    #     for row in reader:
    #         # Append each doc_id to the list for the query
    #         relevant_docs[row[0]].append(row[1])  


    # # match format of results and relevant_docs
    # results = {key: [strip_output_id(item) for item in val] for key, val in results.items()}


    # calculate_MAPK_MARK(relevant_docs, results, query_numbers)


    end_time = time.time()
    duration = end_time - start_time
    print(f"Program completed. Total time taken: {duration:.2f} seconds.")
    return 0


if __name__ == "__main__":
    main()