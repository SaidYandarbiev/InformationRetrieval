from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import os
import sqlite3
from time import sleep
from math import log, sqrt
from math import log, sqrt
import pandas
from preprocess import Preprocessor
import csv
import time

directory = './full_docs_small'
extension = '.txt'
radius = 10
k = 3
max_terms_in_memory = 1000

#This function determines the cosine similarity for a given query and document
def cosine_similarity(vec_a, vec_b):
    dot_product = 0
    magnitude_a = 0
    magnitude_b = 0

    # Combine terms in both vectors
    all_terms = set(vec_a.keys()).union(set(vec_b.keys()))

    # Calculate dot product and magnitudes
    for word in all_terms:
        a_val = vec_a.get(word, 0)
        b_val = vec_b.get(word, 0)
        dot_product += a_val * b_val
        magnitude_a += a_val ** 2
        magnitude_b += b_val ** 2

    # Avoid division by zero
    if magnitude_a == 0 or magnitude_b == 0:
        return 0
    return dot_product / (sqrt(magnitude_a) * sqrt(magnitude_b))

#This function normalizes the tf_idf values so that they can be used to calculate the cosine similarity
def normalize(vector):
    norm = 0
    for value in vector:
        norm += value**2
    norm = sqrt(norm)
    if norm == 0:
        return vector
    for i in range(0, len(vector)):
        vector[i] = vector[i]/norm    
    return vector

def main():
    inverted_index_docs = {}
    file_count = 0
    document_map = {}
    preprocessor = Preprocessor()
    for file in os.listdir(directory):
        file_count += 1
        if file.endswith(extension):
            with open('full_docs_small/' + file, 'r', encoding='utf-8') as document:
                text = document.read()
                modified_wordlist = preprocessor.preprocess(text)    
                filename = os.path.basename(document.name)
                document_map[filename] = modified_wordlist

    for file, wordlist in document_map.items():
        for word in wordlist:
            if word not in inverted_index_docs.keys():
                inverted_index_docs[word] = []   
            if file not in inverted_index_docs[word]:    
                inverted_index_docs[word].append(file)

    file = pandas.read_excel('dev_small_queries.xlsx')
    query_numbers = file['Query number'].tolist()
    queries = file['Query'].tolist()
    new_queries = []
    for query in queries:
        new_queries.append(preprocessor.preprocess(query))    
    queries = new_queries

    document_tf_idf = defaultdict(dict)
    for doc_id, tokens in document_map.items():
        tf = {}
        for word in tokens:
            if word not in tf:
                tf[word] = 0
            tf[word] += 1 
        for word, freq in tf.items():
            tf_value = 1 + log(freq)
            df = len(inverted_index_docs[word])
            idf = log(file_count/df)
            document_tf_idf[doc_id][word] = tf_value * idf 

    for doc_id in document_tf_idf.keys():
        wordlist = []
        normalizelist = []
        for word in document_tf_idf[doc_id].keys():
            wordlist.append(word)
            normalizelist.append(document_tf_idf[doc_id][word])
        normalizelist = normalize(normalizelist)
        for i in range(0, len(wordlist)):
            document_tf_idf[doc_id][wordlist[i]] = normalizelist[i]

    query_tf_idf = defaultdict(dict)
    query_count = 0
    for query in queries:
        tf = {}
        for word in query:
            if word not in tf:
                tf[word] = 0
            tf[word] += 1 
        for word, freq in tf.items():
            tf_value = 1 + log(freq)
            if word in inverted_index_docs:
                df = len(inverted_index_docs[word])
                idf = log(file_count/df)
            else:
                idf = 0 
            query_tf_idf[query_numbers[query_count]][word] = tf_value * idf
        query_count += 1        

    for query_id in query_tf_idf.keys():
        wordlist = []
        normalizelist = []
        for word in query_tf_idf[query_id].keys():
            wordlist.append(word)
            normalizelist.append(query_tf_idf[query_id][word])
        normalizelist = normalize(normalizelist)
        for i in range(0, len(wordlist)):
            query_tf_idf[query_id][wordlist[i]] = normalizelist[i]    

    results = {}
    for query in query_numbers:
        query_list = query_tf_idf[query]
        for doc_id in document_tf_idf.keys():
            doc_list = document_tf_idf[doc_id]
            similarity = cosine_similarity(query_list, doc_list)
            if similarity > 0:
                if query not in results:
                    results[query] = []
                results[query].append([doc_id, similarity])

    new_results = {}
    for query in results.keys():
        similarity_list = []
        for similarity in results[query]:
            similarity_list.append(similarity)
        similarity_list.sort(key=lambda x: x[1], reverse=True)
        # if query == 1086886: #Je kan hier de query nummer vervangen om specifieke queries te checken of dit overeeenkomt met de beste file voor deze query
        #     print(query)
        #     print(similarity_list)
        #     sleep(100) 
        new_results[query] = similarity_list 
    results = new_results

    top_10_results = []
    for query in results.keys():
        i = 0    
        for doc_id in results[query]:
            if i < 10:
                top_10_results.append({"Query_number": query, "Doc_number": doc_id})
                i += 1
            else:
                break
    with open("results.csv", mode="w", newline="") as file:
        fieldnames = ["Query_number", "Doc_number"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for query_result in top_10_results:
            writer.writerow(query_result)

    
    relevant_docs = {}
    with open("dev_query_results_small.csv", newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            relevant_docs[row[0]] = row[1]
            

    total_average_precision = 0
    total_average_recall = 0

    for query in results.keys():
        APK = 0
        ARK = 0
        for i in range(0, len(results[query])):
            if i > 9:
                break
            doc_id = results[query][i][0].split('.')[0]
            doc_id = doc_id.split('_')[1]
            if str(doc_id) == relevant_docs[str(query)]:
                APK = float(1)/float(i+1)
                ARK = 1
                break    
        total_average_precision += APK
        total_average_recall += ARK
    MAPK = total_average_precision/len(query_numbers)
    MARK = total_average_recall/len(query_numbers)

    print(MAPK)
    print(MARK)

    return 0    

if __name__ == "__main__":
    main()  
