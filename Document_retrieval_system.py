from collections import defaultdict
import os
import re
from time import sleep
from math import log, sqrt
from math import log, sqrt
import pandas
import numpy as np
from preprocess import Preprocessor

directory = './full_docs_small'
extension = '.txt'
radius = 10

def cosine_similarity(vec_a, vec_b, file, query):
    dot_product = 0
    magnitude_a = 0
    magnitude_b = 0

    new_vec_b = {}
    for word in vec_a.keys():
        if word in vec_b:
            dot_product += vec_a[word] * vec_b[word]
            new_vec_b[word] = vec_b[word]

    for value in vec_a.values():
        magnitude_a += value**2
    for value in new_vec_b.values():
        magnitude_b += value**2

    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    return dot_product/ (sqrt(magnitude_a) * sqrt(magnitude_b))                

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

def intersection(lst1, lst2):
    return list(set(lst1).intersection(set(lst2)))


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
                new_wordlist = []
                for word in modified_wordlist:
                    word = word.lower()
                    new_wordlist.append(word)
                modified_wordlist = new_wordlist    
                filename = os.path.basename(document.name)
                document_map[filename] = modified_wordlist
                word_count = 0
                for words in modified_wordlist:
                    word_count += 1               
                    if words not in inverted_index_docs.keys():
                        inverted_index_docs[words] = {}
                    if file not in inverted_index_docs[words].keys():    
                        inverted_index_docs[words][file] = [word_count] 
                    inverted_index_docs[words][file].append(word_count)
                  
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
            tf = 1 + log(freq)
            df = len(inverted_index_docs[word].keys())
            idf = log(file_count/df)
            document_tf_idf[doc_id][word] = tf * idf
            
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
            tf = 1 + log(freq)
            if word in inverted_index_docs:
                df = len(inverted_index_docs[word].keys())
                idf = log(file_count/df)
            else:
                idf = 0 
            query_tf_idf[query_numbers[query_count]][word] = tf * idf
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
            similarity = cosine_similarity(query_list, doc_list, doc_id, query)
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
        if query == 1089273: #Je kan hier de query nummer vervangen om specifieke queries te checken of dit overeeenkomt met de beste file voor deze query
            print(query)
            print(similarity_list)
            sleep(100)   
    return 0

if __name__ == "__main__":
    main()  
