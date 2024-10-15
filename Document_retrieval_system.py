import os
import re
from time import sleep
import math
import pandas

directory = './full_docs_small'
extension = '.txt'

def intersection(lst1, lst2):
    return list(set(lst1).intersection(set(lst2)))

def main():
    inverted_index_docs = {}
    file_count = 0
    for file in os.listdir(directory):
        file_count += 1
        if file.endswith(extension):
            #print(file)
            with open('full_docs_small/' + file, 'r', encoding='utf-8') as document:
                text = document.read()
                wordlist =  re.findall(r'\b\w+\b', text)

                # Modify words in the wordslist:
                modified_wordlist = []
                
                for word in wordlist:
                    # Uppercase split
                    # Example: "ProblemsPesticides" -> "Problems", "Pesticides"
                    word = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', word)

                    # Number split
                    # Example "hyper10" -> "hyper", "10"
                    word = re.sub(r'(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])', ' ', word)
                    
                    # Add the modified words to the list
                    modified_wordlist.extend(word.split())
            
                word_count = 0
                for words in modified_wordlist:
                    word_count += 1               
                    if len(words) > 1:
                        if words not in inverted_index_docs.keys():
                            inverted_index_docs[words] = {}
                        if file not in inverted_index_docs[words].keys():    
                            inverted_index_docs[words][file] = [word_count]
                        else:
                            inverted_index_docs[words][file].append(word_count)
                #print(inverted_index_docs) 


    file = pandas.read_excel('dev_small_queries.xlsx')
    query_numbers = file['Query number'].tolist()
    queries = file['Query'].tolist()
    new_queries = []
    for query in queries:
        text = query.split()
        new_queries.append(text)
    query_mapping = {}
    index = 0
    for query in new_queries:
        current_query = []
        cur_loop = 0       
        for word in query:
            if word in inverted_index_docs.keys():
                if len(current_query) == 0 and cur_loop == 0:
                    current_query = list(inverted_index_docs[word].keys())
                else:
                    current_query = intersection(current_query, list(inverted_index_docs[word].keys()))
            else:
                current_query = []            
            cur_loop += 1        
      
        query_mapping[query_numbers[index]] = current_query
        if len(current_query) == 0:
            query_mapping[query_numbers[index]] = '[]'
        else:
            query_mapping[query_numbers[index]] = current_query       
        index += 1
    with open("output_file.txt", "w") as file:
    # Iterate over the dictionary
        for key, value in query_mapping.items():
        # Write each key-value pair to the file in a formatted way
            file.write(f"{key}: {', '.join(map(str, value))}\n")

    term_frequencies = {}
    for query in new_queries:
        for word in query:
            if word in inverted_index_docs.keys():
                docmap = inverted_index_docs[word]
                for docs in docmap.keys():
                    if docs not in term_frequencies.keys():
                        term_frequencies[docs] = {}
                    term_frequencies[docs][word] = len(inverted_index_docs[word][docs])             
    
    document_term_frequency = {}
    for word in inverted_index_docs.keys():
        document_term_frequency[word] = len(inverted_index_docs[word])

    inverse_document_frequency = {}
    for word in document_term_frequency.keys():    
        inverse_document_frequency[word] = math.log10(file_count/document_term_frequency[word])
    
    tf_idf_weighting = {}
    for word in document_term_frequency.keys():
        tf_idf_weighting[word] = (1 + math.log10(document_term_frequency[word])) * inverse_document_frequency[word]

    print(tf_idf_weighting)
 
    return 0

if __name__ == "__main__":
    main()    