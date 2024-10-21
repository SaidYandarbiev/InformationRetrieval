from collections import defaultdict
import os
import re
from time import sleep
from math import log
import pandas


directory = './full_docs_small'
extension = '.txt'


def intersection(lst1, lst2):
    return list(set(lst1).intersection(set(lst2)))


def smart_split(word):
    # List to hold the split words
    split_words = []
    current_word = ""

    # Iterate over each character in the word
    for i, char in enumerate(word):
        if char.isupper():
            if i > 0 and word[i-1].islower():
                # If the previous character is lowercase, split here
                split_words.append(current_word)
                current_word = char
            elif i < len(word) - 1 and word[i+1].islower():
                # If the next character is lowercase, split here
                split_words.append(current_word)
                current_word = char
            else:
                # Otherwise, just add the character to the current word
                current_word += char
        else:
            # If the character is not uppercase, continue adding to current word
            current_word += char
    
    # Append the last processed word
    if current_word:
        split_words.append(current_word)
    
    return split_words


# Function to handle uppercase word splits, underscores, hyphens, numbers and apostrophe's
def preprocess(text):
        
    # Step 1: Replace underscores with spaces
    text = text.replace('_', ' ')

    # Step 2: Extract words using your existing regex
    wordlist = re.findall(r"\b\w+(?:[']\w+)*\b", text)

    # Step 3: Process the word list, keeping only alphabetic characters in words
    cleaned_wordlist = []
    
    for word in wordlist:
        # Keep only alphabetic characters
        cleaned_word = ''.join([char for char in word if char.isalpha() or char == "'"])
        
        # If the cleaned word is not empty, process it using the smart split logic
        if cleaned_word:
            split_result = smart_split(cleaned_word)
            cleaned_wordlist.extend(split_result)

    # Step 4: Filter out any empty strings from the final list
    cleaned_wordlist = [word for word in cleaned_wordlist if word]

    # Return the cleaned word list without empty strings
    return(cleaned_wordlist)

def main():
    inverted_index_docs = {}
    file_count = 0
    document_map = {}
    for file in os.listdir(directory):
        file_count += 1
        if file.endswith(extension):

            with open('full_docs_small/' + file, 'r', encoding='utf-8') as document:
                text = document.read()
                
                # This is where we modify the entire text to our liking
                modified_wordlist = preprocess(text=text)

                filename = os.path.basename(document.name)
                document_map[filename] = modified_wordlist

                word_count = 0
                for words in modified_wordlist:
                    word_count += 1               
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
        modified_wordlist = []
        for word in text:
            word = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', word)
            word = re.sub(r'(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])', ' ', word)              
            modified_wordlist.extend(word.split())
        new_queries.append(modified_wordlist)
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

    tf_idf_weighting = defaultdict(dict)
    for doc_id, tokens in document_map.items():
        tf = {}
        for word in tokens:
            if word not in tf:
                tf[word] = 0
            tf[word] += 1
        for word in tf.keys():
            tf[word] /= len(tokens)
            df = len(inverted_index_docs[word])
            idf = log((file_count/ df + 1))
            tf_idf = tf[word] * idf
            tf_idf_weighting[doc_id][word] = tf_idf

    # term_frequencies = {}
    # for query in new_queries:
    #     for word in query:
    #         if word in inverted_index_docs.keys():
    #             docmap = inverted_index_docs[word]
    #             for docs in docmap.keys():
    #                 if docs not in term_frequencies.keys():
    #                     term_frequencies[docs] = {}
    #                 term_frequencies[docs][word] = len(inverted_index_docs[word][docs])             
    
    # document_term_frequency = {}
    # for word in inverted_index_docs.keys():
    #     document_term_frequency[word] = len(inverted_index_docs[word])

    # inverse_document_frequency = {}
    # for word in document_term_frequency.keys():    
    #     inverse_document_frequency[word] = math.log10(file_count/document_term_frequency[word])
    
    # tf_idf_weighting = {}
    # for doc in term_frequencies.keys():
    #     tf_idf_weighting[doc] = {}
    #     for word in document_term_frequency.keys():
    #         tf = 1 + math.log10(term_frequencies[doc][word]) if word in term_frequencies[doc] else 0
    #         tf_idf_weighting[doc][word] = tf

 
    return 0

if __name__ == "__main__":
    main()    