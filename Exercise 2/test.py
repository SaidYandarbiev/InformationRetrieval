# Import necessary libraries
from collections import defaultdict
import lucene
from java.nio.file import Paths
from java.io import StringReader
from java.util.regex import Pattern  # Import Pattern from java.util.regex
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.document import Document, TextField, Field
from org.apache.lucene.analysis.standard import StandardTokenizer
from org.apache.lucene.analysis.core import LowerCaseFilter, StopFilter
from org.apache.lucene.analysis.miscellaneous import ASCIIFoldingFilter
from org.apache.lucene.analysis.pattern import PatternReplaceFilter
from org.apache.lucene.analysis import TokenStream
from org.apache.lucene.analysis.tokenattributes import CharTermAttribute
from org.apache.lucene.index import IndexWriterConfig, IndexWriter, DirectoryReader
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.analysis.standard import StandardAnalyzer
import os
import pandas as pd
import csv

# Initialize the Lucene VM
lucene.initVM()

# Define a basic list of common English stopwords
stopwords = ["a", "an", "the", "and", "or", "not", "is", "are", "in", "of", "to", "with", "on", "for", "by"]

# Helper function to preprocess text using custom TokenStream
def preprocess_text(text, analyzer):
    #Preprocesses text using a series of filters to remove special characters, 
    #normalize case, and apply stopword filtering.
    
    tokenized_terms = []
    
    # Set up the tokenizer
    tokenizer = StandardTokenizer()
    tokenizer.setReader(StringReader(text))
    
    # Apply filters
    tokenStream = LowerCaseFilter(tokenizer)  # Lowercase all tokens
    tokenStream = ASCIIFoldingFilter(tokenStream)  # Remove accents
    # Create a Java Pattern object for non-alphanumeric characters
    pattern = Pattern.compile("[^a-zA-Z0-9]")
    tokenStream = PatternReplaceFilter(tokenStream, pattern, "", True)  # Remove non-alphanumeric
    tokenStream = StopFilter(tokenStream, StopFilter.makeStopSet(stopwords))  # Remove stopwords using makeStopSet
    
    # Collect tokens
    tokenStream.reset()  # Initialize token stream
    while tokenStream.incrementToken():
        tokenized_terms.append(tokenStream.getAttribute(CharTermAttribute.class_).toString())
    tokenStream.end()
    tokenStream.close()
    
    return " ".join(tokenized_terms)

# Set up directories and configurations
index_dir = "index"
txt_file_path = 'full_docs_small'

if not os.path.exists(index_dir):
    os.makedirs(index_dir)

directory = FSDirectory.open(Paths.get(index_dir))
analyzer = StandardAnalyzer()
indexWriterConfig = IndexWriterConfig(analyzer)
indexWriter = IndexWriter(directory, indexWriterConfig)

# Indexing function that uses preprocessing with token filters
def index_txt_file(ind_writer, file):
    doc = Document()
    # Extract the document ID (filename without extension)
    doc_id = os.path.splitext(os.path.basename(file))[0]  # Remove .txt extension
    with open(file, "r") as f:
        text_to_index = f.read()
        # Preprocess the text using custom filters
        preprocessed_text = preprocess_text(text_to_index, analyzer)
        doc.add(TextField("text_content", preprocessed_text, Field.Store.YES))
        doc.add(TextField("doc_id", doc_id, Field.Store.YES))  # Store document ID as a separate field
        ind_writer.addDocument(doc)

# Index all files in the directory
data_dir = "full_docs_small"
for file in os.listdir(data_dir):
    if file.endswith(".txt"):
        data_path = os.path.join(data_dir, file)
        print(f"Indexing file: {data_path}")
        index_txt_file(indexWriter, data_path)

indexWriter.close()
print("Indexing complete.")

# Query Processing and Retrieval
file = pd.read_excel('dev_small_queries.xlsx')
query_numbers = file['Query number'].tolist()
queries = file['Query'].tolist()

# Preprocess queries with the custom token stream setup
processed_queries = [preprocess_text(query, analyzer) for query in queries]

# Open index directory and create IndexReader and IndexSearcher
reader = DirectoryReader.open(FSDirectory.open(Paths.get(index_dir)))
searcher = IndexSearcher(reader)
query_parser = QueryParser("text_content", analyzer)

# Write search results to a CSV file
output_file = 'results/results.csv'
retrieved_docs = {}
for i, query_text in enumerate(processed_queries):
    query_number = query_numbers[i]
    lucene_query = query_parser.parse(query_text)
    
    # Retrieve top 10 results
    hits = searcher.search(lucene_query, 10).scoreDocs
    retrieved_docs[query_number] = []
    for hit in hits:
        doc = searcher.storedFields().document(hit.doc)
        doc_id = doc.get("doc_id")  # Retrieve document ID with "output_" prefix
        retrieved_docs[query_number].append(doc_id)

# Remove the "output_" prefix from each document ID
for query_number, doc_ids in retrieved_docs.items():
    retrieved_docs[query_number] = [doc_id.replace("output_", "") for doc_id in doc_ids]

# Write cleaned document IDs to the CSV file
with open(output_file, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    # CSV Header
    writer.writerow(["Query Number", "Doc ID"])  
    
    for query_number, doc_ids in retrieved_docs.items():
        for doc_id in doc_ids:
            writer.writerow([query_number, doc_id])

reader.close()
print("Search and retrieval complete.")


ground_truth_file = 'dev_query_results_small.csv'  # Update path as needed
relevant_docs = defaultdict(list)
with open(ground_truth_file, newline="") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        # Append each doc_id to the list for the query
        relevant_docs[row[0]].append(row[1])  

# Convert keys and values in relevant_docs to strings
relevant_docs = {str(query): list(map(str, docs)) for query, docs in relevant_docs.items()}

# Convert keys and values in retrieved_docs to strings
retrieved_docs = {str(query): list(map(str, docs)) for query, docs in retrieved_docs.items()}

print(relevant_docs)
# Step 3: Define MAP@K and MAR@K Calculation Functions

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


def calculate_MAPK_MARK(relevant_docs, results, query_numbers):
    for k in [1,3,5, 10]:
    # Format relevant documents to match the predicted docs structure
        actual_docs = [relevant_docs.get(str(query_id), []) for query_id in query_numbers]
        predicted_docs_list = list(results.values())

        # Calculate MAP@K and MAR@K
        map_at_k = mapk(actual_docs, predicted_docs_list, k)
        mar_at_k = mark(actual_docs, predicted_docs_list, k)

        print(f"MAP@{k}: {map_at_k}")
        print(f"MAR@{k}: {mar_at_k}")

calculate_MAPK_MARK(relevant_docs, retrieved_docs, query_numbers)
