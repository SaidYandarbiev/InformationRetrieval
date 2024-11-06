# Import necessary libraries
import lucene
from java.nio.file import Paths
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.document import Document, TextField, Field
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import IndexWriterConfig, IndexWriter, DirectoryReader, Term
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher, TermQuery
import os
import pandas as pd
from preprocess import Preprocessor
import csv

# Initialize the Lucene VM
lucene.initVM()


# Step 1: Document Analysis and Indexing
# Set up the directory for index storage
index_dir = "index"
txt_file_path = 'full_docs_small/'

exists = True

if os.path.exists(index_dir):
    print("Index exists.")
else:
    print("Index does not exist.")
    exists = False

directory = FSDirectory.open(Paths.get(index_dir))

# Define an analyzer
analyzer = StandardAnalyzer()

# Configure and create the IndexWriter
indexWriterConfig = IndexWriterConfig(analyzer)
indexWriter = IndexWriter(directory, indexWriterConfig)


# Document indexing
if not exists:
    file_count = sum([1 for filename in os.listdir(txt_file_path) if os.path.isfile(os.path.join(txt_file_path, filename))])
    curr_file = 0
    for file in os.listdir(txt_file_path):
        file_path = os.path.join(txt_file_path, file)
        if os.path.isfile(file_path) and file_path.endswith('.txt'):
            curr_file += 1
            print(curr_file/file_count * 100)
            with open(txt_file_path + file, "r") as f:
                text_to_index = f.read()
                token_stream = analyzer.tokenStream("text_content", text_to_index)
                token_stream.reset()  # Reset the stream
                document = Document()
                document.add(TextField("text_content", token_stream))
                document.add(TextField("filename", file, TextField.Store.YES))  # Store filename for uniqueness
                
                # Add the document to the index
                indexWriter.addDocument(document)  # Add document to the index
                indexWriter.commit()  # Commit the changes
# Close the IndexWriter once all documents are indexed
indexWriter.close()


# Step 2: Query Processing
# Re-initialize analyzer (for consistency with indexing) and set up query parser

analyzer = StandardAnalyzer()
query_parser = QueryParser("text_content", analyzer)
preprocessor = Preprocessor()
file = pd.read_excel('dev_small_queries.xlsx')
query_numbers = file['Query number'].tolist()
queries = file['Query'].tolist()

new_queries = []

for query in queries:
    new_queries.append(preprocessor.preprocess(query))

queries = new_queries

# Example queries (different types of queries)
# query1 = query_parser.parse("Milestones")          # Basic term query
# query2 = query_parser.parse('"Milestones your"~3') # Proximity search
# query_parser.setAllowLeadingWildcard(True)
# query3 = query_parser.parse("*lestone*")           # Wildcard search


# Step 3: Document Search and Retrieval
# Open index directory and create IndexReader and IndexSearcher
reader = DirectoryReader.open(FSDirectory.open(Paths.get(index_dir)))
searcher = IndexSearcher(reader)

# Open CSV file for writing results
output_file = 'results/results.csv'
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Query Number", "Doc ID"])  # Header for CSV
    
    # Iterate over each query, search and write results
    retrieved_docs = {}
    for i, query_text in enumerate(queries):
        query_number = query_numbers[i]
        
        # Parse the query
        lucene_query = query_parser.parse(query_text)
        
        # Perform the search and retrieve top 10 results
        hits = searcher.search(lucene_query, 10).scoreDocs
        
        # Store each result for the current query
        retrieved_docs[query_number] = []
        for hit in hits:
            doc_id = hit.doc
            retrieved_docs[query_number].append(doc_id)
            
            # Write query number and doc ID to the CSV file
            writer.writerow([query_number, doc_id])

# Close the IndexReader after completion
reader.close()

ground_truth_file = 'dev_query_results_small.csv'  # Update path as needed
relevant_docs = {}
with open(ground_truth_file, 'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        if row[0] != "Query_number":
            query_number = int(row[0])
            doc_id = int(row[1])
            
            if query_number not in relevant_docs:
                relevant_docs[query_number] = set()
            relevant_docs[query_number].add(doc_id)

# Step 3: Define MAP@K and MAR@K Calculation Functions
def calculate_mapk(relevant_docs, retrieved_docs, k=10):
    avg_precisions = []
    
    for query, relevant_set in relevant_docs.items():
        retrieved = retrieved_docs.get(query, [])[:k]
        relevant_count, precision_sum = 0, 0.0
        
        for i, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant_set:
                relevant_count += 1
                precision_sum += relevant_count / i  # Precision at i for this doc

        # Average Precision for this query
        if relevant_count > 0:
            avg_precision = precision_sum / min(len(relevant_set), k)
            avg_precisions.append(avg_precision)
        else:
            avg_precisions.append(0.0)  # No relevant docs found

    # Mean Average Precision at K
    return sum(avg_precisions) / len(avg_precisions)

def calculate_mark(relevant_docs, retrieved_docs, k=10):
    recalls = []
    
    for query, relevant_set in relevant_docs.items():
        retrieved = retrieved_docs.get(query, [])[:k]
        relevant_retrieved = len([doc_id for doc_id in retrieved if doc_id in relevant_set])
        recall_at_k = relevant_retrieved / len(relevant_set) if relevant_set else 0
        recalls.append(recall_at_k)

    # Mean Average Recall at K
    return sum(recalls) / len(recalls)

# Step 4: Compute MAP@K and MAR@K
k = 10
mapk = calculate_mapk(relevant_docs, retrieved_docs, k)
mark = calculate_mark(relevant_docs, retrieved_docs, k)

print(f"MAP@{k}: {mapk:.4f}")
print(f"MAR@{k}: {mark:.4f}")