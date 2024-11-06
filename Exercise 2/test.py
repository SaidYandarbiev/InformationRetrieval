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


# Example queries (different types of queries)
# query1 = query_parser.parse("Milestones")          # Basic term query
# query2 = query_parser.parse('"Milestones your"~3') # Proximity search
# query_parser.setAllowLeadingWildcard(True)
# query3 = query_parser.parse("*lestone*")           # Wildcard search


# Step 3: Document Search and Retrieval
# Open index directory and create IndexReader and IndexSearcher
reader = DirectoryReader.open(FSDirectory.open(Paths.get(index_dir)))
searcher = IndexSearcher(reader)

for i in range(len(queries[0])):
    query_number = query_numbers[i]
    query = queries[i]

    lucene_query = query_parser.parse(query)
    print(query_number)
    hits = searcher.search(lucene_query, 10).scoreDocs
    print(hits)
    # Optionally, process each hit document
    for hit in hits:
        doc_id = hit.doc
        doc_score = hit.score
        document = searcher.storedFields().document(doc_id)
        
        # Print or store results as needed
        print(f"Query Number: {query_number}, Doc ID: {doc_id}, Score: {doc_score}")

# # Example: Perform a search using a TermQuery
# query = TermQuery(Term("text_content", "Milestones"))  # Modify this term as needed for different queries
# hits = searcher.search(query, 10).scoreDocs            # Retrieve top 10 results

# # Display the search results
# for hit in hits:
#     doc_id = hit.doc
#     doc = searcher.doc(doc_id)
#     print(f"Found document with content: {doc.get('text_content')}")
#     print(f"Score: {hit.score}")

# Close the IndexReader
reader.close()