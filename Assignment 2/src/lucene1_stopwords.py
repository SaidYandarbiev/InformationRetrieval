import lucene

print(lucene.VERSION)

from java.nio.file import Paths
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import IndexWriterConfig
from org.apache.lucene.index import IndexWriter
from org.apache.lucene.document import Document, TextField, Field
from org.apache.lucene.analysis import CharArraySet
import os


print(lucene.VERSION)


# intitialize VM to adapt java lucene to python
lucene.initVM()

data_dir = "full_docs_small"


index_dir = FSDirectory.open(Paths.get("index"))




###################add stopwords
# Create a custom stopwords list
custom_stopwords = ["a","example", "models", "Milestones"]
stopwords_set = CharArraySet(len(custom_stopwords), True)

# Add each custom stopword to the set
for stopword in custom_stopwords:
    stopwords_set.add(stopword)


analyzer = StandardAnalyzer(stopwords_set)



indexWriterConfig = IndexWriterConfig(analyzer)

indexWriter = IndexWriter(index_dir, indexWriterConfig)



# Index a single file
# read file
f = open("docs_small/output_1.txt", "r")
text_to_index = f.read()
print(text_to_index)


# instantiate a document
doc = Document()

# Build document
# add feilds and field content to document
# Field.Store.YES whether to store this field for retrieval
doc.add(TextField("text_content", text_to_index, Field.Store.YES))

# write document to idex
indexWriter.addDocument(doc)

 # Lucene automatically commits changes in certain scenarios, such as when the IndexWriter is closed. 
 # However, if you want to commit changes at specific points without closing the writer, you can call index_writer.commit() manually.
indexWriter.commit()

# index document2
f = open("docs_small/output_2.txt", "r")
text_to_index = f.read()
print(text_to_index)
doc = Document()
doc.add(TextField("text_content", text_to_index, Field.Store.YES))
indexWriter.addDocument(doc)

# close index writer
indexWriter.close()

# def index_txt_file(writer, file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         content = f.read()

#     # Create a Lucene Document
#     doc = Document()

#     # Index the filename (useful to identify the document later)
#     file_name = os.path.basename(file_path)

#     doc.add(StringField("filename", file_name, Field.Store.YES))


#     # Index the content of the file
#     doc.add(TextField("content", content, Field.Store.YES))

#     # Add the document to the index
#     writer.addDocument(doc)
#     print(f"Indexed file: {file_name}")

# # Step 4: Traverse the folder and index each .txt file
# for root, dirs, files in os.walk(txt_folder_path):
#     for file in files:
#         if file.endswith(".txt"):
#             file_path = os.path.join(root, file)
#             index_txt_file(index_writer, file_path)

# # Step 5: Close the IndexWriter after indexing all files
# index_writer.commit()
# index_writer.close()

# print("Indexing complete.")

