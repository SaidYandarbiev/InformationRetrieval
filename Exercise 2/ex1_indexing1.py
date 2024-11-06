import lucene
#Import necessary modules
from java.nio.file import Paths
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.document import Document
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import IndexWriterConfig
from org.apache.lucene.index import IndexWriter
from org.apache.lucene.document import TextField, Field


print(lucene.VERSION)

#initialize virtual machine to adapt python code to java
lucene.initVM()

#path to store indices
index_dir = "index"

directory = FSDirectory.open(Paths.get(index_dir))

# Creates a Lucene Document
document = Document()

#Instantiate analyzer
analyzer = StandardAnalyzer()

#Instantiate indexing and configure index writer
indexWriterConfig = IndexWriterConfig(analyzer)
#indexWriterConfig.setOpenMode(IndexWriterConfig.OpenMode.CREATE) #OpenMode: APPEND, CREATE,CREATE_OR_APPEND
indexWriter = IndexWriter(directory, indexWriterConfig)

# Index a single file
# read file
f = open("full_docs_small/output_1.txt", "r")
text_to_index = f.read()
print(text_to_index)

# Build document
# add fields and field content to document
# Field.Store.YES whether to store this field for retrieval
document.add(TextField("text_content", text_to_index, Field.Store.YES))

# write document to idex
indexWriter.addDocument(document)

indexWriter.commit()

# index document2
f = open("docs_small/output_2.txt", "r")
text_to_index = f.read()
print(text_to_index)
document = Document()
document.add(TextField("text_content", text_to_index, Field.Store.YES))
indexWriter.addDocument(document)
# close index writer
indexWriter.close()


