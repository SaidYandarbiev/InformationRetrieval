import lucene
from org.apache.lucene.store import FSDirectory
from java.nio.file import Paths
from org.apache.lucene.index import DirectoryReader
from java.nio.file import Paths
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.search import TermQuery
from org.apache.lucene.index import Term


# Initialize the Lucene VM
lucene.initVM()

# Path to the Lucene index directory
index_path = "index/"

#Open the index directory
index_dir = FSDirectory.open(Paths.get(index_path))
# create reader object
reader = DirectoryReader.open(index_dir)

# instatiate/define reader
searcher = IndexSearcher(reader)

query = TermQuery(Term("text_content", "Milestones")) #no analyzer so case sensitive

#Execute the search
print("searching")
hits = searcher.search(query, 10).scoreDocs  # Get top 10 results

for hit in hits:
    doc_id = hit.doc
    doc = searcher.doc(doc_id)
    print(f"Found document with content: {doc.get('text_content')}")
    print(f"Path: {doc.get('filepath')}")
    print(f"Score: {hit.score}")

print("search complete")
print(hits)

reader.close()
