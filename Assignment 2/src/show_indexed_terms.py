import lucene
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import DirectoryReader
from java.nio.file import Paths
from org.apache.lucene.util import BytesRefIterator


# Initialize the JVM and Lucene
lucene.initVM()

# Specify the path to the Lucene index directory
index_dir = "index"

# Open the directory and the index reader
directory = FSDirectory.open(Paths.get(index_dir))
reader = DirectoryReader.open(directory)

# Specify the field you want to inspect
field_name = "text_content"

# Loop through each leaf (segment) in the index
for leaf in reader.leaves():
    leaf_reader = leaf.reader()
    
    # Get terms for the specific field in this segment
    terms = leaf_reader.terms(field_name)
    
    if terms is not None:
        terms_enum = terms.iterator()  # Get the terms iterator
        print(f"Field: {field_name} in segment")
        
        # Iterate through the terms in the "content" field
        for term in BytesRefIterator.cast_(terms_enum):
            print(term.utf8ToString())
            # print(f"Term: {term_text}")

# Close the reader
reader.close()
