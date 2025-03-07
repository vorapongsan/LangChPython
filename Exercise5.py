# Exercise 5: Load and Split Documents
# After complete lesson 6
# Task:
# Create a function load_and_split_document(url) that:
# Uses WebBaseLoader to fetch content from a URL.
# Splits the content into chunks of 250 characters with 50-character overlap using RecursiveCharacterTextSplitter.
# Returns the split documents.
# Call the function with:
# "https://python.langchain.com/v0.1/docs/expression_language/"
# and print the number of chunks produced.
# ✅ Hint: Modify the given get_document_from_web() function.

from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter



def load_and_split_document(url):
    """Loads a document from the web and splits it into chunks."""
    
    # Step 1: Load the document from the web
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    # Step 2: Split the document into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,       # Each chunk will have 250 characters
        chunk_overlap=50      # 50 characters will overlap between consecutive chunks
    )
    split_docs = splitter.split_documents(docs)

    # Step 3: Return the split documents
    return split_docs

# Call the function with the given URL
url = "https://python.langchain.com/v0.1/docs/expression_language/"
split_documents = load_and_split_document(url)

# Print the number of chunks produced
print(f"Total number of chunks: {len(split_documents)}")
