# Exercise 6: Building a Document Retrieval System
# In this exercise, you will build a document retrieval system that fetches and splits documents from the web, embeds them using a word-embedding model, and stores them in a vector store for efficient retrieval.

# from Exercise5 continued:
# The document chunks need to be embedded using a word-embedding model. 
# Use  TogetherEmbeddings model to see how they affect document embeddings.
# Creating a Vector Store:
# After embedding the document chunks, you need to store these embeddings in a vector store (like FAISS).
# Task:
# Run the create_db(docs) function to create the vector store from the document chunks.
# Investigate how vector stores like FAISS work by exploring their from_documents() method. 
# Setting up Retrieval:
# The next step involves setting up the retrieval pipeline that allows us to query the stored documents.
# Task:
# Implement a query pipeline that retrieves documents relevant to a user’s question.
# Modify the chain.invoke() call to query the vector store based on a user's question (e.g., "What is LCEL?").
# Ensure that the context returned by the vector store is passed into the LLM for processing.


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

from langchain_together import TogetherEmbeddings

# Initialize the Together AI embedding model
embedding_model = TogetherEmbeddings(
    model="togethercomputer/m2-bert-80M-8k-retrieval"
)

from langchain_community.vectorstores import FAISS

def create_vector_store(docs, embedding_model):
    """Embeds documents and stores them in a FAISS vector store."""
    
    # Create FAISS vector store
    vector_store = FAISS.from_documents(docs, embedding_model)
    
    print(f"Vector store created with {len(docs)} document chunks.")
    return vector_store

# Create FAISS vector store
vector_store = create_vector_store(split_documents, embedding_model)

def retrieve_relevant_docs(query, vector_store, top_k=3):
    """Retrieves the most relevant documents based on a query."""
    
    relevant_docs = vector_store.similarity_search(query, k=top_k)
    
    # Extract only the content of the retrieved documents
    return [doc.page_content for doc in relevant_docs]

# Example query
query = "What is LCEL?"
retrieved_docs = retrieve_relevant_docs(query, vector_store)

print("\nRetrieved Documents:")
for idx, doc in enumerate(retrieved_docs, 1):
    print(f"{idx}. {doc[:200]}...")  # Show first 200 characters
