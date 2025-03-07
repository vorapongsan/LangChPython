# Retrieval Chain : load the document from website and split the document.
# load to word-embedding model and vector stores.
# set the word-embedding model to OllamaEmbeddings
# set the word-embedding model to TogetherEmbeddings
# not yet retrieve the answer from the vector-store.
# Change database to PineconeVectorStore 


from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
#from langchain_ollama import OllamaEmbeddings
from langchain_together import TogetherEmbeddings
from langchain_community.vectorstores.faiss import FAISS
 
from langchain_pinecone import PineconeVectorStore


embedding = OpenAIEmbeddings()

def  get_document_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
    )
    splitDocs = splitter.split_documents(docs)
    # print(len(splitDocs))

    return splitDocs


# def create_db(docs):
#     #embedding = OpenAIEmbeddings()
#     vectorStore = FAISS.from_documents(docs, embedding = embedding) 
#     return vectorStore

docs = get_document_from_web("https://python.langchain.com/v0.1/docs/expression_language/")


index_name = "langchain-test-index"
# Connect to Pinecone index and insert the chunked docs as contents
#db = PineconeVectorStore.from_documents(docs, embedding, index_name=index_name)

# Connect to Pinecone index with the existing index name
db = PineconeVectorStore(embedding = embedding, index_name=index_name)


query = "What is LCEL"
docsFromSearch = db.similarity_search(query)
print(docsFromSearch[0].page_content)
