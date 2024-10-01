# Chat Model 

from dotenv import load_dotenv
load_dotenv()


from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI


llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_retries=2,
)

localLLM = ChatOllama(
    model="llama3.2:1b",
    temperature=0,
)

googleLLM = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
# response = llm.invoke("Write a poem about AI")
# print(response)
# print(response.content)

# response = llm.batch(["How are you","Write a poem about AI"])
# print(response[1].content)

# response = llm.stream("Write a poem about AI")
response = googleLLM.stream("Write a poem about AI")
for chunk in response:
    print(chunk.content, end="", flush=True)

