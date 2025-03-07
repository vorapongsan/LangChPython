# Exercise 2: Generate a List Using CommaSeparatedListOutputParser
# After complete lesson3.py 
# Task:
# Modify the prompt so that the LLM generates a list of 5 planets in our solar system.
# Use CommaSeparatedListOutputParser() to ensure the response is structured as a list.
# Print the structured output.

# ✅ Hint:
# Use:

# CommaSeparatedListOutputParser()
# ✅ Expected Output Example:
# ["Mercury", "Venus", "Earth", "Mars", "Jupiter"]


# Output Parser
# 1. response without output parser
# 2. response with str output parser
# 3. response with list output parser
# 4. response with json output parser

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser

# Initialize the ChatGroq object
llm = ChatGroq(
   #model="llama-3.2-3b-preview",
    model ="llama-3.3-70b-versatile",
    temperature=0.5,
)

#====== with list output parser =======
outputParser = CommaSeparatedListOutputParser()

# create LLM chain
chain =   llm |outputParser

# invoke the chain
response = chain.invoke("Generate a list of 5 planets in our solar system. Return the results as a comma separated list.")
print(response[0])