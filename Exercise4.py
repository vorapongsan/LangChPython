# Exercise 4: Response with StrOutputParser

# Re-use or create a similar prompt as in Exercise 1, but this time use StrOutputParser() at the end of the chain.
# Invoke the chain and store the result in a variable.
# Print the result and observe how the output is always returned as a simple Python string.
# Example Output:

# "1. ...\n2. ...\n3. ...\n4. ...\n5. ..."
# (Where \n indicates newlines.)

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser

# Step 1: Initialize the ChatGroq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# Step 2: Create a Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Provide 5 tips on the given topic."),
    ("user", "{topic}")
])

# Step 3: Add StrOutputParser to ensure output is always a string
output_parser = StrOutputParser()

# Step 4: Create LLM Chain with Output Parser
chain = prompt | llm | output_parser

# Step 5: Invoke the Chain with a Topic (e.g., Time Management)
response = chain.invoke({"topic": "Time Management"})

# Step 6: Print the Parsed String Response
print(response)
