# Exercise 3: Response Without Output Parser

# Create a Prompt Template using ChatPromptTemplate.from_messages().
# The system message should instruct the model to generate 5 tips on the user-provided topic (e.g., “fitness”, “productivity”, etc.).
# Example: “Provide 5 tips on {topic}.”
# Create an LLM Chain by connecting the prompt to the LLM (e.g., ChatGroq).
# Invoke the Chain with a topic of your choice (e.g., “public speaking”).
# Print the response without using any output parser.
# Observe that the response is just raw text.
# Example Output (raw text, unstructured):

# 1. ...
# 2. ...
# 3. ...
# 4. ...
# 5. ...

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

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

# Step 3: Create LLM Chain (Without Output Parser)
chain = prompt | llm

# Step 4: Invoke the Chain with a Topic (e.g., Public Speaking)
response = chain.invoke({"topic": "Public Speaking"})

# Step 5: Print the Raw Response
print(response.content)
