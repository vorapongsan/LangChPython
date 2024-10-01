# Prompt template and LLM chain

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Initialize the ChatGroq object
llm = ChatGroq(
    model="llama-3.2-3b-preview",
    temperature=0,
)





# Create a prompt template
# prompt = ChatPromptTemplate.from_template("Write a poem about {subject}")

# # Create LLM Chain

# chain =  prompt | llm

# resp = chain.invoke({'subject':"AI"})

# print(resp.content)




## create prompt template
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "Write a poem about of the given topic"),
#         ("user", '{subject}')
#     ]
# )

# # create LLM chain
# chain = prompt | llm


# # invoke the chain
# response = chain.invoke({'subject':"AI"})
# print(response.content)


# create prompt template
prompt2 = ChatPromptTemplate.from_messages(
    [
        ("system", "list the synonyms of the given word"),
        ("user", '{subject}')
    ]
)

# create LLM chain
chain = prompt2 | llm


# invoke the chain
response = chain.invoke({'subject':"AI"})
print(response.content)