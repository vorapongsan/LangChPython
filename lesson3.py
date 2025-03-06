# Output Parser
# 1. response without output parser
# 2. response with str output parser
# 3. response with list output parser
# 4. response with json output parser

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser
from pydantic.v1 import BaseModel, Field

# Initialize the ChatGroq object
llm = ChatGroq(
   #model="llama-3.2-3b-preview",
    model ="llama-3.3-70b-versatile",
    temperature=0,
)


# create prompt template
def get_response():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Generate a list of 10 synonyms for the following word. Return the results as a comma separated list."),
            ("user", '{subject}')
        ]
    )
    # create LLM chain
    chain = prompt | llm
    # invoke the chain
    response = chain.invoke({'subject':"AI"})
    return response.content


print(get_response())


def get_response_str_output_parser():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Generate a list of 10 synonyms for the following word. Return the results as a comma separated list."),
            ("user", '{subject}')
        ]
    )
    #====== with output parser =======
    outputParser = StrOutputParser()
    # create LLM chain
    chain = prompt | llm |outputParser

    # invoke the chain
    response = chain.invoke({'subject':"AI"})
    return response
print(get_response_str_output_parser())

#====== with list output parser =======
def get_response_list_output_parser():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Generate a list of 10 synonyms for the following word. Return the results as a comma separated list."),
            ("user", '{subject}')
        ]
    )
    outputParser = CommaSeparatedListOutputParser()
    # create LLM chain
    chain = prompt | llm |outputParser

    # invoke the chain
    response = chain.invoke({'subject':"AI"})
    return response
print(get_response_list_output_parser())

#====== with json output parser =======
def get_response_json_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract information from the following phrase.\nFormatting Instructions: {format_instructions}"),
        ("human", "{phrase}")
    ])

    class Person(BaseModel):
        name: str = Field(description="the name of the person")
        age: int = Field(description="the age of the person")
        

    parser = JsonOutputParser(pydantic_object=Person)

    chain = prompt | llm | parser
    
    return chain.invoke({
        "phrase": "Mary is 25 years old",
        "format_instructions": parser.get_format_instructions()
    })
print(get_response_json_output_parser())