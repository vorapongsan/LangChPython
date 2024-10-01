# Output Parser

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser
from pydantic.v1 import BaseModel, Field

# Initialize the ChatGroq object
llm = ChatGroq(
    #model="llama-3.2-3b-preview",
    model ="llama3-groq-8b-8192-tool-use-preview",
    temperature=0,
)


# create prompt template


def get_response():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Generate a list of 10 synonyms for the following word. Return the results as a comma seperated list."),
            ("user", '{subject}')
        ]
    )
    # create LLM chain
    chain = prompt | llm
    # invoke the chain
    response = chain.invoke({'subject':"AI"})
    return response.content

def get_response_str_output_parser():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Generate a list of 10 synonyms for the following word. Return the results as a comma seperated list."),
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


#====== with list output parser =======
def get_response_list_output_parser():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Generate a list of 10 synonyms for the following word. Return the results as a comma seperated list."),
            ("user", '{subject}')
        ]
    )
    outputParser = CommaSeparatedListOutputParser()
    # create LLM chain
    chain = prompt | llm |outputParser

    # invoke the chain
    response = chain.invoke({'subject':"AI"})
    return response

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




print(get_response())
print(get_response_str_output_parser())
print(get_response_list_output_parser())
print(get_response_json_output_parser())