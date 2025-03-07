# Retrieval Chain : load the document from text.
# 1. Create a document from text
# 2. Create and invoke the document with the chain
# 3. Create and invoke the document with create_stuff_documents_chain


from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
#from langchain.chains.combine_documents import create_stuff_documents_chain

docA = Document(
    page_content= '''LangChain Expression Language, or LCEL, is a declarative way to easily compose chains 
    together. LCEL was designed from day 1 to support putting prototypes in production, 
    with no code changes, from the simplest “prompt + LLM” chain to the most complex chains 
    (we’ve seen folks successfully run LCEL chains with 100s of steps in production). 
    To highlight a few of the reasons you might want to use LCEL'''
)

docB = Document(
    page_content= '''Mahidol University is an autonomous public research university in Thailand. 
    The university was founded as part of Siriraj Hospital in 1888. It was first called the University of Medical Science in 1943, 
    and has been recognized as Thailand's fourth public university. The university was renamed in 1969 by King Bhumibol Adulyadej 
    for his father, Prince Mahidol of Songkhla, known as the "Father of Modern Medicine and Public Health in Thailand".[1]
    Originally focused on the health sciences, it has expanded into other fields. The university hosted Thailand's first medical school, 
    Siriraj Medical School.[2] Mahidol offers a range of graduate (primarily international) and undergraduate programs, 
    from the natural sciences to the liberal arts, with remote campuses in Kanchanaburi, Nakhon Sawan, and Amnat Charoen provinces. 
    There are a total of 629 programs[3] in 17 faculties, six colleges, nine research institutions and six campuses. 
    The university has the largest budget of any public university in Thailand: $430 million in 2019, most of which is for 
    graduate research programs. Mahidol had an acceptance rate of 0.4 percent in medicine for the 2016 academic year, 
    and was ranked Thailand's number-one university in 2011 by QS Asian University Rankings.[4]'''
)

# Initialize the ChatGroq object
llm = ChatGroq(
    #model="llama-3.2-3b-preview",
    model ="llama-3.3-70b-versatile",
    temperature=0,
)

prompt = ChatPromptTemplate.from_template(""" 
    Answer the user's question.
    Context :  {context}
    User Question : {input}
""")

chain = prompt | llm
response = chain.invoke({
    'input':"What is Mahidol",
    'context': [docA, docB]
})
print(response.content)


# chain = create_stuff_documents_chain(
#     llm =llm,
#     prompt = prompt
#     )

# response = chain.invoke({
#     'input':"What is LCEL",
#     'context': [docA]
# })
# print(response)