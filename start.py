from langchain_community.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

import ollama

doc_path = "./data/compass.pdf"
model = "llama3.1"

if doc_path:
    loader = UnstructuredPDFLoader(file_path=doc_path)
    data = loader.load()
    print("Done loading...")
else:
    print("Upload a PDF file")

# Preview the first page
content = data[0].page_content
# print(content[:100])

# Split and chunk
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
chunks = text_splitter.split_documents(data)
print("Done splitting...")

# print(f"Number of chunks: {len(chunks)}")
# print(f"Example chunk: {chunks[0]}")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="simple-rag"
)
print("Done adding to vector database...")

llm = ChatOllama(model=model)

# Generate multiple questions from a single question 

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five 
    diffrrent versio of the given user question to retrieve relevantdocuments from 
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user  overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newslines.
    Original question: {question}"""
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
)

# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

res = chain.invoke(input=("what is the document about?"))

print(res)