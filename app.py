import streamlit as st
import os
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
import ollama

logging.basicConfig(level=logging.INFO)

DOC_PATH = "./data/compass.pdf"
MODEL_NAME = "llama3.1"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DICTIONARY = "./chroma_db"

def ingest_pdf(doc_path):
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info("PDF loaded successfully.")
        return data
    else:
        logging.error(f"PDF file not found at path: {doc_path}")
        st.error("PDF file not found.")
        return None


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks

@st.cache_resource
def load_llm():
    return ChatOllama(model=MODEL_NAME)    

@st.cache_resource
def load_embedding():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)

@st.cache_resource
def load_vector_db():
    embedding = load_embedding()

    if os.path.exists(PERSIST_DICTIONARY):
        return Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DICTIONARY,
        )
    
    data = ingest_pdf(DOC_PATH)
    if data is None:
        return None
    
    chunks = split_documents(data)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        collection_name=VECTOR_STORE_NAME,
        persist_directory=PERSIST_DICTIONARY
    )
    vector_db.persist()

    return vector_db

def create_retriever(vector_db, llm):
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
    logging.info("Retriever created.")
    return retriever

def create_chain(retriever, llm):

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

    logging.info("Chain created with preserved syntax.")
    return chain

def main():
    st.title("Document Assistant")

    user_input = st.text_input("Enter your question:", "")

    if user_input:
        with st.spinner("Generating response..."):
            try:
                llm = load_llm()
                vector_db = load_vector_db()
                if vector_db is None:
                    st.error("Failed to load or create the vector database.")
                    return
                
                retriever = create_retriever(vector_db, llm)
                chain = create_chain(retriever, llm)
                response = chain.invoke(input=user_input)

                st.markdown("**Asistant:**")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            
    else:
        st.info("Please enter a question to get started.")

if __name__ == "__main__":
    main()