import os  # to access the path of directory, files or any other source we want to access

from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter #which is used to split document text into smaller chunk
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

# load environment variable from .env file

load_dotenv()

working_dir = os.path.dirname(os.path.abspath((__file__)))

# Load the Embedding Model

embedding = HuggingFaceEmbeddings()

# Load the llama-3.3-70B Model from groq

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0
)

def process_document_to_chroma_db(file_name):
    # Load the pdf document using UnstructuredPDFLoader
    loader = UnstructuredPDFLoader(f"{working_dir}/{file_name}")
    documents = loader.load()
    #Split the text into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2000,
        chunk_overlap = 500
    )
    text_chunk = text_splitter.split_documents(documents)

    # Store the document chunk in a Chroma Vector Database

    vectordb = Chroma.from_documents(
        documents=text_chunk,
        embedding=embedding,
        persist_directory=f"{working_dir}/doc_vectorstore"
    )
    return 0

def answer_question(user_question):
    # load the persist chromdb
    vectordb = Chroma(
        persist_directory=f"{working_dir}/doc_vectorstore",
        embedding_function=embedding
    )

    # Create retreive from document search
    retriever = vectordb.as_retriever()

    # Create a Retreival QA chain to answer user question using llama-3.3-70B

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type = "stuff",
        retriever = retriever,
        return_source_documents = True
    )

    response = qa_chain.invoke({"query":user_question})
    answer = response["result"]
    
    return answer


