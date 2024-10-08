import streamlit as st
import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_ollama.llms import OllamaLLM


from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")


UPLOAD_DIRECTORY = os.path.abspath(os.path.join(os.getcwd(), "uploaded_pdfs"))
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "vector" not in st.session_state:
    st.session_state.vector = None
if "docs" not in st.session_state:
    st.session_state.docs = []
if "docs" not in st.session_state:
    st.session_state.docs = []

if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False

try:
    if not os.path.exists(UPLOAD_DIRECTORY):
        os.makedirs(UPLOAD_DIRECTORY)
        st.success(f"Created directory: {UPLOAD_DIRECTORY}")
except Exception as e:
    st.error(f"Failed to create directory: {e}")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
web_link = st.text_input("Enter a web link")

if not st.session_state.processing_complete and (uploaded_file or web_link):
    start = time.time()
    st.session_state.docs = []

    if uploaded_file:
        file_path = os.path.join(UPLOAD_DIRECTORY, uploaded_file.name)
        try:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File saved at {file_path}")

            st.session_state.loader = PyPDFLoader(file_path)
            st.session_state.docs = st.session_state.loader.load()
        except Exception as e:
            st.error(f"Failed to save or load file: {e}")
    elif web_link:
        try:
            st.session_state.loader = WebBaseLoader(web_link)
            st.session_state.docs = st.session_state.loader.load()
        except Exception as e:
            st.error(f"Failed to load web content: {e}")

    if st.session_state.docs:

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)

        final_documents = text_splitter.split_documents(st.session_state.docs)

        if final_documents:
            try:
                st.session_state.vectors = Chroma(
                    collection_name="document_collection",  # Collection name should be a string
                    embedding_function=embeddings,
                    persist_directory="./chroma_langchain_db",  # Chroma's persist directory
                )
                st.session_state.vectors.add_documents(final_documents)
                st.success("Successfully created index")
                loading_time = time.time() - start
                st.write(f"Loading time: {loading_time:.2f} seconds")
                st.session_state.processing_complete = True
            except Exception as e:
                st.error(f"Failed to create FAISS index: {e}")
        else:
            st.warning("No documents to process after splitting")
    else:
        st.warning(
            "No documents loaded. Please upload a PDF or provide a valid web link."
        )


st.title("Enterprise Chatbot")
llm = OllamaLLM(model="llama3.1")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Question: {input}
    """
)

if st.session_state.processing_complete:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    user_prompt = st.text_input("Ask a question")

    if user_prompt:
        start = time.time()
        response = retrieval_chain.invoke({"input": user_prompt})
        response_time = time.time() - start
        st.write(response["answer"])
        st.write(f"Response time: {response_time:.2f} seconds")

        with st.expander("Document similarity search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------")
else:
    st.warning(
        "No documents processed yet. Please upload a PDF or provide a valid web link."
    )
