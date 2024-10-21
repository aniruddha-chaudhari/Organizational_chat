from queue import Queue
from threading import Thread
import streamlit as st
import os
from dotenv import load_dotenv
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_ollama.llms import OllamaLLM
from langchain_community.tools import TavilySearchResults

from langchain_core.callbacks.base import BaseCallbackHandler
from datetime import datetime
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

api_key = os.getenv('API_KEY')



embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
unique_directory = f"./chroma_langchain_db_/{timestamp}"
os.makedirs(unique_directory, exist_ok=True)

class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")

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

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)

        final_documents = text_splitter.split_documents(st.session_state.docs)

        if final_documents:
            try:
                st.session_state.vectors = Chroma(
                    collection_name="document_collection",  # Collection name should be a string
                    embedding_function=embeddings,
                    persist_directory=unique_directory,  # Chroma's persist directory
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



st.title("Chatbot")
# def get_llm(callback_handler):
#     return OllamaLLM(
#         model="llama3.1",
#         temperature=0.7,
#         callbacks=[callback_handler]
#     )

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful AI assistant that provides detailed and comprehensive answers based on the given context.
    Always aim to provide thorough explanations and relevant details from the context.
    
    <context>
    {context}
    </context>

    Please answer the following question in detail, using specific information from the context above.
    If the context doesn't contain enough information to answer fully, please state that clearly.
    
    Question: {input}
    
    Instructions for answering:
    1. Provide a complete answer using multiple sentences
    2. Include specific details and examples from the context
    3. Use proper paragraph structure
    4. Maintain a natural, conversational tone
    """
)


if "response_queue" not in st.session_state:
    st.session_state.response_queue = Queue()

# Create streaming callback handler
class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, queue):
        self.queue = queue
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.queue.put(token)





st.title("Chatbot")
user_prompt = st.text_input("Ask a question")

# Check if document processing is complete
if st.session_state.processing_complete:
    retriever = st.session_state.vectors.as_retriever()
    
    if user_prompt:
        start = time.time()

        response_placeholder = st.empty()
        current_response = ""

        # Clear any previous responses in the queue
        while not st.session_state.response_queue.empty():
            st.session_state.response_queue.get()

        # Create streaming handler
        stream_handler = StreamingCallbackHandler(st.session_state.response_queue)

        # Define LLM with callback handler
        llm = OllamaLLM(
            model="llama3.2:3b-instruct-fp16",
            temperature=0.7,
            callbacks=[stream_handler]
        )

        # Create the document chain and retrieval chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response_container = st.empty()

        # Start processing the response in a background thread
        def process_response():
            response = retrieval_chain.invoke({
                "input": user_prompt
            })
            st.session_state.response_queue.put(None)
            return response

        thread = Thread(target=process_response)
        thread.start()

        response_text = ""
        while True:
            if not st.session_state.response_queue.empty():
                token = st.session_state.response_queue.get()
                if token is None:
                    break
                response_text += token
                response_container.markdown(response_text + "▌")
            time.sleep(0.01)

        # Final response display
        response_container.markdown(response_text)
        response_time = time.time() - start
        st.write(f"Response time: {response_time:.2f} seconds")
else:
    if user_prompt:
        st.warning("Please wait for the documents to be processed and indexed before asking questions.")

if st.session_state.processing_complete:
    document_chain = None
    retriever = st.session_state.vectors.as_retriever()
    
    user_prompt = st.text_input("Ask a question")

    if user_prompt:
        start = time.time()
        
        response_placeholder = st.empty()
        current_response = ""

        while not st.session_state.response_queue.empty():
            st.session_state.response_queue.get()
        
        stream_handler = StreamingCallbackHandler(st.session_state.response_queue)
        
        llm = OllamaLLM(
            model="llama3.1",
            temperature=0.7,
            callbacks=[stream_handler]
        )
        
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        response_container = st.empty()
        
        def process_response():
            response = retrieval_chain.invoke({
                "input": user_prompt
            })
            st.session_state.response_queue.put(None)
            return response
        
        thread = Thread(target=process_response)
        thread.start()
        
        response_text = ""
        while True:
            if not st.session_state.response_queue.empty():
                token = st.session_state.response_queue.get()
                if token is None:  # End of response
                    break
                response_text += token
                response_container.markdown(response_text + "▌")
            time.sleep(0.01)  
        
       
        response_container.markdown(response_text)
        
        response_time = time.time() - start
        st.write(f"Response time: {response_time:.2f} seconds")

        with st.expander("Document similarity search"):
            for i, doc in enumerate(thread.response["context"]):
                st.write(f"Relevant Document {i+1}:")
                st.write(doc.page_content)
                st.write("--------------")

