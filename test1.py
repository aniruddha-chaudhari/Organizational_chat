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
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

api_key = os.getenv('TAVILY_API_KEY')

# Define embeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# Generate unique directory for vector database
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
unique_directory = f"./chroma_langchain_db_/{timestamp}"
os.makedirs(unique_directory, exist_ok=True)

# Callback handler for streaming token responses
class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")

# Setup session state variables
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "vector" not in st.session_state:
    st.session_state.vector = None
if "docs" not in st.session_state:
    st.session_state.docs = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "response_queue" not in st.session_state:
    st.session_state.response_queue = Queue()

# Directory for uploaded PDFs
UPLOAD_DIRECTORY = os.path.abspath(os.path.join(os.getcwd(), "uploaded_pdfs"))
try:
    if not os.path.exists(UPLOAD_DIRECTORY):
        os.makedirs(UPLOAD_DIRECTORY)
        st.success(f"Created directory: {UPLOAD_DIRECTORY}")
except Exception as e:
    st.error(f"Failed to create directory: {e}")

# Upload and load PDF or web link
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

            # Load the PDF
            st.session_state.loader = PyPDFLoader(file_path)
            st.session_state.docs = st.session_state.loader.load()
        except Exception as e:
            st.error(f"Failed to save or load file: {e}")
    elif web_link:
        try:
            # Load web content
            st.session_state.loader = WebBaseLoader(web_link)
            st.session_state.docs = st.session_state.loader.load()
        except Exception as e:
            st.error(f"Failed to load web content: {e}")

    if st.session_state.docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
        final_documents = text_splitter.split_documents(st.session_state.docs)

        if final_documents:
            try:
                # Create Chroma index
                st.session_state.vectors = Chroma(
                    collection_name="document_collection",
                    embedding_function=embeddings,
                    persist_directory=unique_directory,
                )
                st.session_state.vectors.add_documents(final_documents)
                st.success("Successfully created index")
                loading_time = time.time() - start
                st.write(f"Loading time: {loading_time:.2f} seconds")
                st.session_state.processing_complete = True
            except Exception as e:
                st.error(f"Failed to create Chroma index: {e}")
        else:
            st.warning("No documents to process after splitting")
    else:
        st.warning(
            "No documents loaded. Please upload a PDF or provide a valid web link."
        )

# Prompt template
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

# Define response queue for streaming tokens
if "response_queue" not in st.session_state:
    st.session_state.response_queue = Queue()

# Streamlit app title and user input
st.title("Chatbot")
user_prompt = st.text_input("Ask a question")

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

    search_tool = TavilySearchResults()

    # Ensure agent is properly initialized
    if st.session_state.agent_executor is None:
        tools = [search_tool]
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Use the tools available to you to answer the user's questions."),
            ("human", "{input}"),
            ("ai", "Certainly! I'd be happy to help. Let me think about how to approach this."),
            ("human", "If you need to search for information, use the search tool provided."),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        st.session_state.agent = create_openai_functions_agent(llm, tools, prompt)
        st.session_state.agent_executor = AgentExecutor(
            agent=st.session_state.agent,
            tools=tools,
            memory=st.session_state.memory,
            verbose=True
        )

    # Function to process the response in the background
    def process_response():
        try:
            response = st.session_state.agent_executor.invoke({
                "input": user_prompt
            })
            st.session_state.response_queue.put(None)
            return response
        except Exception as e:
            st.error(f"Error invoking agent_executor: {e}")

    # Start processing the response in a background thread
    thread = Thread(target=process_response)
    thread.start()

    response_text = ""
    while True:
        if not st.session_state.response_queue.empty():
            token = st.session_state.response_queue.get()
            if token is None:
                break
            response_text += token
            response_placeholder.markdown(response_text + "▌")
        time.sleep(0.01)

    # Final response display
    response_placeholder.markdown(response_text)
    response_time = time.time() - start
    st.write(f"Response time: {response_time:.2f} seconds")
