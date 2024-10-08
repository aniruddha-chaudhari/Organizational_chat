import os
import time
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_ollama.llms import OllamaLLM
from datetime import datetime
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# Create a unique directory for storing the Chroma database
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
unique_directory = f"./chroma_langchain_db_/{timestamp}"
os.makedirs(unique_directory, exist_ok=True)

# Initialize session state variables
if "processing_complete" not in locals():
    processing_complete = False
if "vector" not in locals():
    vector = None
if "docs" not in locals():
    docs = []

# Get user input for the web link
web_link = input("Enter a web link: ")

if not processing_complete and web_link:
    start = time.time()
    docs = []

    # Load documents from the provided web link
    try:
        loader = WebBaseLoader(web_link)
        docs = loader.load()
        print(f"Loaded {len(docs)} documents from the web link.")
    except Exception as e:
        print(f"Failed to load web content: {e}")

    if docs:
        # Split documents into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
        final_documents = text_splitter.split_documents(docs)

        if final_documents:
            try:
                # Initialize Chroma database and add documents
                vectors = Chroma(
                    collection_name="document_collection",
                    embedding_function=embeddings,
                    persist_directory=unique_directory,
                )
                vectors.add_documents(final_documents)
                print("Successfully created index")
                loading_time = time.time() - start
                print(f"Loading time: {loading_time:.2f} seconds")
                processing_complete = True
            except Exception as e:
                print(f"Failed to create Chroma index: {e}")
        else:
            print("No documents to process after splitting.")
    else:
        print("No documents loaded. Please provide a valid web link.")

# Initialize the chatbot model
llm = OllamaLLM(
    model="llama3.1",
    temperature=0.7,
    streaming=True  # Enable streaming
)

# Set up the prompt for the AI assistant
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

if processing_complete:
    # Create retrieval and document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Get user prompt for querying the chatbot
    user_prompt = input("Ask a question: ")

    if user_prompt:
        start = time.time()
        response = retrieval_chain.invoke({"input": user_prompt})
        response_time = time.time() - start
        print(response["answer"])
        print(f"Response time: {response_time:.2f} seconds")

        # Display similar documents
        print("\nDocument similarity search results:")
        for i, doc in enumerate(response["context"]):
            print(doc.page_content)
            print("--------------")
else:
    print("No documents processed yet. Please provide a valid web link.")
