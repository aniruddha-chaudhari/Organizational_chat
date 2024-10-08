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
from langchain.callbacks.base import BaseCallbackHandler

# Initialize response text in session state
if "response_text" not in st.session_state:
    st.session_state.response_text = ""

# Create streaming callback handler
class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        if self.container is not None:
            try:
                self.container.markdown(self.text + "▌")
            except Exception as e:
                print(f"Error updating container: {e}")

# Main chat interface
if st.session_state.processing_complete:
    retriever = st.session_state.vectors.as_retriever()
    
    user_prompt = st.text_input("Ask a question")

    if user_prompt:
        start = time.time()
        
        # Create a placeholder for the streaming response
        response_placeholder = st.empty()
        
        # Create streaming callback handler
        stream_handler = StreamingCallbackHandler(response_placeholder)
        
        # Configure LLM with streaming
        llm = OllamaLLM(
            model="llama3.1",
            temperature=0.7,
            callbacks=[stream_handler]
        )
        
        # Create the document chain with streaming
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        try:
            # Invoke the chain
            response = retrieval_chain.invoke({
                "input": user_prompt
            })
            
            # Store the final response
            st.session_state.response_text = stream_handler.text
            
            response_time = time.time() - start
            st.write(f"Response time: {response_time:.2f} seconds")

            with st.expander("Document similarity search"):
                if "context" in response:
                    for i, doc in enumerate(response["context"]):
                        st.write(f"Relevant Document {i+1}:")
                        st.write(doc.page_content)
                        st.write("--------------")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")