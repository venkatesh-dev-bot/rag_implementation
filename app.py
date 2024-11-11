import os
import streamlit as st
import faiss
from io import BytesIO
from docx import Document
import numpy as np
from langchain_community.document_loaders import WebBaseLoader
from PyPDF2 import PdfReader
from docx import Document
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from dotenv import load_dotenv
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

# Load environment variables
load_dotenv()

def initialize_environment():
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        st.error("Please set OPENAI_API_KEY in your .env file")
        st.stop()
    return openai_api_key

def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def read_docx(file):
    doc = Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def read_txt(file):
    return file.read().decode('utf-8')

def process_document(input_type, input_data):
    try:
        if input_type == "Text":
            return input_data if isinstance(input_data, str) else None
        
        if isinstance(input_data, UploadedFile):
            file_bytes = BytesIO(input_data.read())
            
            if input_type == "PDF":
                return read_pdf(file_bytes)
            elif input_type == "DOCX":
                return read_docx(file_bytes)
            elif input_type == "TXT":
                return read_txt(input_data)
        
        return None
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return None

def process_text(text):
    try:
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(text)
        
        # Create embeddings
        embeddings = OpenAIEmbeddings()
        
        # Create vector store
        document_search = FAISS.from_texts(texts, embeddings)
        
        return document_search, texts
    except Exception as e:
        st.error(f"Error processing text: {str(e)}")
        return None, None

def get_answer(document_search, query):
    try:
        # Create ChatOpenAI instance
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        
        # Create prompt template
        prompt_template = """
        You are a math expert tasked with providing precise, clear answers based on the following information. 
        Carefully consider each part of the question and ensure all relevant mathematical principles are applied correctly. 
        If you are unsure or lack information to answer fully, state, "I don't know" rather than guessing.

        Context:
        {context}

        Question:
        {question}

        Your detailed, step-by-step answer:
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create LLM chain
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        
        # Create StuffDocumentsChain
        chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context"
        )
        
        # Get relevant documents
        docs = document_search.similarity_search(query)
        
        # Get answer
        answer = chain.run(input_documents=docs, question=query)
        
        return answer
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Document Q&A App", page_icon="📚", layout="wide")
    st.title("Document Q&A Application")
    
    # Initialize OpenAI API key
    api_key = initialize_environment()
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Sidebar for file upload
    st.sidebar.title("Document Upload")
    
    # Input selection
    input_type = st.sidebar.selectbox("Select Input Type", ["Text", "PDF", "DOCX", "TXT"])
    
    # File upload or text input
    if input_type == "Text":
        input_data = st.text_area("Enter your text", height=300)
    else:
        file_types = {
            "PDF": ["pdf"],
            "DOCX": ["docx", "doc"],
            "TXT": ["txt"]
        }
        input_data = st.sidebar.file_uploader(
            f"Upload {input_type} file",
            type=file_types[input_type]
        )
    
    # Process input
    if st.sidebar.button("Process Document"):
        if input_data:
            with st.spinner("Processing document..."):
                # Process the document
                processed_text = process_document(input_type, input_data)
                if processed_text:
                    # Create vector store
                    document_search, texts = process_text(processed_text)
                    if document_search and texts:
                        st.session_state["document_search"] = document_search
                        st.success(f"Document processed successfully! Created {len(texts)} text chunks.")
                        
                        # Show sample of processed text
                        with st.expander("View processed text sample"):
                            st.write(processed_text[:500] + "...")
                    else:
                        st.error("Failed to create vector store")
                else:
                    st.error("Failed to process document")
    
    # Question answering
    if "document_search" in st.session_state:
        st.header("Ask Questions")
        query = st.text_input("Enter your question about the document")
        
        if st.button("Get Answer"):
            if query:
                with st.spinner("Generating answer..."):
                    answer = get_answer(st.session_state["document_search"], query)
                    if answer:
                        st.write("Answer:", answer)
            else:
                st.warning("Please enter a question")

if __name__ == "__main__":
    main()