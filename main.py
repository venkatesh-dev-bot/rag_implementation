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
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from datetime import datetime
import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import queue
from threading import Thread
import shutil
import streamlit.components.v1 as components


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
        doc_name = st.sidebar.text_input(
            "Enter a name for this document",
            value=input_data.name if hasattr(input_data, 'name') else "Untitled Document"
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        doc_id = f"{doc_name}_{timestamp}"
        
        if input_type == "Text":
            return input_data if isinstance(input_data, str) else None, doc_id
        
        if isinstance(input_data, UploadedFile):
            file_bytes = BytesIO(input_data.read())
            
            if input_type == "PDF":
                return read_pdf(file_bytes), doc_id
            elif input_type == "DOCX":
                return read_docx(file_bytes), doc_id
            elif input_type == "TXT":
                return read_txt(input_data), doc_id
        
        return None, None
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return None, None

def process_text(text, doc_id):
    try:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(text)
        
        metadatas = [{"source": doc_id, "chunk": i} for i in range(len(texts))]
        
        embeddings = OpenAIEmbeddings()
        
        document_search = FAISS.from_texts(
            texts, 
            embeddings, 
            metadatas=metadatas
        )
        
        # Check vector store size and optimize if needed
        optimize_vector_store(document_search)
        save_vectorstore(document_search, doc_id)
        
        return document_search, texts
    except Exception as e:
        st.error(f"Error processing text: {str(e)}")
        return None, None

def get_answer(document_search, query):
    try:
        # Create ChatOpenAI instance
        llm = ChatOpenAI(
            model_name="gpt-4o-mini", 
            temperature=0,  # Set to 0 for consistent outputs
            streaming=True, 
            verbose=True,
            seed=42
        )
        
        # Get relevant documents with more context
        docs = document_search.similarity_search(
            query,
            k=4,  # Increased from default to get more context
            search_kwargs={"k": 4, "random_state": 42}  # Add seed to similarity search
        )
        
        # Create prompt template with more explicit instructions
        prompt_template = """
        variables for this chain are: "Context", "Question"
 
        System: Act as a precise and consistent teaching assistant. You must:
        1. Base your answers strictly on the provided context
        2. If the exact answer isn't in the context, say "I cannot find a direct answer to this question in the provided context."
        3. Maintain consistency in your responses for identical questions
        4. Provide clear citations from the context when possible
        
        # Don't's -
            1. Do not halucinate on your own.
        
        # Answer Formatting: 
            1. Format your response in HTML with appropriate tags:
            - Use <p> for paragraphs
            - Use <strong> for bold text
            - Use <ul> and <li> for bullet points
            - Use <br> for line breaks
            2. For mathematical expressions:
            - Use $...$ for inline math
            - Use $$...$$ for display math
            3. Use <h3> for subheadings
            4. Ensure proper spacing between elements
            5. Do not use markdown formatting
        
        # Below are the different steps to consider before generating a response:
        
        1. User Question/Query
        
            Case 1 - If the question is present in the "Context":
                Step 1 - Answer the question in 300 words.
                Step 2 - Provide real-world examples related to the question if required based on question.
                Step 3 - Check if the teacher have any other questions or queries.
        
            Case 2 - If the question is not related to the "Context":
                Step 1 - Respond saying "the question doesn't seem related to the chapter".
        
            Case 3 - If the question is related to generating FAQs:
                Step 1 - Respond saying "Refer FAQ section for generating FAQs for a topic"
        
            Case 4 - If the question is out of educational standards:
                Step 1 - Respond saying "The question doesn't seem related to the assistance I provide".
        
            Case 5 - If the "Context" provided is not related to the question or not enough to answer the question:
                Step 1 - Explore out of "Context" to answer the question only if the question is related to the topic of the context.
        
        2. If the question is unclear but the context is not empty, ask the user further information required to answer the question.
        
        3. Add flowcharts to the answer if required for the question.
        
        4. Add real-life examples to the answer based on the question to help understand the concept behind it.
        
        5. If the answer has a formula, explain the formula in detail.
        
        6. Make sure in generating answers for ideas/explanation for science experiments/projects, include precautions, requirements, and procedures for the project/experiment. Emphasize the importance of safety and explicitly instruct that the experiments should not pose any harm to students. Ensure the generated content aligns with ethical and safe educational practices.
        
        
        Context: ```{context}```
        Question: {question}
        Answer:
        """
        
        # Create prompt
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create chain using new method
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # Get answer using new invoke method
        response = document_chain.invoke({
            "context": docs,
            "question": query
        })
        
        return response
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return None

def save_vectorstore(document_search, doc_id, base_directory="vectorstore"):
    """Save the FAISS vector store to disk with document ID"""
    try:
        directory = os.path.join(base_directory, doc_id)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        document_search.save_local(directory)
        
        vectors = len(document_search.index_to_docstore_id)
        size = calculate_faiss_size(vectors)
        
        doc_info = {
            "id": doc_id,
            "date_added": datetime.now().isoformat(),
            "vectors": vectors,
            "size_mb": size['megabytes']
        }
        
        info_path = os.path.join(directory, "info.json")
        with open(info_path, 'w') as f:
            json.dump(doc_info, f)
        
        st.success(f"""Vector store saved successfully:
        - Document ID: {doc_id}
        - Vectors: {vectors:,}
        - Size: {size['megabytes']:.2f} MB""")
        return True
    except Exception as e:
        st.error(f"Error saving vector store: {str(e)}")
        return False

def list_saved_documents(base_directory="vectorstore"):
    """List all saved documents in the vector store"""
    try:
        documents = []
        if os.path.exists(base_directory):
            for doc_id in os.listdir(base_directory):
                info_path = os.path.join(base_directory, doc_id, "info.json")
                if os.path.exists(info_path):
                    with open(info_path, 'r') as f:
                        doc_info = json.load(f)
                        documents.append(doc_info)
        return documents
    except Exception as e:
        st.error(f"Error listing documents: {str(e)}")
        return []

def load_vectorstore(doc_id=None, base_directory="vectorstore"):
    """Load the FAISS vector store from disk"""
    try:
        embeddings = OpenAIEmbeddings()
        
        # If specific document ID is provided
        if doc_id:
            directory = os.path.join(base_directory, doc_id)
            if os.path.exists(directory):
                document_search = FAISS.load_local(
                    directory, 
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                st.success(f"Vector store loaded successfully for document: {doc_id}")
                return document_search
            else:
                st.warning(f"No vector store found for document: {doc_id}")
                return None
        
        # If no specific document ID, try to load the latest
        if os.path.exists(base_directory):
            # Get all subdirectories (document IDs)
            doc_dirs = [d for d in os.listdir(base_directory) 
                       if os.path.isdir(os.path.join(base_directory, d))]
            
            if not doc_dirs:
                st.warning("No vector stores found.")
                return None
            
            # Sort by modification time to get the latest
            latest_doc = max(doc_dirs, 
                           key=lambda d: os.path.getmtime(os.path.join(base_directory, d)))
            
            document_search = FAISS.load_local(
                os.path.join(base_directory, latest_doc),
                embeddings,
                allow_dangerous_deserialization=True
            )
            st.success(f"Loaded latest vector store: {latest_doc}")
            return document_search
        else:
            st.warning("No vector store directory found.")
            return None
            
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

def calculate_faiss_size(num_vectors, embedding_dim=1536):
    """
    Calculate approximate FAISS index size
    """
    bytes_per_vector = embedding_dim * 4
    total_bytes = num_vectors * bytes_per_vector
    
    size_mb = total_bytes / (1024 * 1024)
    size_gb = size_mb / 1024
    
    return {
        "bytes": total_bytes,
        "megabytes": size_mb,
        "gigabytes": size_gb
    }

def optimize_vector_store(document_search, max_vectors=100000):
    """
    Optimize vector store by limiting number of vectors
    """
    current_size = len(document_search.index_to_docstore_id)
    if current_size > max_vectors:
        st.warning(f"Vector store size ({current_size:,}) exceeds recommended limit ({max_vectors:,})")
        st.info("Consider:")
        st.info("1. Increasing chunk size")
        st.info("2. Reducing overlap")
        st.info("3. Filtering less relevant documents")

def monitor_vector_store():
    """
    Display vector store metrics in sidebar
    """
    if "document_search" in st.session_state:
        vectors = len(st.session_state.document_search.index_to_docstore_id)
        size = calculate_faiss_size(vectors)
        st.sidebar.metric("Total Vectors", f"{vectors:,}")
        st.sidebar.metric("Estimated Memory Usage", f"{size['megabytes']:.1f} MB")
        if size['gigabytes'] > 1:
            st.sidebar.warning(f"Large vector store: {size['gigabytes']:.2f} GB")

def verify_vectorstore_structure(base_directory="vectorstore"):
    """Verify and repair vector store directory structure if needed"""
    try:
        if not os.path.exists(base_directory):
            os.makedirs(base_directory)
            return
        
        # Check each document directory
        for doc_id in os.listdir(base_directory):
            doc_dir = os.path.join(base_directory, doc_id)
            if os.path.isdir(doc_dir):
                # Check for required files
                required_files = ['index.faiss', 'index.pkl', 'info.json']
                for file in required_files:
                    file_path = os.path.join(doc_dir, file)
                    if not os.path.exists(file_path):
                        st.warning(f"Missing file {file} in document {doc_id}")
                        
                # Verify info.json structure
                info_path = os.path.join(doc_dir, 'info.json')
                if os.path.exists(info_path):
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                        required_keys = ['id', 'date_added', 'vectors', 'size_mb']
                        for key in required_keys:
                            if key not in info:
                                st.warning(f"Missing key {key} in info.json for document {doc_id}")
                
    except Exception as e:
        st.error(f"Error verifying vector store structure: {str(e)}")

def delete_vectorstore(doc_id, base_directory="vectorstore"):
    """Delete a saved vector store and its associated files"""
    try:
        directory = os.path.join(base_directory, doc_id)
        
        # Check if directory exists
        if not os.path.exists(directory):
            st.error(f"Document {doc_id} not found")
            return False
        
        # Create confirmation dialog
        if st.session_state.get(f'confirm_delete_{doc_id}') is None:
            st.session_state[f'confirm_delete_{doc_id}'] = False
            
        if not st.session_state[f'confirm_delete_{doc_id}']:
            # Show confirmation dialog
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Yes, delete {doc_id}", key=f"yes_delete_{doc_id}"):
                    st.session_state[f'confirm_delete_{doc_id}'] = True
                    st.rerun()
            with col2:
                if st.button("Cancel", key=f"cancel_delete_{doc_id}"):
                    return False
            return False
        
        # If confirmed, proceed with deletion
        shutil.rmtree(directory)
        
        # Clear the document from session state if it's currently loaded
        if "document_search" in st.session_state and \
           "current_doc_id" in st.session_state and \
           st.session_state["current_doc_id"] == doc_id:
            del st.session_state["document_search"]
            del st.session_state["current_doc_id"]
        
        # Clear confirmation state
        del st.session_state[f'confirm_delete_{doc_id}']
        
        st.success(f"Document '{doc_id}' deleted successfully")
        return True
        
    except Exception as e:
        st.error(f"Error deleting document: {str(e)}")
        return False

def main():
    st.set_page_config(page_title="Document Q&A App", page_icon="📚", layout="wide")
    st.title("Document Q&A Application")
    
    # Initialize OpenAI API key
    api_key = initialize_environment()
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Verify vector store structure
    verify_vectorstore_structure()
    
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
                # Process the document with ID
                processed_text, doc_id = process_document(input_type, input_data)
                if processed_text and doc_id:
                    # Create vector store with document ID
                    document_search, texts = process_text(processed_text, doc_id)
                    if document_search and texts:
                        st.session_state["document_search"] = document_search
                        st.success(f"Document '{doc_id}' processed successfully! Created {len(texts)} text chunks.")
                        
                        # Show sample of processed text
                        with st.expander("View processed text sample"):
                            st.write(processed_text[:500] + "...")
                    else:
                        st.error("Failed to create vector store")
                else:
                    st.error("Failed to process document")
    
    # Add document listing to sidebar
    st.sidebar.header("Saved Documents")
    documents = list_saved_documents()
    if documents:
        for doc in documents:
            with st.sidebar.expander(f"📄 {doc['id']}"):
                st.write(f"Added: {doc['date_added']}")
                st.write(f"Vectors: {doc['vectors']:,}")
                st.write(f"Size: {doc['size_mb']:.2f} MB")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Load", key=f"load_{doc['id']}"):
                        loaded_vectorstore = load_vectorstore(doc_id=doc['id'])
                        if loaded_vectorstore:
                            st.session_state["document_search"] = loaded_vectorstore
                            st.session_state["current_doc_id"] = doc['id']
                            st.success(f"Loaded document: {doc['id']}")
                            st.rerun()
                with col2:
                    if st.button("Delete", key=f"delete_{doc['id']}"):
                        if delete_vectorstore(doc['id']):
                            st.rerun()
    else:
        st.sidebar.info("No saved documents found")
    
    # Add vector store monitoring in sidebar
    st.sidebar.header("Vector Store Metrics")
    monitor_vector_store()
    
    # Add vector store management in sidebar
    st.sidebar.header("Vector Store Management")
    if st.sidebar.button("Load Saved Vector Store"):
        loaded_vectorstore = load_vectorstore()
        if loaded_vectorstore:
            st.session_state["document_search"] = loaded_vectorstore
    
    # Add detailed vector store info display
    if "document_search" in st.session_state:
        with st.sidebar.expander("Vector Store Details"):
            st.write("Vector store is loaded and ready")
            if os.path.exists("vectorstore"):
                disk_size = sum(os.path.getsize(os.path.join("vectorstore", f)) 
                              for f in os.listdir("vectorstore"))
                st.write(f"Size on disk: {disk_size/1024/1024:.2f} MB")
                vectors = len(st.session_state.document_search.index_to_docstore_id)
                st.write(f"Number of vectors: {vectors:,}")
                
                # Show warning if vector store is large
                size = calculate_faiss_size(vectors)
                if size['gigabytes'] > 1:
                    st.warning(f"Large vector store detected: {size['gigabytes']:.2f} GB")
                    st.info("Consider optimizing by adjusting chunk size or reducing overlap")

    # Question answering
    if "document_search" in st.session_state:
        st.header("Ask Questions")
        query = st.text_input("Enter your question about the document")
        
        if st.button("Get Answer"):
            if query:
                with st.spinner("Generating answer..."):
                    answer = get_answer(st.session_state["document_search"], query)
                    if answer:
                        # Add MathJax script with better configuration
                        mathjax_script = """
                        <script>
                            window.MathJax = {
                                tex: {
                                    inlineMath: [['$', '$'], ['\\(', '\\)']],
                                    displayMath: [['$$', '$$'], ['\\[', '\\]']],
                                    processEscapes: true
                                },
                                options: {
                                    ignoreHtmlClass: 'tex2jax_ignore',
                                    processHtmlClass: 'tex2jax_process'
                                }
                            };
                        </script>
                        <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
                        <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
                        """
                        
                        # Enhanced CSS styling
                        css_style = """
                        <style>
                            .answer-container {
                                font-family: Arial, sans-serif;
                                line-height: 1.6;
                                padding: 20px;
                                max-width: 800px;
                                margin: 0 auto;
                                background-color: #ffffff;
                                border-radius: 8px;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            }
                            .answer-container h2, .answer-container h3 {
                                color: #1f77b4;
                                margin-bottom: 20px;
                                border-bottom: 2px solid #eee;
                                padding-bottom: 10px;
                            }
                            .answer-container p {
                                margin-bottom: 15px;
                                color: #333;
                                text-align: justify;
                            }
                            .answer-container ul, .answer-container ol {
                                margin-left: 20px;
                                margin-bottom: 15px;
                                color: #333;
                            }
                            .answer-container li {
                                margin-bottom: 8px;
                            }
                            .answer-container strong {
                                color: #2c3e50;
                                font-weight: 600;
                            }
                            .math-container {
                                overflow-x: auto;
                                padding: 10px;
                                margin: 10px 0;
                                background-color: #f8f9fa;
                                border-radius: 4px;
                            }
                            .answer-container br {
                                margin-bottom: 10px;
                            }
                            .answer-container code {
                                background-color: #f7f7f7;
                                padding: 2px 5px;
                                border-radius: 3px;
                                font-family: monospace;
                            }
                        </style>
                        """

                        # Process the answer to properly handle LaTeX
                        processed_answer = answer['output_text'] if isinstance(answer, dict) else str(answer)
                        
                        # Wrap the answer in a container div with proper formatting
                        formatted_answer = f'<div class="answer-container tex2jax_process">{processed_answer}</div>'
                        
                        # Combine everything
                        html_content = f"{mathjax_script}{css_style}{formatted_answer}"
                        
                        # Render HTML with increased height and scrolling
                        components.html(
                            html_content, 
                            height=800, 
                            scrolling=True
                        )
                    else:
                        st.error("No answer was generated")
            else:
                st.warning("Please enter a question")
                
if __name__ == "__main__":
    main()
