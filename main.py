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
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers import ParentDocumentRetriever
from dataclasses import dataclass
from typing import List, Optional, Dict


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

@dataclass
class ChatMessage:
    role: str
    content: str

# Add after existing global variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "temp_rag_prompt" not in st.session_state:
    st.session_state.temp_rag_prompt = None
if "temp_fallback_prompt" not in st.session_state:
    st.session_state.temp_fallback_prompt = None
if "is_editing_prompts" not in st.session_state:
    st.session_state.is_editing_prompts = False

def get_custom_prompt() -> Optional[str]:
    """Get custom prompt from session state or return default if none set"""
    if "custom_prompt" not in st.session_state:
        st.session_state.custom_prompt = """
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
        
        Context: ```{context}```
        Question: {question}
        Answer:
        """
    return st.session_state.custom_prompt

def get_answer(document_search, query):
    try:
        llm = ChatOpenAI(
            model_name="gpt-4o-mini", 
            temperature=0,  # Set to 0 for consistent outputs
            streaming=True, 
            verbose=True,
            seed=42
        )
        
        # Create base retriever from FAISS
        base_retriever = document_search.as_retriever(
            search_kwargs={"k": 8}  # Increased initial retrieval for better reranking
        )
        
        # Setup embeddings filter
        embeddings = OpenAIEmbeddings()
        embeddings_filter = EmbeddingsFilter(
            embeddings=embeddings, 
            similarity_threshold=0.4
        )
        
        # Create compression retriever with embeddings filter
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=embeddings_filter,
            base_retriever=base_retriever
        )
        
        # Get filtered documents
        docs = compression_retriever.get_relevant_documents(query)
        
        # # Display retrieved chunks in an expander
        # with st.expander("📑 Retrieved Chunks", expanded=False):
        #     st.markdown("### Retrieved Context Chunks")
        #     for i, doc in enumerate(docs, 1):
        #         with st.container():
        #             st.markdown(f"**Chunk {i}**")
        #             st.markdown("**Content:**")
        #             st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
        #             st.markdown("**Metadata:**")
        #             st.json(doc.metadata)
        #             st.markdown("---")
            
        #     if not docs:
        #         st.info("No relevant chunks found. Using general knowledge.")
        
        # Rest of the function remains the same...
        has_relevant_docs = len(docs) > 0 and any(doc.page_content.strip() for doc in docs)
        
        if has_relevant_docs:
            rag_prompt = PromptTemplate(
                template="""Use the following pieces of context to answer the question. If the context doesn't contain a direct answer, use your knowledge to provide a relevant response while staying on topic.

                Context: ```{context}```
                Question: {question}

                Instructions:
                1. If the answer is in the context, use that information primarily
                2. If the context is partially relevant, combine it with your knowledge
                3. If the context isn't directly relevant, provide a helpful response based on the general topic
                4. Always maintain a helpful and informative tone
                5. Format your response in HTML with appropriate tags
                
                Answer:""",
                input_variables=["context", "question"]
            )
            
            document_chain = create_stuff_documents_chain(llm, rag_prompt)
            response = document_chain.invoke({
                "context": docs,
                "question": query
            })
        else:
            fallback_prompt = PromptTemplate(
                template="""You are a helpful AI assistant. Please provide a relevant and informative response to the following question:

                Question: {question}

                Instructions:
                1. Provide a comprehensive answer based on your knowledge
                2. Use examples where appropriate
                3. Format your response in HTML with appropriate tags
                4. Be helpful and informative
                
                Answer:""",
                input_variables=["question"]
            )
            
            response = llm.invoke(fallback_prompt.format(question=query))
            response = {"output_text": response.content}

        # Process the response
        if isinstance(response, dict):
            answer_text = response["output_text"]
        else:
            answer_text = str(response)

        # Store raw answer for formatted display
        formatted_answer = answer_text

        # Store in chat history with markdown formatting
        st.session_state.chat_history.append(ChatMessage(role="user", content=query))
        st.session_state.chat_history.append(ChatMessage(
            role="assistant", 
            content=f"""
<div class="chat-answer">
{formatted_answer}
</div>
"""
        ))
        
        return {
            "output_text": formatted_answer,
            "chat_text": formatted_answer
        }
        
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
            # Show confirmation buttons side by side without columns
            if st.sidebar.button(f"Yes, delete {doc_id}", key=f"yes_delete_{doc_id}"):
                st.session_state[f'confirm_delete_{doc_id}'] = True
                st.rerun()
            if st.sidebar.button("Cancel", key=f"cancel_delete_{doc_id}"):
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

def get_prompts_directory() -> str:
    """Create and return the prompts directory path"""
    prompts_dir = "stored_prompts"
    if not os.path.exists(prompts_dir):
        os.makedirs(prompts_dir)
    return prompts_dir

def save_prompt_to_file(prompt_name: str, prompt_content: str, prompt_type: str) -> bool:
    """Save a prompt to a JSON file"""
    try:
        prompts_dir = get_prompts_directory()
        prompts_file = os.path.join(prompts_dir, "custom_prompts.json")
        
        # Load existing prompts
        prompts = load_stored_prompts()
        
        # Add new prompt
        prompts[prompt_name] = {
            "content": prompt_content,
            "type": prompt_type,
            "created_at": datetime.now().isoformat()
        }
        
        # Save to file
        with open(prompts_file, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, indent=2)
            
        return True
    except Exception as e:
        st.error(f"Error saving prompt: {str(e)}")
        return False

def load_stored_prompts() -> Dict:
    """Load all stored prompts from file"""
    try:
        prompts_file = os.path.join(get_prompts_directory(), "custom_prompts.json")
        if os.path.exists(prompts_file):
            with open(prompts_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        st.error(f"Error loading prompts: {str(e)}")
        return {}

def delete_stored_prompt(prompt_name: str) -> bool:
    """Delete a stored prompt"""
    try:
        prompts = load_stored_prompts()
        if prompt_name in prompts:
            del prompts[prompt_name]
            prompts_file = os.path.join(get_prompts_directory(), "custom_prompts.json")
            with open(prompts_file, 'w', encoding='utf-8') as f:
                json.dump(prompts, f, indent=2)
            return True
        return False
    except Exception as e:
        st.error(f"Error deleting prompt: {str(e)}")
        return False

def render_prompt_settings():
    st.sidebar.header("Prompt Settings")
    
    # Load stored prompts
    stored_prompts = load_stored_prompts()
    
    # Toggle for editing mode
    st.session_state.is_editing_prompts = st.sidebar.checkbox(
        "Edit Prompts",
        value=st.session_state.is_editing_prompts,
        key="edit_prompts_toggle"
    )

    if st.session_state.is_editing_prompts:
        # Initialize temp prompts if needed
        if st.session_state.temp_rag_prompt is None:
            st.session_state.temp_rag_prompt = get_custom_prompt()
        if st.session_state.temp_fallback_prompt is None:
            st.session_state.temp_fallback_prompt = st.session_state.get("custom_fallback_prompt", "")

        with st.sidebar.expander("RAG Prompt Template", expanded=True):
            # Prompt name input
            rag_prompt_name = st.text_input(
                "Prompt Name",
                value="",
                key="rag_prompt_name",
                help="Enter a name to save this prompt"
            )
            
            # Edit in temporary state
            st.session_state.temp_rag_prompt = st.text_area(
                "Edit RAG Prompt",
                value=st.session_state.temp_rag_prompt,
                height=300,
                help="Use {context} and {question} as placeholders",
                key="rag_prompt_editor"
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Save RAG", key="save_rag"):
                    st.session_state.custom_prompt = st.session_state.temp_rag_prompt
                    if rag_prompt_name:
                        if save_prompt_to_file(rag_prompt_name, st.session_state.temp_rag_prompt, "rag"):
                            st.success(f"✅ Saved as '{rag_prompt_name}'!")
                    else:
                        st.warning("Please enter a name to save the prompt")
                    
            with col2:
                if st.button("Reset RAG", key="reset_rag"):
                    st.session_state.temp_rag_prompt = get_custom_prompt()
                    st.session_state.custom_prompt = get_custom_prompt()
                    st.success("🔄 Reset!")
                    
            with col3:
                if st.button("Load Saved", key="load_rag"):
                    st.session_state.show_rag_prompts = True

        # Similar structure for fallback prompt...
        
        # Show stored prompts in a separate expander
        with st.sidebar.expander("Stored Prompts", expanded=False):
            st.subheader("Saved RAG Prompts")
            rag_prompts = {k: v for k, v in stored_prompts.items() if v["type"] == "rag"}
            
            for name, prompt_data in rag_prompts.items():
                with st.container():
                    st.write(f"**{name}**")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Load", key=f"load_{name}"):
                            st.session_state.temp_rag_prompt = prompt_data["content"]
                            st.session_state.custom_prompt = prompt_data["content"]
                            st.success(f"Loaded prompt: {name}")
                    with col2:
                        if st.button("Delete", key=f"delete_{name}"):
                            if delete_stored_prompt(name):
                                st.success(f"Deleted prompt: {name}")
                                st.rerun()
                    st.markdown("---")

            st.subheader("Saved Fallback Prompts")
            fallback_prompts = {k: v for k, v in stored_prompts.items() if v["type"] == "fallback"}
            # Similar structure for fallback prompts...

    # Show current active prompts when not editing
    else:
        with st.sidebar.expander("View Active Prompts"):
            st.markdown("**Current RAG Prompt:**")
            st.code(get_custom_prompt(), language="text")
            st.markdown("**Current Fallback Prompt:**")
            st.code(st.session_state.get("custom_fallback_prompt", "Default fallback prompt"), language="text")

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
    input_type = st.sidebar.selectbox("Select Input Type", ["PDF", "Text", "DOCX", "TXT"])
    
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
                # st.write(f"Added: {doc['date_added']}")
                # st.write(f"Vectors: {doc['vectors']:,}")
                # st.write(f"Size: {doc['size_mb']:.2f} MB")
                
                if st.button("Load", key=f"load_{doc['id']}"):
                    loaded_vectorstore = load_vectorstore(doc_id=doc['id'])
                    if loaded_vectorstore:
                        st.session_state["document_search"] = loaded_vectorstore
                        st.session_state["current_doc_id"] = doc['id']
                        st.success(f"Loaded document: {doc['id']}")
                        st.rerun()
                        
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

    # Replace the old prompt customization section with:
    render_prompt_settings()

    # Add this CSS for chat messages
    st.markdown("""
    <style>
    .chat-answer {
        font-family: Arial, sans-serif;
        line-height: 1.6;
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 4px;
        margin: 5px 0;
    }
    .chat-answer p {
        margin-bottom: 10px;
    }
    .chat-answer ul, .chat-answer ol {
        margin-left: 20px;
        margin-bottom: 10px;
    }
    .chat-answer li {
        margin-bottom: 5px;
    }
    .chat-answer strong {
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

    # Question answering
    if "document_search" in st.session_state:
        st.header("Ask Questions")
        
        # Display chat history with MathJax support
        for msg in st.session_state.chat_history:
            with st.chat_message(msg.role):
                # Add MathJax support for chat messages
                if msg.role == "assistant":
                    st.markdown(f"""
                    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
                    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
                    {msg.content}
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(msg.content)
        
        # Query input using chat input
        query = st.chat_input("Enter your question about the document")
        
        if query:
            if st.session_state.get("document_search"):
                # Create a placeholder for the sidebar
                sidebar_placeholder = st.sidebar.empty()
                
                # Hide/disable the sidebar by replacing it with empty content
                with sidebar_placeholder:
                    st.empty()
                
                # Show spinner and generate answer
                with st.spinner("Generating answer..."):
                    answer = get_answer(st.session_state["document_search"], query)
                    if answer:
                        # Add MathJax script and CSS (keep your existing styling)
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

                        responsive_wrapper = """
                        <script>
                        function updateIframeHeight() {
                            // Get viewport height
                            const vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0);
                            // Calculate desired height (70% of viewport height, minimum 400px)
                            const desiredHeight = Math.max(vh * 0.7, 400);
                            // Get the iframe element
                            const iframe = document.querySelector('iframe');
                            if (iframe) {
                                iframe.style.height = desiredHeight + 'px';
                            }
                        }

                        // Update height on load and resize
                        window.addEventListener('load', updateIframeHeight);
                        window.addEventListener('resize', updateIframeHeight);
                        </script>
                        """

                        # Process the answer to properly handle LaTeX
                        processed_answer = answer['output_text'] if isinstance(answer, dict) else str(answer)
                        
                        # Wrap the answer in a container div with proper formatting
                        formatted_answer = f'<div class="answer-container tex2jax_process">{processed_answer}</div>'
                        
                        # Combine everything
                        html_content = f"{mathjax_script}{css_style}{formatted_answer}{responsive_wrapper}"
                        
                        # Render HTML with increased height and scrolling
                        components.html(
                            html_content,
                            height=600, 
                            scrolling=True
                        )
                    else:
                        st.error("No answer was generated")
                
                # Restore the sidebar after answer generation
                sidebar_placeholder.empty()
            else:
                st.warning("Please enter a question")
                
if __name__ == "__main__":
    main()
