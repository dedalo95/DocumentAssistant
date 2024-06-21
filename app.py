import streamlit as st
from langchain_community.document_loaders import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import OpenAIEmbeddings
import tempfile
import os
import shutil
import time

# Setup API keys
st.sidebar.header("API Key Configuration")
openai_api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")
huggingface_api_key = st.sidebar.text_input("Enter Hugging Face API Key:", type="password")

# Model Selection
st.sidebar.header("Embedding Model Configuration")
# Disclaimer about embedding models
st.sidebar.markdown(
    """
    **Description**:
    1) The embedding model for OpenAI is *text-embedding-3-small*
    2) The embedding model for Hugging Face is *sentence-transformers/stsb-xlm-r-multilingual*
    """
)
embedding_option = st.sidebar.radio("**Select Embedding Model:**", ("OpenAI", "Hugging Face"))
st.sidebar.header("Conversation Model Configuration")
# Disclaimer about conversation models
st.sidebar.markdown(
    """
    **Description**:
    1) OpenAI model is *GPT-4o* which supports multilanguage conversation.
    2) Hugging Face model is *Mistral-7b-v0.2* which supports only English conversations. For using it you will need to be granted here: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
    """
)
llm_option = st.sidebar.radio("**Select LLM Model:**", ("OpenAI", "Hugging Face"))

# Vector Store Selection
st.sidebar.header("Vector Store Selection")
st.sidebar.markdown(
    """
    **Description**:
    1) ChromaDB is a comprehensive solution that provides database functionalities specialized for vector processing.
    2) FAISS is a library specifically optimized for similarity search between vectors. 
    """
)
vector_store_option = st.sidebar.radio("**Select Vector Store:**", ("CHROMA", "FAISS"))

# Function to get the current timestamp in milliseconds
def get_timestamp_millis():
    # Get the current timestamp in milliseconds
    timestamp_millis = int(time.time() * 1000)
    return str(timestamp_millis)

# Function to determine the file type of the uploaded file
def get_file_type(uploaded_item, estensioni_supportate=["pdf", "md", "txt"]):
    if hasattr(uploaded_item, "name"):
        file_name = uploaded_item.name
    elif isinstance(uploaded_item, str):
        file_name = uploaded_item
    else:
        raise ValueError("File type is not recognized")

    extension = file_name.split(".")[-1].lower().strip()
    if extension in estensioni_supportate:
        return extension
    else:
        raise ValueError("File format not supported")

# Function to chunk files based on their type
def chunk_file(file_path, chunk_size, chunk_overlap):
    try:
        extension = get_file_type(file_path)
        if extension == "pdf":
            # Load the PDF using PyPDFium2Loader
            loader = PyPDFium2Loader(file_path)
        else:
            return "Unsupported file type"

        # Load the pages and split into chunks
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )
        chunks = text_splitter.transform_documents(pages)
        return chunks
    except ValueError as e:
        print(f"Error: {e}")

# Function to setup the conversation model
def setup_conversation_model(store, use_openai_llm):
    # Define the prompt template
    template = """
        Using the information contained in the context, give a detailed answer to the question. Do not add any extra information. Answer using the language of the question. Context: {context}. Question: {question}
    """
    if use_openai_llm:
        # Set up the OpenAI LLM
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        llm = ChatOpenAI(temperature=0.8, model="gpt-4o", openai_api_key=openai_api_key)
    else:
        # Set up the Hugging Face LLM
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2", model_kwargs={"temperature":0.8}, huggingfacehub_api_token=huggingface_api_key)

    # Create the RetrievalQA chain
    qa_with_source = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=store.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa_with_source

# Function to generate the database
def generate_db(data, use_openai_embedding, use_chroma_store):
    if use_openai_embedding:
        # OpenAI Embedding of data
        embedding_model = "text-embedding-3-small"
        embeddings = OpenAIEmbeddings(
            model=embedding_model, openai_api_key=openai_api_key
        )
    else:
        # Hugging Face Embedding of data
        model_name = "sentence-transformers/stsb-xlm-r-multilingual"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

    # Generate Vector Store
    if use_chroma_store:
        # Create Chroma vector store
        store = Chroma.from_documents(
            data,
            embeddings,
            ids=[f"{item.metadata['source']}-{index}" for index, item in enumerate(data)],
            collection_name="collection"
        )
    else:
        # Create FAISS vector store
        store = FAISS.from_documents(data, embeddings)

    # Set up the conversation model with the vector store
    return setup_conversation_model(store, llm_option == "OpenAI")

# Function to save uploaded files to a temporary location
def save_files(files):
    temp_files = []
    for file in files:
        file_data = file.getvalue()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_data)
            temp_files.append(temp_file.name)
    return temp_files

# Function to chunk the saved files
def file_chunker(files):
    data = []
    for file in files:
        chunks = chunk_file(file, chunk_size=10000, chunk_overlap=2000)
        data.extend(chunks)
    return data

# Function to clear a directory
def clear_directory(directory):
    if os.path.exists(directory):
        try:
            shutil.rmtree(directory)
            os.makedirs(directory)
        except:
            print("NO DBs FOUND")

# Main function to run the Streamlit application
def main():
    st.title("Personal Document Assistant")
    st.markdown("""
                This is a simple demo for experiencing the power of **Retrieval Augmented Generation (RAG)**!\n
                RAG is an advanced AI model that combines retrieval mechanisms with generative capabilities to enhance text generation.\n
                It works by first retrieving relevant information from a dataset based on the input query, then using this information to generate more accurate and contextually appropriate responses.\n
                This approach leverages the strengths of both retrieval and generation, enabling the creation of highly informed and contextually rich text outputs, which are particularly useful in applications like question answering and conversational agents.
                """)
    st.header("Upload PDF")
    uploaded_files = st.file_uploader("Drag and drop PDF files here", type="pdf", accept_multiple_files=True)
    
    if 'assistant' not in st.session_state:
        st.session_state.assistant = None

    # Generate database button
    if st.button("Generate Database"):
        if uploaded_files:
            # Save and chunk uploaded files
            files = save_files(uploaded_files)
            data = file_chunker(files)
            with st.spinner("Generating database..."):
                # Generate the database
                st.session_state.assistant = generate_db(data, embedding_option == "OpenAI", vector_store_option == "CHROMA")
                st.success("Now you can chat with your data!")
        else:
            st.warning("Please upload at least one PDF file.")
    
    # Chat interface
    if st.session_state.assistant:
        st.header("Chat with your data")
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.text_input("You: ", key="input")

        if st.button("Send"):
            if user_input:
                st.session_state.chat_history.append(("User", user_input))
                if(embedding_option == "OpenAI"):
                    assistant_response = st.session_state.assistant(user_input)['result']
                else:
                    assistant_response = st.session_state.assistant(user_input)['result'].split(user_input)[-1]
                print(assistant_response)
                st.session_state.chat_history.append(("Assistant", assistant_response))
            else:
                st.warning("Please enter a message.")

        # Display chat history
        if st.session_state.chat_history:
            for role, message in st.session_state.chat_history:
                st.write(f"**{role}:** {message}")

if __name__ == "__main__":
    db_directory = "db/"
    clear_directory(db_directory)
    main()
