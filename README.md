# Document Assistant

This application leverages Retrieval Augmented Generation (RAG) to enhance the interaction with your personal documents. By combining retrieval mechanisms with generative capabilities, it provides more accurate and contextually appropriate responses. This approach is particularly useful in applications like question answering and conversational agents.

## Features

- **Document Upload:** Upload PDF files to create a searchable database.
- **Model Configuration:** Choose between OpenAI and Hugging Face models for embedding and conversation.
- **Vector Store Options:** Select between ChromaDB and FAISS for vector storage.
- **Chat Interface:** Interact with your documents through a conversational interface.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/dedalo95/DocumentAssistant.git
    cd DocumentAssistant
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the application:
    ```bash
    streamlit run app.py
    ```
2. Open the provided local URL in your web browser.

## Configuration

### API Key Configuration

In the sidebar, enter your API keys for OpenAI and Hugging Face:

- **OpenAI API Key**
- **Hugging Face API Key**

### Model Selection

#### Embedding Model

- **OpenAI:** `text-embedding-3-small`
- **Hugging Face:** `sentence-transformers/stsb-xlm-r-multilingual`

#### Conversation Model

- **OpenAI:** `GPT-4o` (multilanguage support)
- **Hugging Face:** `Mistral-7b-v0.1` (English only)

### Vector Store

- **ChromaDB:** Database functionalities for vector processing.
- **FAISS:** Library optimized for similarity search between vectors.

## How It Works

1. **Upload PDF Files:**
    - Drag and drop PDF files into the uploader.
    - Click "Generate Database" to process the files.

2. **Generate Database:**
    - The files are saved temporarily.
    - Each file is chunked into manageable pieces.
    - The selected embedding model processes the chunks.
    - The selected vector store saves the embeddings.

3. **Chat Interface:**
    - Once the database is ready, use the chat interface to interact with your data.
    - Type your questions and receive detailed answers based on the document content.

## Main Functions

### `get_timestamp_millis()`

Returns the current timestamp in milliseconds.

### `get_file_type(uploaded_item, estensioni_supportate=["pdf", "md", "txt"])`

Determines the file type of the uploaded item.

### `chunk_file(file_path, chunk_size, chunk_overlap)`

Chunks files based on their type.

### `setup_conversation_model(store, use_openai_llm)`

Sets up the conversation model using the selected LLM.

### `generate_db(data, use_openai_embedding, use_chroma_store)`

Generates the database with embeddings and the vector store.

### `save_files(files)`

Saves uploaded files to a temporary location.

### `file_chunker(files)`

Chunks the saved files into manageable pieces.

### `clear_directory(directory)`

Clears the specified directory.

### `main()`

Runs the Streamlit application.

## Additional Information from Notebook

### Retrieval Augmented Generation

In this notebook, data is extracted from Wikipedia or from a custom folder. A Vector DB is created (Chroma or Faiss), and ChatGPT answers questions about the topic!

#### Pre-process Data

##### Use Wikipedia as Data Source

In this example, you will download data from Wikipedia and use it for building the Knowledge Base.

##### Use Documents from a Custom Folder as Data Source

In this example, you will use the documents from the "docs" folder and create chunks from those.

#### Store Chunks into a VectorDB

##### Store Data in a ChromaDB

We will store our chunks in a Vector DB.

##### Store Data with Faiss

We will store our chunks in a Vector DB.

#### Asking Questions to the Virtual Assistant

Let's use OpenAI for answering our questions about information retrieved on Chroma!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This application utilizes several libraries and models from OpenAI, Hugging Face, and LangChain Community. Special thanks to the developers and contributors of these projects.
