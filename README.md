## 🧠 Medical Textbook QA System using LangChain, Pinecone & OpenAI

This project enables intelligent **question-answering** on a medical textbook PDF by leveraging **LangChain**, **OpenAI embeddings**, and **Pinecone vector store**. It allows users to ask natural language queries and retrieves the most relevant answers using a powerful vector search + LLM pipeline.


### 🚀 Features

- 📄 Loads and parses a PDF medical textbook using `UnstructuredPDFLoader`.
- 🧩 Splits the text into chunks using `RecursiveCharacterTextSplitter`.
- 🧠 Creates embeddings using OpenAI’s `text-embedding-ada-002` model.
- 📦 Stores and indexes documents in **Pinecone** for fast retrieval.
- 🤖 Uses **LangChain’s RetrievalQA chain** to answer user queries based on the PDF.
- 🔍 Handles real-time questions like symptoms, disease info, medication use, and more.


### 📁 How It Works

1. **Document Loading**  
   Load the medical textbook PDF using LangChain’s `UnstructuredPDFLoader`.

2. **Text Splitting**  
   The content is split into manageable chunks using `RecursiveCharacterTextSplitter`.

3. **Embedding Generation**  
   Chunks are converted into high-dimensional vectors using `OpenAIEmbeddings`.

4. **Vector Storage**  
   The embeddings are stored in a **Pinecone index**, enabling semantic search.

5. **Retrieval QA**  
   User queries are matched against indexed vectors. The most relevant chunks are passed to an OpenAI model to generate answers.

### 🧪 Example Queries

```python
query = "Explain about the Skin Diseases?"
query = "What is the Tablet used to cure headache?"
query = "What are the Symptoms of fever?"
```

## Overview

The RetrievalQA system is built using the following key technologies:
- **OpenAI**: For generating embeddings and language models.
- **Pinecone**: For creating and managing a high-dimensional vector index.
- **Langchain**: For document loading, text splitting, and chaining the retrieval and QA processes.

The system performs the following steps:
1. Loads and preprocesses PDF documents.
2. Splits documents into smaller, manageable chunks.
3. Creates embeddings for each chunk using OpenAI's models.
4. Indexes these embeddings with Pinecone for efficient retrieval.
5. Builds a RetrievalQA chain that fetches relevant information based on user queries.

## Setup and Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/NaladalaNavya/QARetrival.git
    cd retrievalqa-openai-pinecone
    ```

2. **Set up API keys:**
    - Add your OpenAI API key and Pinecone API key to a `config.py` file or set them as environment variables.
    ```python
    # config.py
    OPENAI_API_KEY = 'your_openai_api_key'
    PINECONE_API_KEY = 'your_pinecone_api_key'
    PINECONE_API_ENVIRONMENT = 'your_pinecone_api_environment'
    PINECONE_INDEX_NAME = 'your_pinecone_index_name'
    ```

## Usage

1. **Run the script:**
    ```bash
    python main.py
    ```

2. **Example Queries:**
    - "Explain about skin diseases."
    - "What is the tablet used to cure headaches?"
    - "What are the symptoms of fever?"
