# RetrievalQA System with OpenAI and Pinecone,langchain

This repository contains the code for building a RetrievalQA system using OpenAI's language models and Pinecone's vector database. The system is designed to efficiently retrieve and answer questions based on the content of large PDF documents.

## Table of Contents
- [Overview](#overview)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Example Queries](#example-queries)
- [Contributing](#contributing)
- [License](#license)

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
    git clone https://github.com/yourusername/retrievalqa-openai-pinecone.git
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
