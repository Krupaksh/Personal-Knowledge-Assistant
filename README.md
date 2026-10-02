# Personal Knowledge Assistant

A local Retrieval-Augmented Generation (RAG) based Personal Knowledge Assistant that processes PDF documents, retrieves relevant information using vector search, and generates answers using a locally running Large Language Model (LLM).

## Overview

The Personal Knowledge Assistant allows users to ask questions about their PDF documents and receive context-based answers with source document and page references.

The project uses Docling for document processing, Sentence Transformers for generating embeddings, FAISS for similarity search, and Ollama to run the Qwen3 language model locally.

## Features

- **PDF Processing:** Extracts structured content from PDF documents using Docling.
- **Document Chunking:** Splits documents into smaller, meaningful chunks using HybridChunker.
- **Text Embeddings:** Converts text into numerical embeddings using BAAI/bge-small-en-v1.5.
- **Vector Search:** Uses FAISS to retrieve relevant document chunks.
- **Local LLM:** Generates answers using Qwen3 8B through Ollama.
- **Source References:** Displays source PDF filenames and page numbers for retrieved content.
- **Privacy-focused:** Processes queries and generates answers locally.

## Tech Stack

- Python 3.13.5
- Docling 2.125.0
- Sentence Transformers 6.1.0
- BAAI/bge-small-en-v1.5
- FAISS CPU 1.15.1
- Ollama 0.6.2
- Qwen3 8B

## Project Structure

```text
personal-knowledge-assistant/
├── Data/
│   └── Documents/
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

PDF documents are stored locally in `Data/Documents/` and are not included in this repository.

## Prerequisites

- Python 3.13.5
- Ollama installed
- Qwen3 8B model downloaded through Ollama

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR-USERNAME/personal-knowledge-assistant.git
   ```

2. Navigate to the project directory:

   ```bash
   cd personal-knowledge-assistant
   ```

3. (Optional) Create and activate a virtual environment.

4. Install the required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

5. Download the Qwen3 8B model:

   ```bash
   ollama pull qwen3:8b
   ```

6. Place your PDF documents in the `Data/Documents/` folder.

## Usage

Run the application:

```bash
python main.py
```

Enter a question when prompted. The assistant retrieves relevant document chunks and generates an answer based on the retrieved context.

Type `exit` to quit the application.

## How It Works

1. **Document Conversion:** Docling extracts structured content from PDF files.
2. **Chunking:** HybridChunker splits the extracted content into manageable chunks.
3. **Embedding:** BAAI/bge-small-en-v1.5 converts chunks into vector embeddings.
4. **Indexing:** FAISS stores the embeddings for similarity search.
5. **Retrieval:** The user's question is embedded, and the most relevant chunks are retrieved.
6. **Answer Generation:** Qwen3 uses the retrieved context to generate a response.
7. **Source Display:** The application displays the source filenames and page numbers associated with retrieved chunks.

## Current Limitations

- Supports PDF files only.
- Runs as a command-line application.
- Rebuilds the FAISS index when the application starts.
- Retrieves the top three chunks for each question.
- Answer accuracy depends on the quality and relevance of the retrieved context.

## License

A license has not yet been selected for this project.
