# RAG Document Q&A Bot

A local Retrieval-Augmented Generation (RAG) application that lets users ask questions about their documents and inspect the retrieved evidence behind each answer.

Built with Python, Streamlit, Sentence Transformers, FAISS, and Ollama.

## Features

- **Document ingestion:** Upload PDF and TXT files through the Streamlit interface.
- **Text chunking:** Split extracted document text into manageable chunks while retaining page and section metadata where available.
- **Semantic search:** Generate embeddings with Sentence Transformers and retrieve relevant passages using FAISS.
- **Hybrid retrieval logic:** Combine numbered clause lookup, heading matching, keyword matching, and semantic retrieval.
- **Local LLM inference:** Generate answers through a locally hosted Ollama model.
- **Evidence and citations:** Display retrieved excerpts, source filenames, page information, section metadata, and similarity scores.
- **Document management:** Index documents, remove indexed documents, and restrict questions to selected sources.
- **Persistent vector index:** Save the FAISS index and associated chunk metadata locally.
- **Grounded responses:** Instruct the model to answer from retrieved excerpts and abstain when the requested information cannot be found.

## Architecture

```text
                 User
                  |
                  v
          Streamlit Interface
                  |
          Upload / Ask Question
                  |
                  v
          Document Processing
          - PDF / TXT extraction
          - Text chunking
          - Page / section metadata
                  |
                  v
          Sentence Transformers
             Embeddings
                  |
                  v
           FAISS Vector Store
          + Retrieval Strategies
                  |
                  v
            Retrieved Evidence
                  |
                  v
             Ollama LLM
                  |
                  v
       Answer + Source References
```

The numbered-clause, named-heading, and keyword retrieval paths supplement semantic search for document queries that require more precise matching.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| User interface | Streamlit |
| Document parsing | pypdf, python-docx |
| Embeddings | Sentence Transformers |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector search | FAISS CPU |
| Local LLM runtime | Ollama |
| Default generation model | `qwen2.5:3b` |
| Configuration | python-dotenv |

## Project Structure

```text
rag-doc-qa-bot/
├── app.py                  # Streamlit application
├── chat.py                 # Command-line chat interface
├── check_setup.py          # Basic setup check
├── requirements.txt
├── .env.example            # Example configuration
├── data/
│   ├── uploads/            # Local uploaded documents
│   └── index/              # Persisted vector index and metadata
└── src/
    ├── chunking.py         # Text chunking and section metadata
    ├── config.py           # Application configuration
    ├── embeddings.py       # Embedding model
    ├── llm.py              # Ollama integration
    ├── loaders.py          # PDF, TXT and DOCX loaders
    ├── logger.py           # Logging configuration
    ├── pipeline.py         # End-to-end RAG pipeline
    ├── prompts.py          # Grounding instructions and context formatting
    ├── retriever.py        # Semantic retrieval and threshold
    └── vector_store.py     # FAISS storage and persistence
```

The Streamlit upload interface currently accepts PDF and TXT files. Although the document loader supports DOCX, that format is not yet enabled in the interface.

## Getting Started

### Prerequisites

- Python 3.10 or later
- [Ollama](https://ollama.com/) installed and running
- Internet access on the first run to download Python packages and the embedding model

### 1. Clone the repository

```bash
git clone https://github.com/Anshus9639/rag-doc-qa-bot.git
cd rag-doc-qa-bot
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, consult the Python virtual-environment documentation for your system's policy settings.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the application

Create your local environment file from the example:

```powershell
Copy-Item .env.example .env
```

The example configuration uses:

```dotenv
UPLOAD_DIR=data/uploads
INDEX_DIR=data/index
LOG_LEVEL=INFO
CHUNK_SIZE=800
CHUNK_OVERLAP=150
TOP_K=4
MIN_SCORE=0.30
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
OLLAMA_MODEL=qwen2.5:3b
MAX_UPLOAD_MB=20
OLLAMA_HOST=http://localhost:11434
```

Adjust these values in `.env` if needed. Do not commit `.env` or private documents to GitHub.

### 5. Download the generation model

With Ollama running, execute:

```powershell
ollama pull qwen2.5:3b
```

The model can run locally, but generation speed and answer quality depend on your hardware and model configuration.

### 6. Start the application

```powershell
python -m streamlit run app.py --server.fileWatcherType none
```

Open the local URL printed in your terminal by Streamlit.

## How to Use

1. Upload a PDF or TXT file in the sidebar.
2. Select **Index uploaded documents**.
3. Wait for indexing to finish.
4. Ask a question in the chat box.
5. Expand **View sources and evidence** to inspect retrieved excerpts and source metadata.
6. Use the sidebar to restrict questions to selected documents or remove a document from the index.

The application stores uploaded files and the generated vector index in local data directories by default. These directories are excluded from the repository's tracked project contents.

## Limitations

- Answers depend on document text extraction, chunking, retrieval quality, and the selected language model.
- Scanned or image-only PDFs may require OCR; OCR is not implemented in the current loader.
- Retrieved citations identify evidence chunks but are not a guarantee that every statement is fully supported by the cited text.
- The current numeric-claim check is a heuristic, not a complete hallucination detector.
- Model performance, memory use, and response time vary by hardware.
- The application is a document-assistance tool, not a substitute for professional legal advice.

## Planned Improvements

- Automated unit and integration tests
- A repeatable RAG evaluation dataset with expected answers and source references
- Retrieval metrics such as Recall@k
- Answer correctness, citation support, and refusal-quality evaluation
- More robust citation validation
- Improved setup documentation and reproducible dependency versions

## Privacy

The default configuration uses a local Ollama server and local document and index storage. The first run may download software packages, embedding model files, and the Ollama generation model. Review your configuration and dependencies before using confidential documents.

## License

No license has been added yet. Specify a license before permitting reuse or redistribution of this project.
