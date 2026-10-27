# RAG Document Q&A Bot

[![Tests](https://github.com/Anshus9639/rag-doc-qa-bot/actions/workflows/tests.yml/badge.svg)](https://github.com/Anshus9639/rag-doc-qa-bot/actions/workflows/tests.yml)

A **fully local** Retrieval-Augmented Generation (RAG) app for asking questions about your documents. Upload PDFs or text files, ask in plain English, and get an answer together with the exact source file, page, section and excerpt it came from. If the answer isn't in the documents, the bot says so instead of guessing.

No API keys, no cloud services: embeddings, vector search and the LLM all run on your own machine.

Built with Python, Streamlit, Sentence Transformers, FAISS and Ollama.

## Features

- **Multi-document library:** upload several files, search all of them or restrict a question to selected ones, and remove documents at any time.
- **Page- and section-aware chunking:** every chunk remembers its file, page and section heading, and a chunk never spans two headings.
- **Hybrid retrieval:** named-heading lookup, direct clause/section lookup (`Summarize section 3`, `Articles 4.6 and 7.1`), keyword matching and semantic (FAISS) search, tried in that order.
- **Grounded answers with citations:** the local LLM answers only from retrieved excerpts, and the citation list is built from stored metadata (file, page, section, snippet), not from what the model claims.
- **Anti-hallucination layers:** similarity threshold, a strict prompt that tells the model to refuse, and a numeric check that withholds answers containing figures absent from the sources (see [Limitations](#limitations)).
- **Persistent index:** the FAISS index and chunk metadata are saved to disk, so documents are embedded only once.
- **Two interfaces:** a Streamlit web app and a terminal chat.
- **Engineering practices:** `.env` configuration, input validation, error handling, logging, 18 automated tests, CI on every push, and a reproducible evaluation harness.

## How it works

```text
 INDEXING (once per document)
 ┌────────┐   ┌──────────┐   ┌───────────┐   ┌────────────┐   ┌───────────────┐
 │ Upload │──▶│  Loader  │──▶│  Chunker  │──▶│ Embeddings │──▶│  FAISS index  │
 │PDF/TXT │   │ text +   │   │ heading-  │   │ MiniLM-L6  │   │ + chunk       │
 │        │   │ page no. │   │ aware     │   │ (384-dim)  │   │ metadata      │
 └────────┘   └──────────┘   └───────────┘   └────────────┘   └───────┬───────┘
                                                                      │ saved to disk
 ANSWERING (per question)                                             │
 ┌──────────┐   ┌─────────────────────────────────────────────┐       │
 │ Question │──▶│ 1. named heading   2. clause/section number │◀──────┘
 └──────────┘   │ 3. keyword match   4. semantic top-k + min  │
                │    (each route is tried only if the         │
                │     previous one found nothing)             │
                └───────────────────┬─────────────────────────┘
                                    ▼
                       ┌────────────────────────┐
                       │ Numbered excerpts +    │
                       │ strict grounding prompt│
                       └───────────┬────────────┘
                                   ▼
                       ┌────────────────────────┐
                       │ Local LLM (Ollama)     │
                       └───────────┬────────────┘
                                   ▼
                  ┌─────────────────────────────────────┐
                  │ Checks: refusal? numbers supported? │
                  │ Citations built from chunk metadata │
                  └───────────────────┬─────────────────┘
                                      ▼
                         Answer + file / page / section
```

**Retrieval routes**

| # | Route | Used when | Example question |
|---|---|---|---|
| 1 | Named heading | the question shares two or more informative words with a section heading | "What does the offer and acceptance clause say?" |
| 2 | Clause / section number | the question names a number | "Summarize section 3", "Explain Article 5.9" |
| 3 | Keyword | at least two shared terms and at least 50% lexical overlap with a chunk | "Is a minimum volume of work guaranteed?" |
| 4 | Semantic | none of the above matched; the best similarity must reach `MIN_SCORE` | "When can the agreement be ended?" |

If route 4 finds nothing above `MIN_SCORE`, the bot refuses **without calling the LLM**.

## Technology stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| UI | Streamlit |
| Document parsing | pypdf, python-docx |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector search | FAISS (CPU, exact inner-product search) |
| LLM runtime | Ollama (default model `qwen2.5:3b`) |
| Configuration | python-dotenv |
| Testing / CI | pytest, GitHub Actions |

## Project structure

```text
rag-doc-qa-bot/
├── app.py                     # Streamlit web app
├── chat.py                    # Terminal chat interface
├── evaluate_multidoc.py       # Evaluation: synthetic multi-document benchmark
├── evaluate_retrieval.py      # Evaluation: retrieval (page/section targets)
├── evaluate_rag.py            # Evaluation: answers, citations, refusals
├── requirements.txt
├── requirements-dev.txt       # adds pytest
├── .env.example               # configuration template
├── .github/workflows/tests.yml
├── src/
│   ├── config.py              # settings loaded from .env
│   ├── logger.py              # logging setup
│   ├── loaders.py             # PDF / TXT / DOCX -> text + page numbers
│   ├── chunking.py            # heading-aware chunking with metadata
│   ├── embeddings.py          # sentence-transformers wrapper
│   ├── vector_store.py        # FAISS index: add, search, remove, save, load
│   ├── retriever.py           # semantic retrieval + similarity threshold
│   ├── prompts.py             # grounding prompt and context formatting
│   ├── llm.py                 # Ollama client and error handling
│   └── pipeline.py            # retrieval routes, verification, citations
├── tests/                     # pytest suite (18 tests)
├── evals/
│   ├── cases.json             # answer cases for a real contract (PDF not included)
│   ├── multidoc_cases.json    # cases for the synthetic benchmark
│   └── fixtures/              # 8 synthetic policy documents
└── data/
    ├── uploads/               # uploaded files (git-ignored)
    └── index/                 # saved FAISS index + metadata (git-ignored)
```

## Getting started

### Prerequisites

- Python 3.10 or later
- [Ollama](https://ollama.com/download), installed and running
- Internet access on first run (Python packages, the embedding model and the Ollama model are downloaded once)
- About 4 GB of free RAM for the default 3B-parameter model

### Install

```bash
git clone https://github.com/Anshus9639/rag-doc-qa-bot.git
cd rag-doc-qa-bot
python -m venv venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source venv/bin/activate
```

Then install dependencies, create your config and download the model:

```bash
pip install -r requirements.txt
cp .env.example .env          # Windows PowerShell: Copy-Item .env.example .env
ollama pull qwen2.5:3b
```

### Run the web app

```bash
python -m streamlit run app.py --server.fileWatcherType none
```

1. Upload one or more PDF/TXT files in the sidebar and click **Index uploaded documents**.
2. Ask a question in the chat box.
3. Open **View sources and evidence** under an answer to see the file, page, section and excerpt used.
4. Use the sidebar to restrict questions to specific documents or to remove a document.

### Run the terminal chat

```bash
python chat.py path/to/document.pdf
```

Commands inside the chat: `/add <path>`, `/docs`, `/remove <file name>`, `/help`, `/quit`.

## Configuration

All settings live in `.env` (see `.env.example`).

| Variable | Default | Meaning |
|---|---|---|
| `CHUNK_SIZE` | 800 | Target chunk length in characters |
| `CHUNK_OVERLAP` | 150 | Characters shared between neighbouring chunks |
| `TOP_K` | 4 | Chunks retrieved by semantic search |
| `MIN_SCORE` | 0.30 | Minimum cosine similarity for semantic retrieval; below it the bot refuses without calling the LLM |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model (changing it requires re-indexing) |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Generation model |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `MAX_UPLOAD_MB` | 20 | Maximum file size accepted by the loader |
| `UPLOAD_DIR`, `INDEX_DIR` | `data/uploads`, `data/index` | Storage locations |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

If you change `CHUNK_SIZE`, `CHUNK_OVERLAP` or `EMBEDDING_MODEL`, delete `data/index/*` and re-index your documents.

## Testing and evaluation

### Automated tests

```bash
pip install -r requirements-dev.txt
python -m pytest -v
```

The 18 tests cover loading, chunking, the vector store (replace, remove, save/load), refusal behaviour, the numeric check, citation mapping and the evaluation datasets. The LLM is mocked, so no Ollama is needed. The same suite runs in GitHub Actions on every push and pull request.

### Evaluation scripts

These run against a real Ollama model, so they check actual answer behaviour rather than mocks.

| Script | What it checks | Needs |
|---|---|---|
| `python evaluate_multidoc.py` | Retrieval and answers across 8 synthetic policy documents, including questions that must be refused | Ollama only (uses an isolated in-memory index and leaves `data/index` untouched) |
| `python evaluate_retrieval.py` | Whether the right page and section is retrieved for a real contract | The contract PDF indexed in the app |
| `python evaluate_rag.py` | Answer wording, citation page/section and evidence text, plus refusals, for the same contract | The contract PDF indexed in the app |

Each answer case checks three things: the **answer** contains the required facts, the **citation** points at the expected source/page/section, and the **evidence** excerpt contains the supporting text.

### Results

Measured on 2026-10-28 with `qwen2.5:3b` and `all-MiniLM-L6-v2`:

| Evaluation | Cases | Result |
|---|---|---|
| Synthetic multi-document retrieval | 9 answerable | Hit@1 100%, Hit@3 100%, MRR 1.00 |
| Synthetic multi-document answers | 12 (9 answerable, 3 must refuse) | 12/12 answer, citation and evidence checks |
| Real contract retrieval (page/section targets) | 4 | Hit@1 100%, MRR 1.00 |
| Real contract answers | 6 (4 answerable, 2 must refuse) | 6/6 answer, citation and evidence checks |

**How to read these numbers.** The benchmarks are small and were written alongside the system, so they are regression checks and a demonstration of method, not a general accuracy claim. In particular:

- In the retrieval runs, every hit came from the named-heading or keyword routes; the semantic route was never the deciding one. The synthetic documents are also one chunk each.
- Two of the real-contract cases are close to the worked examples in the system prompt (`src/prompts.py`).
- Of the 5 must-refuse questions, one was stopped by the similarity threshold and four were refused by the LLM itself, so refusal quality depends on the model you run.

The real contract is not included in this repository, so `evals/cases.json` is skipped unless you index your own copy of that file.

## Limitations

- **Answer quality depends on the local model.** A 3B-parameter model can still misread or over-answer. Larger Ollama models (for example `llama3.1:8b`) are more reliable but slower.
- **Citations show evidence, not proof.** They identify the excerpts the answer was built from; they don't guarantee every sentence is fully supported.
- **The numeric check is a heuristic.** It withholds answers containing digits that appear nowhere in the retrieved text. It does not catch fabricated words, does not match "sixty" to "60", and a numbered list in an answer can trigger a false refusal.
- **Chat history is display-only.** The UI keeps the conversation on screen, but each question is answered independently, so follow-ups like "what about section 4?" get no context from earlier turns.
- **No OCR.** Scanned or image-only PDFs have no extractable text and are rejected with a clear message.
- **Parsing is basic.** PDF text extraction can merge or split words, DOCX tables are ignored, and heading detection is pattern-based (numbered or "Section/Article N" headings). The loader supports DOCX but the web UI currently accepts only PDF and TXT.
- **Single-user, local design.** The index is a flat in-memory FAISS index, which is exact and fast for thousands of chunks but not built for concurrent users or millions of documents.
- Not a substitute for professional legal advice.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ollama` is not recognized (Windows) | Close and reopen VS Code/terminal after installing Ollama, or run `$env:LOCALAPPDATA\Programs\Ollama\ollama.exe`. |
| `Can't reach Ollama at http://localhost:11434` | Start the Ollama app and check `OLLAMA_HOST` in `.env`. |
| `Model 'qwen2.5:3b' is not installed` | Run `ollama pull qwen2.5:3b`; the name must match `OLLAMA_MODEL`. |
| `Could not load embedding model` | The first run downloads the model; check your internet connection. |
| `Vector size ... doesn't match the index` | You changed the embedding model; delete `data/index/*` and re-index. |
| Section questions miss the right chunk after upgrading | Delete `data/index/*` and re-index so documents are chunked with the current chunker. |
| FAISS error saving the index on Windows | Move the project to a path without non-English characters. |

## Privacy

With the default configuration, documents, embeddings and the index stay on your machine, and questions are answered by a local Ollama server. Network access is only needed to download packages and models. Uploaded files are stored in `data/uploads/` and the index in `data/index/`; both are excluded from git.

## Roadmap

- Pass previous turns to the pipeline so follow-up questions work
- Semantic-only vs hybrid retrieval ablation, and evaluation fixtures with multiple chunks per document and paraphrased questions
- Enable DOCX upload in the UI and extract DOCX tables
- Optional OCR for scanned PDFs
- Pinned dependency versions and a Docker setup

## License

Released under the MIT License. See [LICENSE](LICENSE).