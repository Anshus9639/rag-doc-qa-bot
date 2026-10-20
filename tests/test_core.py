import numpy as np
import pytest

from src import pipeline as pipeline_module
from src import vector_store as vector_store_module
from src.chunking import Chunk, chunk_pages
from src.loaders import DocumentLoadError, DocumentPage, load_document, _normalize_pdf_text
from src.pipeline import RAGPipeline
from src.prompts import NOT_FOUND_MESSAGE, format_context
from src.retriever import RetrievalResult
from src.vector_store import VectorStore
from src.pipeline import RAGPipeline, _build_citations


def make_chunk(
    text="The contractor must obtain prior written permission from the Warden.",
    source="contract.pdf",
    page=1,
    chunk_id="contract.pdf#0",
    section="4.6 Prior written permission",
):
    return Chunk(
        text=text,
        source=source,
        page=page,
        chunk_id=chunk_id,
        section=section,
    )


def make_pipeline(chunk):
    """Build a pipeline with a small in-memory FAISS index."""
    store = VectorStore()
    store.add_chunks(
        [chunk],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.store = store
    return pipeline


# --------------------------------------------------
# Document loading
# --------------------------------------------------

def test_load_txt_preserves_text_and_filename(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("The contract begins on 1 January.", encoding="utf-8")

    pages = load_document(path)

    assert len(pages) == 1
    assert pages[0].source == "notes.txt"
    assert pages[0].page is None
    assert "contract begins" in pages[0].text


def test_loader_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "notes.csv"
    path.write_text("name,value\nitem,1", encoding="utf-8")

    with pytest.raises(DocumentLoadError, match="Unsupported file type"):
        load_document(path)


# --------------------------------------------------
# Chunking and evidence formatting
# --------------------------------------------------

def test_chunking_respects_size_and_preserves_metadata():
    text = (
        "SECTION 1 Offer and Acceptance\n\n"
        + (
            "The contractor agrees to provide qualified workers "
            "for jobs assigned by the company. "
        )
        * 15
    )

    pages = [
        DocumentPage(
            text=text,
            source="agreement.pdf",
            page=7,
        )
    ]

    chunks = chunk_pages(
        pages,
        chunk_size=120,
        overlap=20,
    )

    assert chunks
    assert all(len(chunk.text) <= 120 for chunk in chunks)
    assert all(chunk.source == "agreement.pdf" for chunk in chunks)
    assert all(chunk.page == 7 for chunk in chunks)
    assert any(
        chunk.section and "SECTION 1" in chunk.section
        for chunk in chunks
    )


def test_format_context_includes_source_location():
    chunk = make_chunk(
        text="The contractor must obtain prior written permission.",
        source="agreement.pdf",
        page=7,
        section="4.6 Prior written permission",
    )

    context = format_context([(chunk, 0.87)])

    assert "[1]" in context
    assert "Source: agreement.pdf, page 7" in context
    assert "section: 4.6 Prior written permission" in context
    assert "prior written permission" in context


# --------------------------------------------------
# FAISS vector store
# --------------------------------------------------

def test_vector_store_returns_best_match_first():
    store = VectorStore()

    chunks = [
        make_chunk(
            text="First document",
            source="first.pdf",
            chunk_id="first.pdf#0",
            section=None,
        ),
        make_chunk(
            text="Second document",
            source="second.pdf",
            chunk_id="second.pdf#0",
            section=None,
        ),
    ]

    vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )

    store.add_chunks(chunks, vectors)

    results = store.search(
        np.array([[1.0, 0.0]], dtype=np.float32),
        k=2,
    )

    assert len(results) == 2
    assert results[0][0].source == "first.pdf"
    assert results[0][1] == pytest.approx(1.0)


def test_reingesting_same_document_replaces_old_chunks():
    store = VectorStore()

    store.add_chunks(
        [make_chunk(text="Old document text")],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )

    store.add_chunks(
        [
            make_chunk(
                text="Updated document text",
                chunk_id="contract.pdf#new",
            )
        ],
        np.array([[0.0, 1.0]], dtype=np.float32),
    )

    assert store.total == 1
    assert store.chunks[0].text == "Updated document text"
    assert store.list_documents() == {"contract.pdf": 1}


def test_vector_store_can_be_saved_and_loaded(tmp_path, monkeypatch):
    monkeypatch.setattr(
        vector_store_module,
        "ensure_dirs",
        lambda: None,
    )

    store = VectorStore()
    chunk = make_chunk()

    store.add_chunks(
        [chunk],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    store.save(tmp_path)

    loaded = VectorStore.load(tmp_path)

    assert loaded.total == 1
    assert loaded.chunks == [chunk]
    assert loaded.index is not None
    assert loaded.index.ntotal == 1


# --------------------------------------------------
# RAG pipeline
# --------------------------------------------------

def test_empty_store_does_not_call_the_llm(monkeypatch):
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.store = VectorStore()

    monkeypatch.setattr(
        pipeline_module,
        "generate_answer",
        lambda messages: pytest.fail("LLM should not be called"),
    )

    result = pipeline.ask("What does this document say?")

    assert result.answered is False
    assert result.note == "empty store"
    assert "No documents are indexed" in result.text


def test_numbered_clause_question_returns_a_citation(monkeypatch):
    chunk = make_chunk(
        text=(
            "Article 4.6 requires the contractor to obtain prior "
            "written permission from the Warden before appointing "
            "a sub-service provider."
        ),
        section="4.6 Prior written permission",
    )
    pipeline = make_pipeline(chunk)

    monkeypatch.setattr(
        pipeline_module,
        "build_messages",
        lambda question, hits: [{"role": "user", "content": question}],
    )
    monkeypatch.setattr(
        pipeline_module,
        "generate_answer",
        lambda messages: (
            "The contractor must obtain prior written permission "
            "from the Warden before appointing a sub-service provider [1]."
        ),
    )

    result = pipeline.ask("What does Article 4.6 require?")

    assert result.answered is True
    assert "written permission" in result.text.lower()
    assert result.citations
    assert result.citations[0].source == "contract.pdf"


def test_pipeline_rejects_an_unsupported_number(monkeypatch):
    chunk = make_chunk()
    pipeline = make_pipeline(chunk)

    monkeypatch.setattr(
        pipeline_module,
        "build_messages",
        lambda question, hits: [{"role": "user", "content": question}],
    )
    monkeypatch.setattr(
        pipeline_module,
        "generate_answer",
        lambda messages: "The agreement requires 12 workers [1].",
    )

    result = pipeline.ask("Explain Article 4.6.")

    assert result.answered is False
    assert result.text == NOT_FOUND_MESSAGE
    assert "numbers not found" in result.note


def test_pipeline_refuses_when_retrieval_is_below_threshold(monkeypatch):
    chunk = make_chunk(
        text="Wages shall be paid by the seventh day of each month.",
        section="Payment terms",
    )
    pipeline = make_pipeline(chunk)

    monkeypatch.setattr(
        pipeline_module,
        "retrieve",
        lambda store, question, sources=None: RetrievalResult(
            question,
            [],
            0.30,
        ),
    )
    monkeypatch.setattr(
        pipeline_module,
        "generate_answer",
        lambda messages: pytest.fail("LLM should not be called"),
    )

    result = pipeline.ask(
        "What exact monthly salary does each worker receive?"
    )

    assert result.answered is False
    assert result.text == NOT_FOUND_MESSAGE
    assert result.note == "best chunk is below the similarity threshold"


def test_citations_match_explicit_answer_markers():
    first = make_chunk(
        text="The tenant must provide advance notice.",
        source="first.pdf",
        page=1,
        chunk_id="first.pdf#0",
        section="Notice",
    )

    second = make_chunk(
        text="The contractor must obtain written authorization.",
        source="second.pdf",
        page=4,
        chunk_id="second.pdf#0",
        section="4.6 Authorization",
    )

    hits = [
        (first, 0.91),
        (second, 0.82),
    ]

    citations = _build_citations(
        "The contractor must obtain written authorization [2].",
        hits,
    )

    assert len(citations) == 1
    assert citations[0].number == 2
    assert citations[0].source == "second.pdf"
    assert citations[0].page == 4
    assert citations[0].section == "4.6 Authorization"

def test_citation_snippet_focuses_on_relevant_evidence():
    boilerplate = (
        "Industries Department, Haryana. "
        "Template regarding Commercial Contracts. "
    )
    filler = "General contract template information. " * 20
    evidence = (
        "The First Party does not give guarantee any volume of work "
        "at any time during the period of agreement."
    )

    chunk = make_chunk(
        text=boilerplate + filler + evidence,
        source="agreement.pdf",
        page=3,
        section="4. RIGHTS & OBLIGATIONS",
    )

    citations = _build_citations(
        "No. The agreement does not guarantee any volume of work [1].",
        [(chunk, 0.9)],
    )

    assert len(citations) == 1
    assert "volume of work" in citations[0].snippet.lower()
    assert "does not give guarantee" in citations[0].snippet.lower()


def test_pdf_normalizer_repairs_known_split_words():
    extracted = (
        "The First Party does not give guarant ee "
        "any volume of work. "
        "T he First Party may distribute work. "
        "All payment shall be subject to the agreement."
    )

    cleaned = _normalize_pdf_text(extracted)

    assert "does not give guarantee" in cleaned
    assert "The First Party may distribute work" in cleaned
    assert "payment shall be subject" in cleaned

    assert "guarant ee" not in cleaned
    assert "T he First Party" not in cleaned