from pathlib import Path

import streamlit as st

from src.pipeline import RAGPipeline


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide",
)

st.title("RAG Document Q&A")
st.caption(
    "Ask questions about your documents and inspect "
    "the sources used to generate each answer."
)


# --------------------------------------------------
# Load the existing RAG pipeline
# --------------------------------------------------

@st.cache_resource
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


try:
    pipeline = get_pipeline()
except Exception as error:
    st.error(f"Could not load the RAG pipeline: {error}")
    st.stop()


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


def citation_to_dict(citation) -> dict:
    """Convert a Citation object into session-safe data."""
    return {
        "number": citation.number,
        "source": citation.source,
        "page": citation.page,
        "section": citation.section,
        "score": citation.score,
        "snippet": citation.snippet,
    }


def display_sources(citations: list[dict]) -> None:
    """Display source metadata and the retrieved text."""
    if not citations:
        return

    with st.expander("View sources and evidence"):
        for citation in citations:
            location = citation["source"]

            if citation["page"] is not None:
                location += f", page {citation['page']}"

            if citation["section"]:
                location += f", {citation['section']}"

            st.write(
                f"**[{citation['number']}] {location}**"
            )
            st.caption(
                f"Retrieval similarity: {citation['score']:.3f}"
            )

            if citation["snippet"]:
                st.text(citation["snippet"])


# --------------------------------------------------
# Sidebar: upload and manage documents
# --------------------------------------------------

with st.sidebar:
    st.header("Document library")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Uploaded files are saved in data/uploads.",
    )

    if st.button(
        "Index uploaded documents",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    ):
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        for uploaded_file in uploaded_files:
            # Strip directory components from the supplied filename.
            filename = Path(uploaded_file.name).name
            destination = UPLOAD_DIR / filename

            try:
                destination.write_bytes(uploaded_file.getvalue())

                with st.spinner(f"Indexing {filename}..."):
                    chunk_count = pipeline.ingest_file(destination)

                st.success(
                    f"{filename}: indexed {chunk_count} chunks."
                )

            except Exception as error:
                st.error(f"Could not index {filename}: {error}")

        st.rerun()

    st.divider()
    st.subheader("Indexed documents")

    documents = pipeline.documents()
    document_names = list(documents.keys())

    if documents:
        for name, chunk_count in documents.items():
            st.write(f"**{name}**")
            st.caption(f"{chunk_count} indexed chunks")

        document_to_remove = st.selectbox(
            "Choose a document to remove",
            options=document_names,
            index=None,
            placeholder="Select a document",
        )

        if st.button(
            "Remove selected document",
            disabled=document_to_remove is None,
            use_container_width=True,
        ):
            try:
                removed_count = pipeline.remove_document(
                    document_to_remove
                )

                if removed_count:
                    st.success(
                        f"Removed {document_to_remove} "
                        f"({removed_count} chunks)."
                    )
                else:
                    st.warning("No matching indexed chunks were removed.")

                st.rerun()

            except Exception as error:
                st.error(f"Could not remove document: {error}")

        st.divider()

        selected_documents = st.multiselect(
            "Restrict questions to",
            options=document_names,
            default=[],
            help=(
                "Leave empty to search every indexed document. "
                "Otherwise, only the selected documents are searched."
            ),
        )

        source_filter = (
            set(selected_documents)
            if selected_documents
            else None
        )

    else:
        st.info("Upload and index a document to get started.")
        source_filter = None

    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------
# Main area: conversation history
# --------------------------------------------------

if not pipeline.documents():
    st.info(
        "Start by uploading a PDF or TXT document in the sidebar."
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            display_sources(message.get("citations", []))


# --------------------------------------------------
# Main area: ask questions
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about your uploaded documents..."
)

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents and generating an answer..."):
                result = pipeline.ask(
                    question,
                    sources=source_filter,
                )

            st.markdown(result.text)

            citations = [
                citation_to_dict(citation)
                for citation in result.citations
            ]

            display_sources(citations)

            if result.note:
                with st.expander("Retrieval diagnostics"):
                    st.write(f"Reason: {result.note}")
                    st.write(
                        f"Best retrieval score: {result.best_score:.3f}"
                    )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result.text,
                    "citations": citations,
                }
            )

        except Exception as error:
            error_message = (
                f"Sorry, the question could not be processed: {error}"
            )

            st.error(error_message)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "citations": [],
                }
            )