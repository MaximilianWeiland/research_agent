import hashlib
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config.settings import DOCS_DIR, DB_DIR
from dotenv import load_dotenv
load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-large"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def get_file_hash(file_path: Path) -> str:
    """Create a hash of a file to detect changes."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def load_documents():
    """Load PDFs and attach metadata."""
    all_docs = []

    for pdf_file in DOCS_DIR.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()

        file_hash = get_file_hash(pdf_file)

        for doc in docs:
            doc.metadata["source"] = pdf_file.name
            doc.metadata["file_hash"] = file_hash

        all_docs.extend(docs)

    return all_docs


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def build_index():
    print("Loading documents...")
    docs = load_documents()
    print(f"Loaded {len(docs)} pages")

    print("Splitting documents...")
    splits = split_documents(docs)
    print(f"Created {len(splits)} chunks")

    print("Initializing embeddings...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    print("Building vector store...")
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
    )

    vector_store.persist()
    print(f"Index saved to {DB_DIR}")


# ---- Incremental Update (Optional but Powerful) ----
def update_index():
    """
    Only re-embed changed files instead of rebuilding everything.
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    vector_store = Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
    )

    print("Loading documents...")
    docs = load_documents()

    print("Splitting documents...")
    splits = split_documents(docs)

    print("Adding documents (incremental)...")
    vector_store.add_documents(splits)
    vector_store.persist()

    print("Index updated.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["build", "update"],
        default="build",
        help="Build from scratch or update existing index",
    )

    args = parser.parse_args()

    if args.mode == "build":
        build_index()
    else:
        update_index()