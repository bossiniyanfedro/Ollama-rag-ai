from pathlib import Path
from typing import Sequence

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from app.config.settings import settings


def _load_documents(docs_dir: Path) -> Sequence:
    loaders = [
        DirectoryLoader(str(docs_dir), glob="**/*.md", loader_cls=TextLoader, show_progress=True),
        DirectoryLoader(str(docs_dir), glob="**/*.txt", loader_cls=TextLoader, show_progress=True),
        DirectoryLoader(str(docs_dir), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True),
    ]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    return documents


def _split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        length_function=len,
        separators=["\n\n", "\n", ". ", ".", " "]
    )
    return splitter.split_documents(documents)


def build_vectorstore():
    docs_dir = settings.docs_dir
    persist_dir = settings.chroma_persist_dir
    persist_dir.mkdir(parents=True, exist_ok=True)

    documents = _load_documents(docs_dir)
    if not documents:
        raise SystemExit(f"No documents found in {docs_dir}. Add files and retry.")

    chunks = _split_documents(documents)

    if settings.embed_provider == "openai":
        embeddings = OpenAIEmbeddings(model=settings.embed_model)
    else:
        embeddings = HuggingFaceEmbeddings(model_name=settings.embed_model)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )
    print(f"Vectorstore built with {len(chunks)} chunks at {persist_dir}")


