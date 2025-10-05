from typing import List, Optional

import logging
import time
import uuid

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from app.config.settings import settings


app = FastAPI(title="RAG Chat API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)


@app.middleware("http")
async def add_request_id_and_logging(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    start = time.time()
    response = await call_next(request)
    duration_ms = int((time.time() - start) * 1000)
    response.headers["x-request-id"] = request_id
    logging.getLogger("uvicorn.access").info(
        "method=%s path=%s status=%s duration_ms=%s request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        request_id,
    )
    return response


class ChatRequest(BaseModel):
    question: str
    k: int = 4


class Source(BaseModel):
    source: str
    score: Optional[float] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]


def _get_retriever(k: int = 4):
    vectorstore = Chroma(
        persist_directory=str(settings.chroma_persist_dir),
        embedding_function=OpenAIEmbeddings(model=settings.embed_model),
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using the provided context.\n"
    "If the answer cannot be found in the context, say you don't know.\n"
    "Always cite sources as a list of filenames at the end."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Question: {question}\n\nContext:\n{context}"),
])


def _format_docs(docs):
    parts = []
    for d in docs:
        metadata = d.metadata or {}
        source = metadata.get("source") or metadata.get("file_path") or "unknown"
        parts.append(f"[Source: {source}]\n{d.page_content}")
    return "\n\n".join(parts)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/config")
def config_info():
    return {
        "llm_provider": settings.llm_provider,
        "model_name": settings.model_name,
        "embed_provider": settings.embed_provider,
        "embed_model": settings.embed_model,
    }


@app.post("/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=422, detail="question must be non-empty")
    retriever = _get_retriever(k=req.k)
    if settings.llm_provider == "openai":
        llm = ChatOpenAI(model=settings.model_name)
    else:
        llm = ChatOllama(model=settings.model_name)

    # Build RAG chain
    setup_and_retrieval = RunnableParallel({
        "context": retriever | _format_docs,
        "question": RunnablePassthrough(),
    })

    chain = setup_and_retrieval | prompt | llm

    docs = retriever.get_relevant_documents(req.question)
    answer = chain.invoke({"question": req.question})

    sources = []
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source") or meta.get("file_path") or "unknown"
        sources.append(Source(source=str(src)))

    return ChatResponse(answer=answer.content, sources=sources)





