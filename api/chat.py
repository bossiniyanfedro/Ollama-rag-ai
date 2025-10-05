from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from config.settings import settings


app = FastAPI(title="RAG Chat API", version="0.1.0")


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


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    retriever = _get_retriever(k=req.k)
    llm = ChatOpenAI(model=settings.model_name)

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.port)


