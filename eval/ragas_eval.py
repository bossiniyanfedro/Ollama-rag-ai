from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from app.config.settings import settings


def run_ragas_eval(samples: list[dict]):
    ds = Dataset.from_list(samples)
    vectorstore = Chroma(
        persist_directory=str(settings.chroma_persist_dir),
        embedding_function=OpenAIEmbeddings(model=settings.embed_model),
    )
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model=settings.model_name)

    result = evaluate(
        ds,
        metrics=[answer_relevancy, faithfulness],
        llm=llm,
        retriever=retriever,
    )
    print(result)
    return result


if __name__ == "__main__":
    raise SystemExit(
        "Provide your evaluation dataset (list of dict) to run_ragas_eval(samples)."
    )


