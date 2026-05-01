from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from ragas.run_config import RunConfig
import os
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
# -------------------------------
# Initialize models
# -------------------------------
llm = ChatOllama(model="llama3.2")

eval_llm = ChatOllama(
    model="gemma3:1b",   # (Or "llama3.2:1b" if you decided to use the faster one)
    temperature=0,
    format="json"
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# -------------------------------
# PDF Processing
# -------------------------------
def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunked_docs = text_splitter.split_documents(raw_docs)

    if not os.path.exists("./chroma_db"):
        vectorstore = Chroma.from_documents(
            documents=chunked_docs,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        vectorstore.persist()
    else:
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )

    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    bm25_retriever = BM25Retriever.from_documents(chunked_docs)
    bm25_retriever.k = 3

    return chroma_retriever, bm25_retriever


# -------------------------------
# Hybrid Search
# -------------------------------
def weighted_hybrid_search(query, bm25_retriever, chroma_retriever, bm25_weight=0.4, chroma_weight=0.6, top_k=3):

    bm25_results = bm25_retriever.invoke(query)
    chroma_results = chroma_retriever.invoke(query)

    combined_scores = {}

    for i, doc in enumerate(bm25_results[:top_k]):
        combined_scores[doc.page_content] = combined_scores.get(doc.page_content, 0) + bm25_weight * (top_k - i)

    for i, doc in enumerate(chroma_results[:top_k]):
        combined_scores[doc.page_content] = combined_scores.get(doc.page_content, 0) + chroma_weight * (top_k - i)

    sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    final_docs = [
        doc for doc in bm25_results + chroma_results
        if doc.page_content in dict(sorted_docs[:top_k])
    ]

    return final_docs[:top_k]


# -------------------------------
# Query Transformation (UNCHANGED)
# -------------------------------
def transform_query(question: str, llm, history: list):

    history.append(HumanMessage(content=question))

    system_instruction = """
    You are a query rewriting expert.
    Based on the provided chat history, rephrase the current 'Follow-Up Question' into a complete, standalone question that can be understood without any prior context.
    Only output the rewritten question in very short and nothing else.
    """

    messages = [SystemMessage(content=system_instruction)] + history

    response = llm.invoke(messages)

    history.pop()

    return response.content.strip()


# -------------------------------
# Chat Function (Streamlit-friendly version)
# -------------------------------
def chatting(user_problem: str, history: list, bm25_retriever, chroma_retriever):
    standalone_question = transform_query(user_problem, llm, history)

    results5 = weighted_hybrid_search(
        standalone_question,
        bm25_retriever=bm25_retriever,
        chroma_retriever=chroma_retriever,
        bm25_weight=0.4,
        chroma_weight=0.6,
        top_k=3
    )

    # Extract raw text for Ragas evaluation
    context_texts = [doc.page_content for doc in results5]

    context_parts = []
    for i, doc in enumerate(results5):
        page = doc.metadata.get("page", "Unknown")
        context_parts.append(f"[Source {i+1}: Page {page+1}] {doc.page_content}")

    context = "\n\n".join(context_parts)

    system_instruction = f"""
You are a helpful PDF assistant trained to answer questions using the provided document only.
Instructions:
- Base your answers strictly on the given context.
- If the answer is missing or irrelevant, say:
  "I could not find the answer in the provided document."
- Always include citations for every fact you mention.
  Use the format: [Source X: Page Y]
- Keep your answer short, factual, and clear.
Context from the document:
{context}
"""

    history.append(SystemMessage(content=system_instruction))
    history.append(HumanMessage(content=user_problem))

    response = llm.invoke(history)
    answer = response.content.strip()

    history.append(AIMessage(content=answer))

    # RETURN CONTEXT TEXTS NOW
    return answer, standalone_question, context_texts

# -------------------------------
# Ragas Evaluation Function
# -------------------------------
def evaluate_response(question: str, answer: str, contexts: list):
    """
    Evaluates a single chat interaction using Ragas reference-free metrics.
    """
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts]
    }
    dataset = Dataset.from_dict(data)

    # Configuration to stop local Ollama from timing out
    run_config = RunConfig(
        timeout=300,
        max_workers=1
    )

    # Wrap your LangChain local models so Ragas understands them
    ragas_llm = LangchainLLMWrapper(eval_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=ragas_llm,                  # Use the wrapped LLM
        embeddings=ragas_embeddings,    # Use the wrapped Embeddings
        run_config=run_config,
        raise_exceptions=False
    )

    return result.to_pandas()
