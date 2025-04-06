from pathlib import Path
import gradio as gr
from langchain_ollama import ChatOllama
from langchain_nomic.embeddings import NomicEmbeddings
import chromadb
from langchain_core.messages import HumanMessage
from sentence_transformers import CrossEncoder
import numpy as np
from build_vector_store import build_vector_store


local_llm_name = "gemma3:latest"
local_llm = ChatOllama(model=local_llm_name, temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define base directory relative to the project
base_dir = Path(__file__).resolve().parent
local_embedding_model = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
local_cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Define paths dynamically
persist_path = base_dir / "vectorstore"

if not persist_path.exists():
    print(f"Vector store not found at {persist_path}. Building vector store...")
    persist_path.parent.mkdir(parents=True, exist_ok=True)
    build_vector_store()  # Call the function to build the vector store
else:
    print(f"Vector store found at {persist_path}. Proceeding with the workflow.")

chroma_client = chromadb.PersistentClient(path=str(persist_path))
chroma_collection = chroma_client.get_or_create_collection("acko")

# Prompt
final_rag_prompt = """
        You are an assistant for question-answering tasks.
        Here is the context to use to answer the question:
        {context}
        Think carefully about the above context.
        Now, review the user question:
        {question}
        Provide an answer to this questions using only the above context.

        Use three sentences maximum and keep the answer concise.

        Answer:
"""

def generate_answer(
        question,
        embedding_model,
        retriever,
        llm,
        rag_prompt
):
    query_embedding = embedding_model.embed_query(question)
    results = retriever.query(
        query_embeddings=[query_embedding], n_results=10, include=["documents"]
    )
    retrieved_documents = results["documents"][0]
    docs_txt = "\n".join(retrieved_documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return generation.content

def rerank(retrieved_documents, question, cross_encoder):
    unique_documents = set()
    for documents in retrieved_documents:
        for document in documents:
            unique_documents.add(document)

    unique_documents = list(unique_documents)
    pairs = []
    for doc in unique_documents:
        pairs.append([question, doc])

    scores = cross_encoder.predict(pairs)
    top_indices = np.argsort(scores)[::-1][:5]
    top_documents = [unique_documents[i] for i in top_indices]

    # Concatenate the top documents into a single context
    context = "\n\n".join(top_documents)
    return context

def generate_answer_with_re_rank(
        question,
        embedding_model,
        retriever,
        llm,
        rag_prompt,
        cross_encoder
):
    query_embedding = embedding_model.embed_query(question)
    results = retriever.query(
        query_embeddings=[query_embedding], n_results=10, include=["documents"]
    )
    retrieved_documents = results["documents"]
    context = rerank(retrieved_documents, question, cross_encoder)
    rag_prompt_formatted = rag_prompt.format(context=context, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return generation.content

# Define the function that takes user input and returns both answers
def rag_bot(user_question, history):
    if not user_question.strip():
        return "Please enter a question."

    # Answer with Naive RAG
    naive_answer = generate_answer(
        question=user_question,
        embedding_model=local_embedding_model,
        retriever=chroma_collection,
        llm=local_llm,
        rag_prompt=final_rag_prompt,
    )

    # Answer with RAG + Re-ranking
    rerank_answer = generate_answer_with_re_rank(
        question=user_question,
        embedding_model=local_embedding_model,
        retriever=chroma_collection,
        llm=local_llm,
        rag_prompt=final_rag_prompt,
        cross_encoder=local_cross_encoder
    )

    # Format both responses
    combined_response = (
        f"**Naive RAG Answer:**\n{naive_answer}\n\n"
        f"**Re-ranked RAG Answer:**\n{rerank_answer}"
    )

    return combined_response

sample_questions = [
    "What is Insured's Declared Value (IDV)?",
    "How is No Claim Bonus (NCB) calculated?",
    "What are some general exceptions under the policy?",
    "What is the geographical scope of the policy?",
    "What is the procedure for the insured to notify the insurer in the event of an accident?",
    "What coverage is provided under the Personal Accident Cover for the Owner-Driver?",
    "What are the typical deductions for depreciation on replaced parts in the event of a claim?",
    "How is the depreciation rate calculated for replaced parts?"
]

# Launch the chatbot
gr.ChatInterface(
    fn=rag_bot,
    title="Comparison of Naive RAG and Re-ranked RAG",
    description="Ask a question and get answers using Naive RAG and Re-ranked RAG.",
    examples=sample_questions
).launch()
