import os
import chromadb
from groq import Groq
from groq import APIStatusError
from dotenv import load_dotenv
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.embeddings import resolve_embed_model

# Reconnect to already stored embeddings (persistent memory).
def load_index():
    # Load persistent ChromaDB
    db_client = chromadb.PersistentClient(path="./chroma_db")

    # SAFETY: get_or_create avoids silent empty collection issues
    chroma_collection = db_client.get_or_create_collection("infosys_report")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # SAME embedding model used during ingestion
    embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model
    )
    # # Add this to your script temporarily
    # nodes = list(index.docstore.docs.values())
    # print(f"Total nodes found: {len(nodes)}")
    # for node in nodes[:5]: # Print first 5 nodes to check structure
    #     print(node.metadata, node.text[:50])
    return index

# Fetch only relevant sections, not entire documents.
def retrieve_context(index, query, top_k=5):
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    return nodes

# Prevent hallucination and force grounded answers
def build_prompt(query, nodes):
    context = "\n\n".join(
        [f"[Section: {node.metadata.get('section','')}] {node.text}"
         for node in nodes]
    )

    prompt = f"""
You are a financial analyst AI.

Based on the context below, provide a clear, high-level summary.
Do NOT say "Not found" unless the topic is completely absent."

Context:
{context}

Question:
{query}

Answer:
"""
    return prompt

# use Groq’s hosted Llama-3 model as the generation layer, passing a strictly grounded prompt built from retrieved financial sections. 
# Low temperature ensures deterministic, audit-safe answers.
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(prompt):
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a finance expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return completion.choices[0].message.content.strip()

    except APIStatusError as e:
        # Handle token overflow / request too large
        if e.status_code == 413 or "tokens" in str(e).lower():
            return (
                "The request is too broad and exceeds the context limit.\n"
                "Please ask a more specific question or request a shorter summary."
            )

        # Generic API error
        return "An error occurred while generating the answer. Please try again."

# Glue Everything Together (Final Pipeline)
def answer_question(query: str) -> str:
    index = load_index()
    nodes = retrieve_context(index, query)

    # CRITICAL: guard against empty retrieval (prevents blank answers)
    if not nodes:
        return "Not found in the report."

    prompt = build_prompt(query, nodes)
    answer = generate_answer(prompt)
    return str(answer)

if __name__ == "__main__":
    while True:
        q = input("\nAsk a financial question (or 'exit'): ")
        if q.lower() == "exit":
            break
        print("\nAnswer:\n", answer_question(q))
