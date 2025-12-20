# test_retrieval.py
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.embeddings import resolve_embed_model
import chromadb

# Load the index
db_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db_client.get_or_create_collection("infosys_report")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model
)

# Create retriever
retriever = index.as_retriever(similarity_top_k=5)

# Test query
query = "what's in the balance sheet"
nodes = retriever.retrieve(query)

print(f"\n🔍 RETRIEVAL TEST for: '{query}'")
print(f"Found {len(nodes)} chunks\n")

for i, node in enumerate(nodes, 1):
    print(f"--- Chunk {i} (Score: {node.score:.3f}) ---")
    print(f"Metadata: {node.metadata}")
    print(f"Content: {node.text[:10000]}...")
    print()