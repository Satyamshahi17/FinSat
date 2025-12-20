from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core import StorageContext, VectorStoreIndex, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.embeddings import resolve_embed_model
from groq import Groq
from dotenv import load_dotenv
import chromadb
import os
import re

load_dotenv()
client = Groq()

# --- CONFIGURATION ---
INPUT_FILE = "full_report_parsed.md"

def verbalize_table(table_text: str, context: str) -> str:
    prompt = f"""
Convert the following financial table into concise natural-language sentences. 
Only restate what is explicitly present. 

Context:
{context}

Table:
{table_text}

Return only the description.
"""

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a financial analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print("Table verbalization failed:", e)
        return ""

def process_tables(md_text: str) -> str:
    pattern = '\|.+\|[\r\n]+\|[-\s:|]+\|(?:[\r\n]+\|.+\|)*'

    def replace_table(match):
        table = match.group(0)

        start = max(0, match.start() - 200)
        context = md_text[start:match.start()].strip()

        verbalized = verbalize_table(table, context)

        return f"\n{verbalized}\n\n{table}\n"

    return re.sub(pattern, replace_table, md_text)

def main():
    print("Starting Chunking Process...")
    
    # 1. Load Markdown file
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found!")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        md_text = f.read()
    
    print(f"Loaded {len(md_text)} characters from {INPUT_FILE}")

    # 2. 
    try:
        # Process tables
        print("Processing tables...")
        md_text = process_tables(md_text)
    except Exception as e:
        print(f"Table processing failed: {e}")

    # 3. Create Document
    document = Document(
        text=md_text,
        metadata={
            "filename": "report.pdf",
            "year": 2025,                
            "company": "Infosys Ltd",  
            "doc_type": "FY 2025-26 Q2 Financial Report"
        }
    )

    # 4. Initialize Parser
    parser = MarkdownNodeParser(
        include_metadata=True,      # this ensures that document-level metadata and parser-generated structural metadata (such as section headers and hierarchy) are attached to each chunk, enabling contextual retrieval.
        include_prev_next_rel=True  # Helps the AI read "previous" chunks for context
    )

    # 5. Generate Chunks
    print("Splitting text into chunks...")
    nodes = parser.get_nodes_from_documents([document])
    print(f"Created {len(nodes)} chunks!")

    # 6. VERIFICATION
    print("\n--- INSPECTING CHUNKS ---")
    for i, node in enumerate(nodes[:3]):
        print(f"\n[Chunk {i+1}]")
        print(f"Metadata: {node.metadata}")
        print(f"Content Preview: {node.text[:150]}...")
        print("-" * 50)

    # 7. SAVE TO DATABASE
    print("\nSaving to ChromaDB...")
    db_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db_client.get_or_create_collection("infosys_report")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

    index = VectorStoreIndex(
        nodes, 
        storage_context=storage_context,
        embed_model=embed_model
    )

    print("SUCCESS! Data searchable in './chroma_db'")

if __name__ == "__main__":
    main()