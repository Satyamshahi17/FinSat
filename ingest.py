import os
import time
# from dotenv import load_dotenv
from llama_parse import LlamaParse
from pypdf import PdfReader, PdfWriter

# load_dotenv()

# --- CONFIGURATION ---
INPUT_PDF = "report.pdf"  
OUTPUT_FILE = "full_report_parsed.md"
CHUNK_SIZE = 5  # Number of pages to parse at a time (Safe number)

def split_and_parse():
    # 1. Setup Parser
    parser = LlamaParse(
        result_type="markdown",
        verbose=True,
        language="en",
        num_workers=2 # Reduced workers to be safe
    )

    # 2. Read the PDF
    try:
        reader = PdfReader(INPUT_PDF)
        total_pages = len(reader.pages)
        print(f"Found {total_pages} pages in {INPUT_PDF}. Starting batch process...")
    except FileNotFoundError:
        print("Error: report.pdf not found!")
        return

    full_text = ""
    
    # 3. Loop through chunks
    for i in range(0, total_pages, CHUNK_SIZE):
        start = i
        end = min(i + CHUNK_SIZE, total_pages)
        print(f"\nProcessing pages {start+1} to {end}...")

        # Create a temporary PDF for this chunk
        temp_pdf_name = f"temp_chunk_{start}_{end}.pdf"
        writer = PdfWriter()
        for page_num in range(start, end):
            writer.add_page(reader.pages[page_num])
        
        with open(temp_pdf_name, "wb") as f:
            writer.write(f)

        # Parse the temporary chunk
        try:
            docs = parser.load_data(temp_pdf_name)
            chunk_text = "\n\n".join([doc.text for doc in docs])
            full_text += f"\n\n--- PART {start+1}-{end} ---\n\n" + chunk_text
            print(f"Chunk {start+1}-{end} done!")
        except Exception as e:
            print(f"Error on pages {start+1}-{end}: {e}")
        
        # Cleanup temp file
        if os.path.exists(temp_pdf_name):
            os.remove(temp_pdf_name)
        
        # Sleep briefly to be nice to the API
        time.sleep(2)

    # 4. Save Final Result
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    print(f"\nDONE! Full text saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    split_and_parse()