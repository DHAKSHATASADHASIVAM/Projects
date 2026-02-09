"""
This scripts created the vector databases for the three different coding documentation pdfs.
It converts them to text, splits them into chunks, and stores those chunks in a ChromaDB.
"""
import os
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader

#  import your chosen text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize ChromaDB with persistent storage
CHROMA_PATH = Path(__file__).parent / "chroma_db"
client = chromadb.PersistentClient(path=str(CHROMA_PATH))

# Use sentence-transformers for embeddings
# : Take a look at ChromaDBs embedding options on this page: https://docs.trychroma.com/docs/embeddings/embedding-functions
# Find the default option and fill in the information here:
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# PDF to collection mapping
PDF_COLLECTION_MAP = {
    "Python.pdf": "python_book",
    "Java.pdf": "java_book",
    "Javascript.pdf": "javascript_book",
    "C.pdf": "C_book"
}

#  Fill in the text splitter parameters
# This page has further information langchain's text splitters: https://docs.langchain.com/oss/python/integrations/splitters
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False
)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            # Setting errors="ignore" will ignore any invalid characters in the PDFs
            clean_text = page_text.encode("utf-8", errors="ignore").decode("utf-8")
            text += f"\n--- Page {page_num + 1} ---\n{clean_text}"

    return text


def process_pdf(pdf_path: str, collection_name: str):
    """Process a single PDF: extract text, chunk, and store in ChromaDB."""
    print(f"\nProcessing: {pdf_path}")

    # Extract text
    print("  Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)

    if not text.strip():
        print(f"  Warning: No text extracted from {pdf_path}")
        return

    # Split into chunks
    print("  Splitting into chunks...")
    #  fill in the call to the text splitter  
    chunks = text_splitter.split_text(text)
    print(f"  Created {len(chunks)} chunks")

    # Get or create collection (defaults to using L2 distance as metric for similarity search)
    collection = client.get_or_create_collection(
        name=collection_name, 
        embedding_function=embedding_fn
    )

    # Prepare data for ChromaDB
    ids = [f"{collection_name}_chunk_{i:04d}" for i in range(len(chunks))]
    metadatas = [{"source": os.path.basename(pdf_path), "chunk_index": i} for i in range(len(chunks))]

    # Store in ChromaDB (embeddings are created automatically)
    print("  Creating embeddings and storing in ChromaDB...")
    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )

    print(f"  Stored {len(chunks)} chunks in '{collection_name}' collection")


def main():
    source_dir = Path(__file__).parent.parent / "ingest_files"

    print("=" * 50)
    print("Document Ingestion Pipeline")
    print("=" * 50)
    print(f"ChromaDB storage: {CHROMA_PATH}")

    for pdf_name, collection_name in PDF_COLLECTION_MAP.items():
        pdf_path = source_dir / pdf_name

        if not pdf_path.exists():
            print(f"\nWarning: {pdf_path} not found, skipping...")
            continue

        process_pdf(str(pdf_path), collection_name)

    print("\n" + "=" * 50)
    print("Document ingestion complete!")
    print("Collections created: python_book, java_book, javascript_book")
    print("=" * 50)


if __name__ == "__main__":
    main()
