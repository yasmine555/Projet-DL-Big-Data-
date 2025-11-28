
import os
import pickle
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import camelot
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from llama_index.core.schema import TextNode, Document
from llama_index.core.node_parser import SentenceWindowNodeParser
from transformers import AutoTokenizer, AutoModel
import torch
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import SimpleVectorStore
from typing import List
import pickle
from config import PDF_DIR, OUT_DIR, CHUNK_SIZE, OVERLAP_SIZE, EMBEDDING_MODEL_NAME 

def extract_text_and_tables(pdf_path):
    
    doc = fitz.open(pdf_path)
    nodes = []
    
    print(f"Processing: {os.path.basename(pdf_path)}")
    
    for page_number, page in enumerate(doc, start=1):
        # Extract text
        text = page.get_text()
        if text.strip():
            nodes.append(text)
        
        # Extract tables
        try:
            tables = camelot.read_pdf(pdf_path, pages=str(page_number))
            for table_idx, table in enumerate(tables):
                table_text = f"\n[Table {table_idx+1} from page {page_number}]\n{table.df.to_string()}\n"
                nodes.append(table_text)
        except Exception as e:
            pass  
    
    print(f"  ✓ Extracted {len(nodes)} chunks")
    return nodes

# Process all PDFs
all_nodes = []
for file in os.listdir(PDF_DIR):
    if file.lower().endswith(".pdf"):
        file_path = os.path.join(PDF_DIR, file)
        nodes = extract_text_and_tables(file_path)
        all_nodes.extend(nodes)

print(f"\nTotal: {len(all_nodes)} text/table chunks extracted")

# Convert to Documents
documents = [Document(text=n) for n in all_nodes]
print(f"Created {len(documents)} Document objects")

def chunk_nodes(documents, save_dir="chunks", sample_n=5):
    """Chunk Document objects into nodes and save results; return nodes list."""
    os.makedirs(save_dir, exist_ok=True)

    parser = SentenceWindowNodeParser(
        window_size=CHUNK_SIZE,
        overlap_size=OVERLAP_SIZE
    )

    print("Chunking documents...")
    chunks = parser.get_nodes_from_documents(documents)

    # ----- Save chunks as pickle -----
    chunk_pkl_path = os.path.join(save_dir, "chunks.pkl")
    with open(chunk_pkl_path, "wb") as f:
        pickle.dump(chunks, f)

    # ----- Save readable text for inspection -----
    chunk_txt_path = os.path.join(save_dir, "chunks.txt")
    with open(chunk_txt_path, "w", encoding="utf-8") as f:
        for i, node in enumerate(chunks):
            text = node.get_content() if hasattr(node, "get_content") else str(node)
            f.write(f"[CHUNK {i}]\n{text}\n\n{'='*60}\n\n")

    # show sample
    for i, node in enumerate(chunks[:sample_n]):
        text = node.get_content() if hasattr(node, "get_content") else str(node)
        print(f"--- sample chunk {i} ---\n{text[:400].replace('\\n',' ')}\n")

    return chunks

def mean_pooling(output, mask):
    embeddings = output[0]  # First element contains all token embeddings
    mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
    return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

def build_documents_and_index():
    # Process all PDFs
    all_nodes = []
    if not os.path.isdir(PDF_DIR):
        print(f"PDF_DIR does not exist: {PDF_DIR}")
        return

    for file in sorted(os.listdir(PDF_DIR)):
        if file.lower().endswith(".pdf"):
            file_path = os.path.join(PDF_DIR, file)
            nodes = extract_text_and_tables(file_path)
            all_nodes.extend(nodes)

    print(f"\nTotal: {len(all_nodes)} text/table chunks extracted")

    # Convert to Documents
    documents = [Document(text=n) for n in all_nodes]
    print(f"Created {len(documents)} Document objects")

    # Prepare chunk directory and try to load existing chunks if present
    chunk_dir = os.path.join(OUT_DIR, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_pkl_path = os.path.join(chunk_dir, "chunks.pkl")

    if os.path.exists(chunk_pkl_path):
        print(f"Loading existing chunks from {chunk_pkl_path}")
        with open(chunk_pkl_path, "rb") as f:
            chunk_nodes_list = pickle.load(f)
    else:
        # create chunks and save them
        chunk_nodes_list = chunk_nodes(documents, save_dir=chunk_dir)

    if not chunk_nodes_list:
        print("No chunks available to embed. Exiting.")
        return

    # Load model and compute embeddings for each chunk
    print("Loading PubMedBERT for embeddings...")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"✓ Model loaded on {device}")

    # Create TextNode list with embeddings
    nodes_with_embeddings = []
    print("\nGenerating embeddings for chunks...")
    for i, chunk in enumerate(tqdm(chunk_nodes_list, desc="Embedding chunks")):
        text = chunk.get_content() if hasattr(chunk, "get_content") else str(chunk)
        inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
        with torch.no_grad():
            output = model(**inputs)
        embedding = mean_pooling(output, inputs['attention_mask']).cpu().squeeze().tolist()

        if i == 0:
            print(f"\n✓ First embedding generated with PubMedBERT:")
            print(f"  - Length: {len(embedding)} dimensions")
            print(f"  - Sample values: {embedding[:5]}")

        node = TextNode(text=text, embedding=embedding, id_=f"node_{i}")
        nodes_with_embeddings.append(node)

    # Build and save index
    print("\nBuilding VectorStoreIndex...")
    from llama_index.core import Settings
    Settings.embed_model = None

    vector_store = SimpleVectorStore()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    for node in nodes_with_embeddings:
        vector_store.add([node])

    index = VectorStoreIndex(nodes=[], storage_context=storage_context, embed_model=None)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "vector_index.pkl"), "wb") as f:
        pickle.dump(index, f)
    with open(os.path.join(OUT_DIR, "nodes.pkl"), "wb") as f:
        pickle.dump(nodes_with_embeddings, f)

    print(f"\n✓ DONE! Index and nodes saved to {OUT_DIR}")
    return nodes_with_embeddings

def main():
    print("Starting document processing pipeline...")
    build_documents_and_index()
    print("Processing finished.")


if __name__ == "__main__":
    main()