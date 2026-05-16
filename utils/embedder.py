import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def create_vector_store(pdf_folder="data/sample_pdfs"):
    texts = []
    
    # 1. Extract text from all PDFs
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            path = os.path.join(pdf_folder, filename)
            text = extract_text_from_pdf(path)
            # Better chunking
            chunks = [text[i:i+500] for i in range(0, len(text), 400)]
            texts.extend(chunks)
    
    if not texts:
        print("No PDFs found!")
        return

    # 2. Create embeddings (once, after all texts collected)
    print(f"Creating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, convert_to_numpy=True)

    # 3. Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # 4. Save
    with open("faiss_index.pkl", "wb") as f:
        pickle.dump((index, texts), f)

    print(f"✅ Vector store created successfully with {len(texts)} chunks")