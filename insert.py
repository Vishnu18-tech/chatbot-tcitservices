from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

# Load embedding model (384 dims)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

# Config

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80   # 🔥 Added overlap to prevent sentence cutting
UPSERT_BATCH = 40
# Improved chunk function
def chunk_text(text, size=400, overlap=80):
    chunks = []
    start = 0

    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)

        start += size - overlap  # 🔥 move forward with overlap

    return chunks

# Read TXT file
with open("tcitservices.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Optional: basic cleaning
text = text.replace("\n", " ").strip()

# Split into chunks
chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
print(f"📄 Total chunks: {len(chunks)}")

vectors = []
# Embed + Upsert
for i, chunk in enumerate(chunks):
    embedding = embedder.encode(chunk).tolist()

    vectors.append({
        "id": f"chunk-{i}",
        "values": embedding,
        "metadata": {
            "text": chunk[:1000]  # safe metadata size
        }
    })

    # Upsert safely in batches
    if len(vectors) >= UPSERT_BATCH:
        index.upsert(vectors=vectors)
        print(f"✅ Upserted {len(vectors)} vectors")
        vectors.clear()

# Flush remaining vectors
if vectors:
    index.upsert(vectors=vectors)
    print(f"✅ Final upsert: {len(vectors)} vectors")

print("🎉 INGESTION COMPLETE")