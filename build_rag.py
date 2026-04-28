import pandas as pd
import chromadb
import openai
import os
from dotenv import load_dotenv
import tempfile

load_dotenv()
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# ── Load Excel ─────────────────────────────────────────────
print("Loading parameter baseline...")
df = pd.read_excel(
    r'C:\Users\SujitR\Desktop\rfai\5G_BL_public.xlsx',
    usecols=[
        'Full Name', 'Abbreviated Name', 'MO Class',
        'Description', 'Operator recommended value',
        'Range and step', 'Default Value',
        'Related Functions', 'Related Parameters',
        'Comments', 'Parameter Category'
    ]
)

print(f"Loaded {len(df)} parameters")

# ── Build text chunks ──────────────────────────────────────
def build_chunk(row):
    parts = []

    if pd.notna(row['Full Name']):
        parts.append(f"Parameter: {row['Full Name']}")

    if pd.notna(row['Abbreviated Name']):
        parts.append(f"Abbreviated name: {row['Abbreviated Name']}")

    if pd.notna(row['MO Class']):
        parts.append(f"MO Class: {row['MO Class']}")

    if pd.notna(row['Description']):
        desc = str(row['Description']).strip()[:500]
        parts.append(f"Description: {desc}")

    if pd.notna(row['Operator recommended value']):
        parts.append(f"Recommended value: {row['Operator recommended value']}")

    if pd.notna(row['Range and step']):
        parts.append(f"Range: {row['Range and step']}")

    if pd.notna(row['Default Value']):
        parts.append(f"Default value: {row['Default Value']}")

    if pd.notna(row['Related Functions']):
        funcs = str(row['Related Functions']).strip()[:200]
        parts.append(f"Related functions: {funcs}")

    if pd.notna(row['Related Parameters']):
        params = str(row['Related Parameters']).strip()[:200]
        parts.append(f"Related parameters: {params}")

    if pd.notna(row['Comments']):
        comments = str(row['Comments']).strip()[:200]
        parts.append(f"Comments: {comments}")

    return '\n'.join(parts)

chunks    = []
ids       = []
metadatas = []

for i, row in df.iterrows():
    chunk = build_chunk(row)
    if len(chunk) > 50:  # skip empty rows
        chunks.append(chunk)
        ids.append(f"param_{i}")
        metadatas.append({
            'full_name':   str(row['Full Name'])        if pd.notna(row['Full Name'])        else '',
            'abbrev_name': str(row['Abbreviated Name']) if pd.notna(row['Abbreviated Name']) else '',
            'mo_class':    str(row['MO Class'])         if pd.notna(row['MO Class'])         else '',
            'rec_value':   str(row['Operator recommended value']) if pd.notna(row['Operator recommended value']) else 'N/A',
            'range':       str(row['Range and step'])   if pd.notna(row['Range and step'])   else '',
            'category':    str(row['Parameter Category']) if pd.notna(row['Parameter Category']) else '',
        })

print(f"Built {len(chunks)} chunks")

# ── Create embeddings ──────────────────────────────────────
print("Creating embeddings — this takes 1-2 minutes...")

def get_embeddings_batch(texts, batch_size=100):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(embeddings)
        print(f"  Embedded {min(i+batch_size, len(texts))}/{len(texts)} parameters...")
    return all_embeddings

embeddings = get_embeddings_batch(chunks)
print(f"Created {len(embeddings)} embeddings")

# ── Store in ChromaDB ──────────────────────────────────────
print("Storing in ChromaDB...")

chroma_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chroma_db')
os.makedirs(chroma_path, exist_ok=True)
print(f"ChromaDB path: {chroma_path}")

chroma_client = chromadb.PersistentClient(path=chroma_path)

# Delete existing collection if rebuilding
try:
    chroma_client.delete_collection("telecom_params")
except:
    pass

collection = chroma_client.create_collection(
    name="telecom_params",
    metadata={"hnsw:space": "cosine"}
)

# Add in batches
batch_size = 100
for i in range(0, len(chunks), batch_size):
    collection.add(
        documents=  chunks[i:i+batch_size],
        embeddings= embeddings[i:i+batch_size],
        ids=        ids[i:i+batch_size],
        metadatas=  metadatas[i:i+batch_size],
    )
    print(f"  Stored {min(i+batch_size, len(chunks))}/{len(chunks)}...")

print(f"\nDone! {collection.count()} parameters indexed in ChromaDB")
print("Vector database saved to: chroma_db/")
print("\nNow add the RAG endpoint to network_health_api.py")