import json 
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_all():
    with open("outputs/chunks.json" , "r") as f: 
        chunks = json.load(f)
    for chunk in chunks:
        embedding = model.encode(chunk["text"])
        chunk["embedding"] = embedding.tolist()
    return chunks

embeddings = embed_all()
with open("outputs/embedded_chunks.json", "w") as f:
    json.dump(embeddings, f)



    
