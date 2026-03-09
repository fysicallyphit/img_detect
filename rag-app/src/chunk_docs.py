from load_docs import open_files
import json 

def chunk_text(chunk_size, overlap):
    docs = open_files()
    chunks = []
    for doc in docs:
        text = doc["text"]
        source = doc["source"]
        chunk_number = 0    
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i : i + chunk_size]
            chunk_id = f"{source}_{chunk_number}"
            chunk_dict = {
                "chunk_id" : chunk_id,
                "source" : source,
                "text" : chunk,
            }
            chunks.append(chunk_dict)
            chunk_number +=1
    return chunks

chunks = chunk_text(700, 150)
with open("outputs/chunks.json", "w") as f:
    json.dump(chunks, f)

