import faiss
import numpy as np

index = faiss.IndexFlatL2(512)
image_paths = []

def add_embedding(embedding, path):
    global index, image_paths
    index.add(embedding)
    image_paths.append(path)

def search(query_embedding, k=5):
    D, I = index.search(query_embedding, k)
    results = [image_paths[i] for i in I[0]]
    return results