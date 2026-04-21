import chromadb
from sentence_transformers import SentenceTransformer


model_emb = SentenceTransformer("BAAI/bge-m3")

client = chromadb.Client()
VDB = client.create_collection("base_knowledge")

