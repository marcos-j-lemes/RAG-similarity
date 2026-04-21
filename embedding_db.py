from sentence_transformers import SentenceTransformer
from sklean.cluster import KMeans
from chromadb import Client

# o arquivo csv possui 10 categorias
num_clusters = 10

model = SentenceTransformer('all-MiniLM-L6-v2') # 80MB, rápido


# ler arquivo em ./dataset_clean.txt
# também pode ser um csv
with open('./dataset_clean.txt', 'r') as f:
    data = f.read().splitlines()
    embeddings = model.encode(data)

kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)

# criar um dicionário para armazenar os clusters
clusters = {i: [] for i in range(num_clusters)}


client = Client()
collection = client.create_collection("knowledge_base")

# adicionar os dados ao banco de dados
for i, label in enumerate(kmeans.labels_):
    clusters[label].append(data[i])
    collection.add(
        documents=[data[i]],
        metadatas={"cluster": label},
        ids=[str(i)]
    )


