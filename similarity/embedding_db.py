import json
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Configurações
CAMINHO_ARQUIVO = './dataset_clean/dataset_clean.json'
PASTA_BANCO = './chroma_db'

# 2. Carregar o modelo EM INGLÊS (mais rápido, ideal para o seu caso)
print("Carregando modelo...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Ler os dados do JSON
print("Lendo dados...")
with open(CAMINHO_ARQUIVO, 'r', encoding='utf-8') as f:
    raw_data = json.load(f) # raw_data é um dicionário: {"Categoria": [itens...]}

# 4. "Achatar" (Flatten) o dicionário para criar listas separadas
texts = []
ids = []
categories = []

# Percorre as chaves (nomes das categorias) e os valores (listas de dicionários)
for categoria_nome, lista_de_itens in raw_data.items():
    for item in lista_de_itens:
        texts.append(item['combined'])         # Pega o texto final
        ids.append(str(item['id']))            # Pega o ID
        categories.append(item['category'])    # Pega a categoria oficial do JSON

print(f"Foram lidos {len(texts)} documentos.")

# 5. Gerar Embeddings
print("Gerando embeddings...")
embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

# 6. Conectar ao Banco Vetorial Persistente
print("Salvando no banco vetorial...")
client = chromadb.PersistentClient(path=PASTA_BANCO)

collection_name = "knowledge_base"
if collection_name in [c.name for c in client.list_collections()]:
    client.delete_collection(collection_name)
    
collection = client.create_collection(collection_name)

# 7. Preparar metadados (agora usando a categoria REAL do JSON)
metadatas = [{"category": cat, "token_count": 0} for cat in categories] 
# Nota: O ChromaDB só aceita strings, inteiros, floats ou booleanos nos metadados. 

# Transforma embeddings de Numpy para Listas (exigência do Chroma)
embeddings_list = [emb.tolist() for emb in embeddings]

# 8. Inserir tudo de uma vez no ChromaDB
collection.add(
    ids=ids,
    documents=texts,
    embeddings=embeddings_list,
    metadatas=metadatas
)

print(f"✅ Concluído! {len(texts)} documentos salvos no banco usando suas categorias oficiais.")

# --- Exemplo prático de como você vai buscar depois: ---
# Suponha que você queira buscar algo só na categoria de Technology
# results = collection.query(
#     query_texts=["How do neural networks work?"],
#     n_results=3,
#     where={"category": "Technology & Computing"} # Filtra pela categoria real!
# )