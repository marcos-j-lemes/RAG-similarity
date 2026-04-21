import chromadb
from sentence_transformers import SentenceTransformer

# 1. Apontar para a MESMA pasta onde o banco foi salvo
PASTA_BANCO = './chroma_db'

# 2. Carregar o MESMO modelo usado para criar os embeddings
# Como salvamos vetores em inglês no script anterior, usamos o mesmo aqui
print("Carregando modelo...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Conectar ao banco existente
client = chromadb.PersistentClient(path=PASTA_BANCO)

# 4. PEGAR a collection existente (não usar create_collection!)
# Se você usar o create_collection aqui, vai dar erro de que já existe
collection = client.get_collection("knowledge_base")

# ==========================================
# EXEMPLO 1: Busca Simples (Sem filtro de categoria)
# ==========================================
pergunta_do_usuario = "How do neural networks learn?"

# Transformar a pergunta em embedding (o Chroma exige isso porque não 
# salvamos um "modelo padrão" atrelado à collection no script anterior)
embedding_da_pesquisa = model.encode(pergunta_do_usuario).tolist()

print(f"\nBuscando por: '{pergunta_do_usuario}'\n")
resultados = collection.query(
    query_embeddings=[embedding_da_pesquisa], # Passamos o vetor da pergunta
    n_results=2 # Traz os 2 melhores resultados
)

# Mostrar os resultados
for i in range(len(resultados['ids'][0])):
    print(f"Resultado {i+1} (Distância: {resultados['distances'][0][i]:.4f})")
    print(f"Texto: {resultados['documents'][0][i][:150]}...") # é possivel remover para não ter limites
    print("-" * 50)


# ==========================================
# EXEMPLO 2: Busca Filtrada (Dentro de uma categoria específica)
# ==========================================
# Aqui entra a vantagem de ter salvado a categoria como metadado!
pergunta_filtrada = "What is python?"
embedding_filtrado = model.encode(pergunta_filtrada).tolist()

print(f"\nBuscando por: '{pergunta_filtrada}' APENAS em 'Technology & Computing'\n")
resultados_filtrados = collection.query(
    query_embeddings=[embedding_filtrado],
    n_results=2,
    where={
        "category": "Technology & Computing" # Filtra a busca!
    }
)

for i in range(len(resultados_filtrados['ids'][0])):
    print(f"Resultado {i+1}:")
    print(f"Texto: {resultados_filtrados['documents'][0][i][:150]}...")
    print("-" * 50)