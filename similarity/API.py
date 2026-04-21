from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import chromadb
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. Configurações Iniciais (Rodam ao ligar a API)
# ==========================================
PASTA_BANCO = './chroma_db'
COLLECTION_NAME = "knowledge_base"

# Inicializa a API
app = FastAPI(title="RAG Knowledge Base API")

# Carrega o modelo de embedding na memória
print("API Iniciando: Carregando modelo de IA...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Conecta ao banco de dados existente
print("API Iniciando: Conectando ao ChromaDB...")
client = chromadb.PersistentClient(path=PASTA_BANCO)

try:
    collection = client.get_collection(COLLECTION_NAME)
    print("API Iniciando: Banco conectado com sucesso!")
except Exception as e:
    print(f"ERRO: Não foi possível encontrar a collection '{COLLECTION_NAME}'. Verifique se a pasta '{PASTA_BANCO}' existe.")
    raise e


# ==========================================
# 2. Modelos de Dados (Pydantic)
# Aqui definimos o formato do JSON que a API vai receber e responder
# ==========================================
class SearchRequest(BaseModel):
    query: str                          # A pergunta do usuário (obrigatório)
    n_results: int = 3                  # Quantos resultados quero (padrão: 3)
    category_filter: Optional[str] = None # Se quiser filtrar por categoria (opcional)

class SearchResultItem(BaseModel):
    id: str
    text: str
    category: str
    distance: float                     # Quanto menor, mais parecido é (distância vetorial)

class SearchResponse(BaseModel):
    query: str
    results_count: int
    results: List[SearchResultItem]


# ==========================================
# 3. O Endpoint (A rota da API)
# ==========================================
@app.post("/search", response_model=SearchResponse)
async def search_data(request: SearchRequest):
    
    # Verifica se o usuário mandou texto vazio
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="A query não pode estar vazia.")

    # 1. Transforma a pergunta do usuário em embedding
    query_embedding = model.encode(request.query).tolist()

    # 2. Monta o filtro de categoria (se o usuário tiver passado um)
    where_filter = None
    if request.category_filter:
        where_filter = {"category": request.category_filter}

    # 3. Faz a busca no ChromaDB
    try:
        db_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=request.n_results,
            where=where_filter
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao buscar no banco: {str(e)}")

    # 4. Formata os resultados para devolver como JSON bonito
    formatted_results = []
    
    # O ChromaDB retorna listas de listas (porque aceita várias queries de uma vez)
    # Como mandamos só 1 query, pegamos o índice [0]
    if db_results['ids'] and db_results['ids'][0]:
        for i in range(len(db_results['ids'][0])):
            formatted_results.append(
                SearchResultItem(
                    id=db_results['ids'][0][i],
                    text=db_results['documents'][0][i],
                    category=db_results['metadatas'][0][i]['category'],
                    distance=db_results['distances'][0][i]
                )
            )

    # 5. Retorna a resposta final
    return SearchResponse(
        query=request.query,
        results_count=len(formatted_results),
        results=formatted_results
    )