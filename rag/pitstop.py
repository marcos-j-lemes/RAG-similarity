import os
from openai import OpenAI          # pip install openai
import chromadb                    # pip install chromadb
from sentence_transformers import SentenceTransformer  # pip install sentence-transformers

# ==========================================
# 1. Configurações e Inicialização
# ==========================================
PASTA_BANCO = './chroma_db'
COLLECTION_NAME = "knowledge_base"
MODELO_EMBEDDING = 'all-MiniLM-L6-v2'
MODELO_LLM = "llama3.2"  # modelo leve para estudo — troque por "phi3" ou "gemma:2b" se quiser ainda mais leve

# Conecta ao Ollama via interface compatível com OpenAI
client_llm = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

print("Carregando modelo de embedding...")
model_embed = SentenceTransformer(MODELO_EMBEDDING)

print("Conectando ao banco de dados...")
client_db = chromadb.PersistentClient(path=PASTA_BANCO)

# Cria a collection se não existir (autossuficiente)
try:
    collection = client_db.get_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' carregada com {collection.count()} documentos.")
except Exception:
    print(f"Collection '{COLLECTION_NAME}' não encontrada. Criando com dados de exemplo...")
    collection = client_db.create_collection(COLLECTION_NAME)

    # Dados de exemplo para poder testar sem precisar de script separado
    docs_exemplo = [
        "Python é uma linguagem de programação de alto nível, interpretada e de propósito geral.",
        "RAG significa Retrieval-Augmented Generation. É uma técnica que combina busca em base de dados com geração de texto por LLMs.",
        "ChromaDB é um banco de dados vetorial open-source voltado para aplicações de IA.",
        "Ollama é uma ferramenta que permite rodar modelos de linguagem localmente, sem precisar de internet.",
        "Embeddings são representações numéricas (vetores) de textos que capturam seu significado semântico.",
    ]
    metadatas_exemplo = [
        {"category": "programacao"},
        {"category": "ia"},
        {"category": "banco_de_dados"},
        {"category": "ferramentas"},
        {"category": "ia"},
    ]
    ids_exemplo = [f"doc_{i}" for i in range(len(docs_exemplo))]
    embeddings_exemplo = model_embed.encode(docs_exemplo).tolist()

    collection.add(
        documents=docs_exemplo,
        metadatas=metadatas_exemplo,
        ids=ids_exemplo,
        embeddings=embeddings_exemplo,
    )
    print(f"Inseridos {len(docs_exemplo)} documentos de exemplo.")

# ==========================================
# 2. Função de Recuperação (Retrieval)
# ==========================================
def buscar_contexto(pergunta: str, n_resultados: int = 3) -> str:
    """Busca no ChromaDB e retorna um texto único com os contextos encontrados"""
    
    total = collection.count()
    n = min(n_resultados, total)  # evita erro se houver menos docs que o solicitado
    if n == 0:
        return "Nenhum documento na base de conhecimento."

    embedding_pergunta = model_embed.encode(pergunta).tolist()

    resultados = collection.query(
        query_embeddings=[embedding_pergunta],
        n_results=n
    )

    contextos = []
    if resultados['documents'] and resultados['documents'][0]:
        for i, doc in enumerate(resultados['documents'][0]):
            categoria = resultados['metadatas'][0][i].get('category', 'sem categoria')
            contextos.append(f"[Documento {i+1} - Categoria: {categoria}]:\n{doc}")

    return "\n\n---\n\n".join(contextos)

# ==========================================
# 3. Função de Geração (Augmented Generation)
# ==========================================
def gerar_resposta(pergunta: str, contexto: str) -> str:
    """Monta o prompt e envia para o LLM local via Ollama"""

    prompt_sistema = """
    Você é um assistente especialista e prestativo.
    Sua tarefa é responder à pergunta do usuário UTILIZANDO APENAS as informações fornecidas no contexto abaixo.

    REGRAS ESTRITAS:
    1. Se a resposta não estiver no contexto, diga claramente: "Desculpe, não encontrei informações sobre isso na base de conhecimento."
    2. Não invente informações (não alucine).
    3. Não use seu conhecimento interno prévio, use estritamente o texto do contexto.
    4. Formate a resposta de forma clara, humanizada e fácil de ler, usando bullet points se necessário.
    5. Responda no mesmo idioma da pergunta do usuário.
    """

    mensagem_usuario = f"""
    CONTEXTO RECUPERADO DA BASE DE CONHECIMENTO:
    {contexto}

    ---

    PERGUNTA DO USUÁRIO:
    {pergunta}
    """

    resposta = client_llm.chat.completions.create(
        model=MODELO_LLM,
        temperature=0.2,
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": mensagem_usuario}
        ]
    )

    return resposta.choices[0].message.content

# ==========================================
# 4. Loop de Interação no Terminal
# ==========================================
def main():
    print("\n" + "="*50)
    print("🤖 Sistema RAG Iniciado! (Digite 'sair' para encerrar)")
    print("="*50 + "\n")

    while True:
        pergunta = input("❓ Você: ")

        if pergunta.lower() in ['sair', 'exit', 'quit']:
            print("Encerrando... Até logo!")
            break

        if not pergunta.strip():
            continue

        print("\n⏳ Buscando no banco e gerando resposta...\n")

        contexto = buscar_contexto(pergunta, n_resultados=3)
        resposta_final = gerar_resposta(pergunta, contexto)

        print(f"🤖 Assistente:\n{resposta_final}\n")
        print("-" * 50)

if __name__ == "__main__":
    main()