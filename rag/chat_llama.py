import os
# from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. Configurações e Inicialização
# ==========================================
PASTA_BANCO = './chroma_db'
COLLECTION_NAME = "knowledge_base"
MODELO_EMBEDDING = 'all-MiniLM-L6-v2'
MODELO_LLM = "gpt-4o-mini" # Troque para "gpt-3.5-turbo" se quiser mais barato

client_llm = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
MODELO_LLM = "llama3"

print("Carregando modelo de embedding...")
model_embed = SentenceTransformer(MODELO_EMBEDDING)

print("Conectando ao banco de dados...")
client_db = chromadb.PersistentClient(path=PASTA_BANCO)
collection = client_db.get_collection(COLLECTION_NAME)

# ==========================================
# 2. Função de Recuperação (Retrieval)
# ==========================================
def buscar_contexto(pergunta: str, n_resultados: int = 3) -> str:
    """Busca no ChromaDB e retorna um texto único com os contextos encontrados"""
    
    # Gera o embedding da pergunta
    embedding_pergunta = model_embed.encode(pergunta).tolist()
    
    # Consulta o banco
    resultados = collection.query(
        query_embeddings=[embedding_pergunta],
        n_results=n_resultados
    )
    
    # Formata os resultados em um único bloco de texto
    contextos = []
    if resultados['documents'] and resultados['documents'][0]:
        for i, doc in enumerate(resultados['documents'][0]):
            categoria = resultados['metadatas'][0][i]['category']
            contextos.append(f"[Documento {i+1} - Categoria: {categoria}]:\n{doc}")
            
    return "\n\n---\n\n".join(contextos)

# ==========================================
# 3. Função de Geração (Augmented Generation)
# ==========================================
def gerar_resposta(pergunta: str, contexto: str) -> str:
    """Monta o prompt e envia para o LLM"""
    
    # O coração do RAG é o SYSTEM PROMPT. 
    # Dizemos claramente ao modelo como ele deve se comportar.
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
    
    # Monta a mensagem final
    mensagem_usuario = f"""
    CONTEXTO RECUPERADO DA BASE DE CONHECIMENTO:
    {contexto}
    
    ---
    
    PERGUNTA DO USUÁRIO:
    {pergunta}
    """
    
    # Chamada para a API da OpenAI
    resposta = client_llm.chat.completions.create(
        model=MODELO_LLM,
        temperature=0.2, # Baixa temperatura = respostas mais focadas e factuais (menos criatividade/delírio)
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
        
        # Passo 1: Puxa os textos do banco
        contexto = buscar_contexto(pergunta, n_resultados=3)
        
        # Passo 2: Manda para a IA gerar a resposta
        resposta_final = gerar_resposta(pergunta, contexto)
        
        print(f"🤖 Assistente:\n{resposta_final}\n")
        print("-" * 50)

if __name__ == "__main__":
    main()