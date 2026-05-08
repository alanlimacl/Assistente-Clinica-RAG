import os
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.tools.retriever import create_retriever_tool
from langchain.agents import create_agent
from embeddings_vectorstore import base_vetorial

load_dotenv()

OPENROUTER_API = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_BASE = os.getenv('OPENROUTER_BASE_URL')
MODELO = os.getenv('MODELO')

llm = ChatOpenRouter(api_key=OPENROUTER_API,
                     app_url=OPENROUTER_BASE,
                     model=MODELO)

arquivos = ["clinica_vidaesaude.pdf"]
vector_store = base_vetorial(arquivos)

retriever = vector_store.as_retriever(search_kwargs={"k": 4})

ferramenta_busca = create_retriever_tool(
    retriever,
    name="pesquisa_documentos_pdf",
    description="Pesquise nos documentos do usuário. Use esta ferramenta sempre que precisar responder perguntas sobre os PDFs anexados."
)

tools = [ferramenta_busca]

prompt = """Você é um assistente prestativo. Use a ferramenta de pesquisa de 
documentos para basear suas respostas nas informações dos PDFs do usuário. Se não encontrar a resposta nos documentos, diga que não sabe."""

agente = create_agent(model=llm,
                      tools=tools,
                      system_prompt=prompt,
                      debug=True,
                      )

pergunta = "Quais exames tem?"
resposta = agente.invoke({"messages": [("user", pergunta)]})

print("\n--- RESPOSTA FINAL ---")
print(resposta['messages'][-1].content)