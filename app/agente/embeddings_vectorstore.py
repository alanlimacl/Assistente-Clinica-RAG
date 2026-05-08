import os
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from documento_corte import processar_arquivos

BANCO_DADOS = os.getenv("BANCO_DADOS", 'banco_dados')
CAMINHO_BASE = Path(__file__).resolve().parents[2]
ARQUIVO_BASE = CAMINHO_BASE / BANCO_DADOS

def base_vetorial(arquivos):
    chunks = processar_arquivos(arquivos)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(ARQUIVO_BASE)
    )
    
    return vector_store

