import os
import logging
from typing import Union, List
from pathlib import Path
from dotenv import load_dotenv
from langchain_classic.globals import set_debug
from langchain_core.documents import Document
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
set_debug(True)

BANCO_DADOS = os.getenv("BANCO_DADOS", 'banco_dados')
CAMINHO_BASE = Path(__file__).resolve().parents[2]
ARQUIVO_BASE = CAMINHO_BASE / BANCO_DADOS


def verificacao_arquivo(nome_arquivo: str) -> Path:
    arquivo = ARQUIVO_BASE / nome_arquivo
    
    if not arquivo.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {arquivo}")

    return arquivo


def carregamento_arquivo(caminho: Path) -> list[Document]:
    try:
        loader = PyPDFLoader(str(caminho))
        documentos = loader.load()
        
        if not documentos: 
            raise ValueError("Arquivo vazio ou Inválido!")
        
        return documentos
    
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar PDF: {caminho}") from e
    

def cortar_documento(documentos: list[Document]) -> list[Document]:    
    corte_documento = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100)
    
    documentos = corte_documento.split_documents(documentos)
    
    return documentos

def processar_arquivos(arquivos: Union[str, List[str]]):
    if isinstance(arquivos, str):
        arquivos = [arquivos]
    
    lista_chunks = []
    
    for arquivo in arquivos:
        try: 
            caminho_documento = verificacao_arquivo(arquivo)
            documentos = carregamento_arquivo(caminho_documento)
            
            for doc in documentos:
                doc.metadata['source'] = arquivo
            
            chunks = cortar_documento(documentos)
            lista_chunks.extend(chunks)
            
            logging.info(f"Arquivo: {arquivo} processado com sucesso. Gerou {len(chunks)} chunks.")
            
        except Exception as e:
            logging.error(f"Erro ao processar o arquivo {arquivo}: {e}")

    return lista_chunks