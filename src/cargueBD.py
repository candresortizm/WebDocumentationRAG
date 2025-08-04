from langchain_chroma import Chroma
import argparse
import os
import shutil
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rag_app.get_embedding import get_embedding_function
from langchain.schema.document import Document
from query_model import HistorialModel
import re
from bs4 import BeautifulSoup


CHROMA_PATH = "src/data/chroma"
DATA_SOURCE_PATH = "src/data/texto"

async def load_documents():
    document_loader = DirectoryLoader(path=DATA_SOURCE_PATH, glob="**/*.txt", loader_cls=TextLoader,loader_kwargs={'encoding': 'utf-8'})
    return await document_loader.aload()

async def cargar_txt():
    documents = await load_documents()
    chunks = await split_documents(documents)
    await add_to_chroma(chunks)
    
async def load_web_main(url_string: str, historial_model):
    documents = await load_web(url_string)
    chunks = await split_documents(documents)
    await add_to_chroma(chunks)
    query = historial_model.update_item()
    print("Finalizado")

#Cargar recursivamente la documentación
async def load_web(url_string, profundidad=2):
    document_loader = RecursiveUrlLoader(url_string, extractor=bs4_extractor,max_depth=profundidad,encoding='utf-8')
    return await document_loader.aload()

#Extractor para extraer el texto plano del HTML
def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()

async def guardar_txt(documentos):
    directory = "src/data/texto/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    for documento in documentos:
        # Crea el archivo txt
        file = open(directory+documento.metadata["source"].replace("https://","").replace("/","_").replace(".","-")+".txt", "w", encoding='utf-8')
        print(str(file))
        #recorre las imagenes de cada pdf
        file.write(documento.page_content)
        file.write("\n")

async def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=500,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


async def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = await calculate_chunk_ids(chunks)
    
    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        print("Documents added")
    else:
        print("✅ No new documents to add")

async def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index
    current_chunk_index = 0

    for chunk in chunks:

        chunk.metadata = {k: v for k, v in chunk.metadata.items()
            if isinstance(v, (str, int, float, bool)) and v is not None}
        
        source = chunk.metadata.get("source")

        # If the page ID is the same as the last one, increment the index.
        current_chunk_index += 1

        # Calculate the chunk ID.
        chunk_id = f"{source}:{current_chunk_index}"

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id
    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
