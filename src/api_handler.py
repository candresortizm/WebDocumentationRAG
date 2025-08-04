import os
import uvicorn
import boto3
import json
from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel
from query_model import HistorialModel
from rag_app.query_rag import query_rag
from cargueBD import load_web_main, cargar_txt
from langchain.schema import AIMessage, HumanMessage
import asyncio
import functools

WORKER_LAMBDA_NAME = os.environ.get("WORKER_LAMBDA_NAME", None)

app = FastAPI()
handler = Mangum(app)  # Entry point for AWS Lambda.

v1_router = APIRouter(prefix="/api/v1")

origins = [
    "http://localhost:8000",
    "http://localhost:5000", #Para el consumo desde el proyecto frontend
    "https://gestiondeinformacion.unal.edu.co:8000",
    "https://gestiondeinformacion.unal.edu.co"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SubmitQueryRequest(BaseModel):
    query_text: str
    chat_history: list[tuple[str, str]]

class SubmitLoadRequest(BaseModel):
    url_string: str
    chat_id: str


@app.get("/")
def index():
    return {"Hello": "World"}


@app.get("/get_query")
def get_query_endpoint(chat_id: str) -> HistorialModel:
    query = HistorialModel.get_item(chat_id)
    return query

@v1_router.post("/process-documentation")
async def cargar_pagina(request: SubmitLoadRequest, background_tasks: BackgroundTasks):
    new_query = HistorialModel(query_text=request.url_string,chat_id=request.chat_id,answer_text="Procesando...")
    new_query.put_item()
    background_tasks.add_task(load_web_main,request.url_string,new_query)
    return "En progreso"

@app.get("/cargar_txts")
async def cargar_txts(background_tasks: BackgroundTasks):
    background_tasks.add_task(cargar_txt)
    return "Taran"

@v1_router.get("/processing-status/{chatId}")
async def verificar_estado(chatId, background_tasks: BackgroundTasks):
    partial_func = functools.partial(HistorialModel.get_item, chatId)
    consulta_estado = await asyncio.to_thread(partial_func)
    if consulta_estado == None:
        raise HTTPException(status_code=404, detail="Item not found")
    return consulta_estado

@v1_router.post("/chat/{chatId}")
def submit_query_endpoint(request: SubmitQueryRequest,chatId) -> HistorialModel:
    # Create the query item, and put it into the data-base.
    consulta_estado = HistorialModel.get_item(chatId)
    if consulta_estado == None:
        raise HTTPException(status_code=404, detail="Item not found")
    if consulta_estado.answer_text != "DOCUMENTOS CARGADOS":
        raise HTTPException(status_code=404, detail="Los documentos de ese id no están cargados")
    new_query = HistorialModel(query_text=request.query_text,chat_id=chatId)

    if WORKER_LAMBDA_NAME:
        # Make an async call to the worker (the RAG/AI app).
        new_query.put_item()
        invoke_worker(new_query)
    else:
        # Make a synchronous call to the worker (the RAG/AI app).
        query_response = query_rag(request)
        new_query.answer_text = query_response.response_text
        new_query.sources = query_response.sources
        new_query.is_complete = True
        new_query.put_item()

    return new_query


@v1_router.get("/chat-history/{chatId}")
def verificar_estado(chatId, background_tasks: BackgroundTasks):
    consulta_estado = HistorialModel.get_history(chatId)
    if consulta_estado == None:
        raise HTTPException(status_code=404, detail="Item not found")
    return consulta_estado

app.include_router(v1_router)

if __name__ == "__main__":
    # Run this as a server directly.
    port = 8000
    print(f"Running the FastAPI server on port {port}.")
    uvicorn.run("api_handler:app", host="0.0.0.0", port=port)
