from dataclasses import dataclass
from typing import List
from langchain.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock
from rag_app.get_chroma_db import get_chroma_db
from rag_app.get_model import get_model_llm
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import AIMessage, HumanMessage
from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field


PROMPT_TEMPLATE = """
Eres un asistente para la respuesta de preguntas sobre los documentos sobre documentación cargada.
Usa únicamente los siguientes documentos para responder la pregunta.
Si no sabes la respuesta, dí que no la sabes y si tienes sugerencias preséntalas.
Se amable, recuerda que eres un asistente.
Sé muy conciso con la respuesta y házlo en español:
Documentos: {context}
Respuesta:
"""

HISTORY_PROMPT_TEMPLATE = """
Dado un historial de chat y la última pregunta del usuario que podría hacer referencia al contexto del historial de chat o no,
Si la pregunta está relacionada con el historial NO responda la pregunta,
sólo formule una pregunta independiente que pueda entenderse sin el historial de chat.
Si no está relacionada con mensajes anteriores sólo responda la pregunta.
"""

@dataclass
class QueryResponse:
    query_text: str
    response_text: str
    sources: List[str]

def format_chat_history(chat_history, len_history=6):
    formatted_history = []
    if len(chat_history) > 0:
        for human, ai in chat_history[-len_history:]:
            formatted_history.append(HumanMessage(content=human))
            formatted_history.append(AIMessage(content=ai))
        return formatted_history
    else:
        return chat_history


def rag_chain():
    model = get_model_llm()
    db = get_chroma_db()
    condense_question_prompt = ChatPromptTemplate(
        [
            ("system", HISTORY_PROMPT_TEMPLATE),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
                                    model, db.as_retriever(search_type="similarity", search_kwargs={"k": 3}), condense_question_prompt
                                )

    qa_prompt = ChatPromptTemplate(
        [
            ("system", PROMPT_TEMPLATE),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    qa_chain = create_stuff_documents_chain(model, qa_prompt)
    convo_qa_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return convo_qa_chain


def query_rag(query_object) -> QueryResponse:
    rag_response = rag_chain()

    chat_formatted = format_chat_history(query_object.chat_history)

    response = rag_response.invoke({
        "input": query_object.query_text,
        "chat_history": chat_formatted,
    })
    response_text = response["answer"]
    results = response["context"]

    sources = [doc.metadata.get("id", None) for doc in results]
    print(f"Response: {response_text}\nSources: {sources}")

    return QueryResponse(
        query_text=query_object.query_text, response_text=response_text, sources=sources
    )
