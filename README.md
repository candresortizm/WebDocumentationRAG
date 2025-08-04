# Implementación RAG a partir de documentación web

## Arquitectura:
La aplicación es un API construída con FastAPI que expone los endpoints necesarios para interactuar con el sistema: 
- POST /api/v1/process-documentation
- GET /api/v1/processing-status/{chatId}
- POST /api/v1/chat/{chatId}
- GET /api/v1/chat-history/{chatId}

Los modelos de embedding y de LLM usados se encuentran en Amazon Bedrock, y para acceder a ellos se usó el SDK para python (boto3), se escogieron estos modelos porque el consumo no resulta tan costoso y porque ya tenía una implementación previa.
Como Base de datos se usó Chroma por su poco peso, fácil manejo y gratuidad.
 Para las particiones del texto se usó el RecursiveCharacterTextSplitter para partir el texto de manera semántica.
 Para la carga de los documentos se usó el RecursiveUrlLoader con una profundidad de 2.

## Ejecución local:

Es necesario contar con el CLI de AWS y con las credenciales de API KEY registradas.

Se recomienda primero la creación de un entorno virtual.

python -m venv env

Activar el entorno virtual:

.\env\Scripts\activate

Instalar las dependencias:

pip install .

Ejecutar el Api:

python .\src\api_handler.py  


## TODO:
- Implementación de la asincronía
- Implementación del gráfo con langgraph

## Referencias:

para la construcción de este proyecto se consultaron principalmente estas fuentes:

(https://www.youtube.com/watch?v=ldFONBo2CR0&t=2744s) https://github.com/pixegami/deploy-rag-to-aws/tree/main 

https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_agentic_rag.ipynb
