from langchain_aws import BedrockEmbeddings
#from langchain_nomic.embeddings import NomicEmbeddings

def get_embedding_function():
    embedding = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
    return embedding

#def get_embedding_function():
#    embedding = NomicEmbeddings(
#                                    model="nomic-embed-text-v1.5",
#                                    inference_mode="local",
#                                )
#    return embedding