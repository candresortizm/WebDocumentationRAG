from langchain_aws import ChatBedrock

MODEL_LLM = None  # Reference to singleton instance of ChromaDB
BEDROCK_MODEL_ID = "amazon.nova-lite-v1:0"

def get_model_llm():
    global MODEL_LLM
    if not MODEL_LLM:
        # Prepare the DB.
        MODEL_LLM = ChatBedrock(model_id=BEDROCK_MODEL_ID)
        print(f"✅ Init Model LLM {MODEL_LLM}")

    return MODEL_LLM