from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from config.settings import DB_DIR
from dotenv import load_dotenv
load_dotenv()

def get_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
    )