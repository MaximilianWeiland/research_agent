import os
import psycopg
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.base import BaseCheckpointSaver as Checkpointer
from langgraph.checkpoint.postgres import PostgresSaver
from rag.vectorstore import get_vectorstore
from tools.retrieval import get_retrieval_tool
from tools.web_search import get_web_tool
from tools.arxiv_retrieval import get_arxiv_tool
from tools.youtube_transcripts import get_youtube_tool
from tools.wikipedia_search import get_wikipedia_tool
from agent.prompt import SYSTEM_PROMPT


def get_checkpointer() -> PostgresSaver:
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()
    return checkpointer


def build_agent(checkpointer: Checkpointer | None = None, model: str = "gpt-4o", temperature: float = 1.0):
    llm = ChatOpenAI(model=model, temperature=temperature)
    vector_store = get_vectorstore()

    tools = [
        get_retrieval_tool(vector_store),
        get_web_tool(),
        get_arxiv_tool(vector_store),
        get_youtube_tool(),
        get_wikipedia_tool()
    ]

    return create_agent(llm, tools, system_prompt=SYSTEM_PROMPT, checkpointer=checkpointer)
