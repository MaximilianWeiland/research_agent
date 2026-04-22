from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.base import BaseCheckpointSaver as Checkpointer
from rag.vectorstore import get_vectorstore
from tools.retrieval import get_retrieval_tool
from tools.web_search import get_web_tool
from tools.arxiv_retrieval import get_arxiv_tool
from tools.youtube_transcripts import get_youtube_tool
from tools.wikipedia_search import get_wikipedia_tool
from agent.prompt import SYSTEM_PROMPT


def build_agent(checkpointer: Checkpointer | None = None):
    model = ChatOpenAI(model="gpt-4o")
    vector_store = get_vectorstore()

    tools = [
        get_retrieval_tool(vector_store),
        get_web_tool(),
        get_arxiv_tool(vector_store),
        get_youtube_tool(),
        get_wikipedia_tool()
    ]

    return create_agent(model, tools, system_prompt=SYSTEM_PROMPT, checkpointer=checkpointer)
