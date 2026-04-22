from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool

def get_web_tool():
    search = TavilySearchResults(k=2)

    @tool
    def web_search(query: str):
        """Perform a web search with a custom query."""
        return str(search.invoke(query))

    return web_search