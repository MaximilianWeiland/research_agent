import wikipedia
from langchain.tools import tool


def get_wikipedia_tool():

    @tool
    def wikipedia_search(query: str):
        """Search Wikipedia and return a summary for the given query."""
        try:
            return wikipedia.summary(query, sentences=10)
        except wikipedia.DisambiguationError as e:
            return wikipedia.summary(e.options[0], sentences=10)
        except wikipedia.PageError:
            return f"No Wikipedia page found for '{query}'."

    return wikipedia_search
