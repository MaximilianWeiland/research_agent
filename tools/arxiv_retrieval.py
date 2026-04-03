import arxiv
from langchain.tools import tool

def get_arxiv_tool():

    @tool
    def arxiv_search(query: str) -> str:
        """Search ArXiv for recent papers matching a query. Returns titles, authors, links, and abstracts."""
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        try:
            results = []
            for paper in client.results(search):
                results.append(
                    f"Title: {paper.title}\n"
                    f"Authors: {', '.join(a.name for a in paper.authors)}\n"
                    f"Published: {paper.published.strftime('%Y-%m-%d')}\n"
                    f"URL: {paper.entry_id}\n"
                    f"Abstract: {paper.summary}"
                )
            if not results:
                return "No papers found for this query."
            return "\n\n---\n\n".join(results)
        except Exception as e:
            return f"ArXiv search failed: {e}"

    return arxiv_search