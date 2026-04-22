import re
import arxiv
import tempfile
import requests
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

SEMANTIC_SCHOLAR_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
CANDIDATE_POOL = 100
TOP_N = 3


def _extract_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _arxiv_id(entry_id: str) -> str:
    """Extract bare ArXiv ID (no version) from entry_id URL."""
    match = re.search(r"abs/([^v]+)", entry_id)
    return match.group(1) if match else entry_id


def _fetch_citation_counts(arxiv_ids: list[str]) -> dict[str, int]:
    """Batch-fetch citation counts from Semantic Scholar. Returns {arxiv_id: count}."""
    ids = [f"arXiv:{id_}" for id_ in arxiv_ids]
    try:
        response = requests.post(
            SEMANTIC_SCHOLAR_BATCH_URL,
            params={"fields": "citationCount,externalIds"},
            json={"ids": ids},
            timeout=10,
        )
        response.raise_for_status()
        counts = {}
        for item in response.json():
            if item is None:
                continue
            ext_ids = item.get("externalIds") or {}
            arxiv_id = ext_ids.get("ArXiv")
            if arxiv_id:
                counts[arxiv_id] = item.get("citationCount", 0)
        return counts
    except Exception as e:
        print(f"Semantic Scholar fetch failed: {e}")
        return {}


def get_arxiv_tool(vector_store=None):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    @tool
    def arxiv_search(query: str) -> str:
        """Search ArXiv for papers matching a query, ranked by citation count. Returns titles, authors, links, and abstracts."""
        client = arxiv.Client()
        search = arxiv.Search(
            query=f"ti:{query}",
            max_results=CANDIDATE_POOL,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        try:
            papers = list(client.results(search))
            if not papers:
                return "No papers found for this query."

            arxiv_ids = [_arxiv_id(p.entry_id) for p in papers]
            citation_counts = _fetch_citation_counts(arxiv_ids)
            papers.sort(key=lambda p: citation_counts.get(_arxiv_id(p.entry_id), 0), reverse=True)

            results = []
            for paper in papers[:TOP_N]:
                arxiv_id = _arxiv_id(paper.entry_id)
                citations = citation_counts.get(arxiv_id, "unknown")
                results.append(
                    f"Title: {paper.title}\n"
                    f"Authors: {', '.join(a.name for a in paper.authors)}\n"
                    f"Published: {paper.published.strftime('%Y-%m-%d')}\n"
                    f"Citations: {citations}\n"
                    f"URL: {paper.entry_id}\n"
                    f"Abstract: {paper.summary}"
                )
                if vector_store is not None:
                    metadata = {
                        "title": paper.title,
                        "authors": ", ".join(a.name for a in paper.authors),
                        "published": paper.published.strftime("%Y-%m-%d"),
                        "url": paper.entry_id,
                        "source": "arxiv",
                    }
                    full_text = None

                    # attempt three times to retrieve the full text, otherwise fallback to abstract
                    for attempt in range(3):
                        try:
                            with tempfile.TemporaryDirectory() as tmpdir:
                                pdf_path = paper.download_pdf(dirpath=tmpdir)
                                full_text = _extract_pdf_text(pdf_path)
                            break
                        except Exception:
                            if attempt == 2:
                                full_text = paper.summary
                    chunks = splitter.split_text(full_text)
                    docs = [Document(page_content=chunk, metadata=metadata) for chunk in chunks]
                    vector_store.add_documents(docs)

            return "\n\n---\n\n".join(results)
        except Exception as e:
            return f"ArXiv search failed: {e}"

    return arxiv_search

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from rag.vectorstore import get_vectorstore
    vectorstore = get_vectorstore()
    arxiv_tool = get_arxiv_tool(vectorstore)
    result = arxiv_tool.invoke("Deepseek")
    print(result)
