from langchain.tools import tool

def get_retrieval_tool(vector_store):

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve relevant chunks from documents using a vector store."""
        docs = vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}"
            for doc in docs
        )
        return serialized, docs

    return retrieve_context