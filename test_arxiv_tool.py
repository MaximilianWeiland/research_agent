from langchain_community.utilities import ArxivAPIWrapper

arxiv = ArxivAPIWrapper()
docs = arxiv.run("1605.08386")
docs