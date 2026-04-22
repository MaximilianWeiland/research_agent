# Research Agent

A conversational research assistant built with LangChain, LangGraph, and Streamlit. The agent answers questions about research topics using a combination of RAG over indexed papers and external tools.

## Features

- RAG over a local collection of research papers (ChromaDB vector store)
- Web search via Tavily
- ArXiv paper retrieval
- Wikipedia search
- YouTube transcript retrieval
- Conversation history persisted in PostgreSQL via LangGraph checkpointing

## Project Structure

```
research_agent/
├── src/
│   ├── agent/          # LangGraph agent and system prompt
│   ├── checkpointing/  # PostgreSQL conversation persistence
│   ├── config/         # Settings and secrets loading
│   ├── rag/            # Vector store setup
│   └── tools/          # Agent tools
├── indexing/           # Script to build the vector index from PDFs
├── data/               # ChromaDB vector store (not committed)
├── docs/               # Source PDF papers (not committed)
├── app.py              # Streamlit entry point
├── Dockerfile
└── docker-compose.yml
```

## Local Development

**1. Install dependencies**
```
uv sync
```

**2. Set up environment variables**

Create a `.env` file:
```
OPENAI_API_KEY=...
TAVILY_API_KEY=...
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/research_agent
```

**3. Start Postgres**
```
docker compose up -d
```

**4. Build the vector index** (requires PDFs in `docs/`)
```
uv run python indexing/build_index.py
```

**5. Run the app**
```
uv run streamlit run app.py
```

## Deployment

The app is containerized and deployed on AWS:

- **ECR** — Docker image registry
- **EC2 + nginx** — runs the container, nginx handles WebSocket proxying
- **RDS** — managed PostgreSQL for conversation history
- **Secrets Manager** — stores API keys

**Build and push image (Apple Silicon):**
```
docker buildx build --platform linux/amd64 -t research-agent .
docker tag research-agent:latest <ECR_URI>:latest
docker push <ECR_URI>:latest
```

**Redeploy on EC2:**
```
ssh -i research-agent-key.pem ec2-user@<EC2_IP>
docker pull <ECR_URI>:latest
docker stop research-agent && docker rm research-agent
docker run -d --name research-agent --restart unless-stopped -p 8501:8501 \
  -e AWS_SECRETS_NAME=research-agent/prod \
  -e AWS_REGION=eu-central-1 \
  <ECR_URI>:latest
```
