# Production RAG API

A production-style Retrieval-Augmented Generation (RAG) API built with FastAPI. This repository includes ingestion, retrieval, and answer generation routes backed by a local Chroma vector store and an OpenRouter-powered LLM.

## Project Structure

- `API/main.py` - FastAPI application entrypoint and server configuration.
- `API/Routes/` - HTTP route handlers for ingestion, retrieval, and question-answering.
- `Services/` - Core application services for ChromaDB, embeddings, document ingestion, and LLM integration.
- `Utils/get_answer.py` - High-level RAG pipeline orchestration for query embedding, retrieval, and LLM generation.
- `Embedding_Model/` - Local embedding model assets and configuration.
- `Database/` - Chroma SQLite database files and generated collection data.
- `UI/` - Static frontend assets for a simple client interface.

## Recommended Dependencies

Install the following Python packages before running the application:

- `fastapi`
- `uvicorn`
- `requests`
- `python-dotenv`
- `chromadb`
- `sentence-transformers` (or any dependency required by the local embedding model loader)

> If you do not have a `requirements.txt`, install packages directly with `pip`.

## Environment Variables

Create a `.env` file in the repository root with at least the following values:

```env
OPEN_ROUTER_KEY=your_openrouter_api_key
APP_NAME=production-rag-api
APP_VERSION=1.0.0
HOST=0.0.0.0
PORT=8100
DEBUG=true
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
```

`TELEGRAM_BOT_TOKEN` is optional. If it is set, the bot will start automatically when `python API/main.py` is run.

## Running the API

Start the application from the repository root:

```bash
python API/main.py
```

Then open the API in your browser or send requests to:

- `http://localhost:8100/`
- `http://localhost:8100/docs` for the FastAPI interactive docs

## Telegram Bot

If `TELEGRAM_BOT_TOKEN` is set in `.env`, the Telegram bot will start alongside the backend when `python API/main.py` is executed. The bot uses the same RAG service and query pipeline as the API.

To stop the application, press `Ctrl+C` in the terminal. The backend and Telegram polling process are both shut down cleanly.

## Available Routes

- `POST /ingest` - Ingest documents into the vector store.
- `POST /retrieval` - Search the vector store for relevant context.
- `POST /get-answer` - Execute the full RAG pipeline and generate an answer.

## Semantic Caching

- The RAG endpoint now stores generated answers in a semantic query cache collection.
- If a new query is sufficiently similar to a cached query, the cached answer is returned immediately.
- The cache is stored in an additional ChromaDB collection named `<collection_name>_query_cache`.

## Notes

- The app uses a local `Embedding_Model/` folder for embeddings and a Chroma SQLite store in `Database/`.
- The LLM integration is implemented in `Services/llm.py` and expects `OPEN_ROUTER_KEY` to be set.
- CORS is enabled for all origins by default; adjust in production as needed.

## Quick Start

1. Clone or open the repository.
2. Install dependencies: `pip install fastapi uvicorn requests python-dotenv chromadb sentence-transformers`
3. Create a `.env` file.
4. Run `python API/main.py`.
5. Use the `/docs` UI to test ingestion, retrieval, and answer generation.
