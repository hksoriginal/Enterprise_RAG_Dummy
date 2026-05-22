import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from Services.chroma_db_service import ChromaDBService
from Services.embedding_service import EmbeddingService
from Services.llm import LLMService


# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# ------------------------------------------------------------------------------
# Custom Exceptions
# ------------------------------------------------------------------------------

class RAGPipelineError(Exception):
    """Base exception for RAG pipeline."""


class RetrievalError(RAGPipelineError):
    """Raised when retrieval fails."""


class GenerationError(RAGPipelineError):
    """Raised when generation fails."""


# ------------------------------------------------------------------------------
# RAG Service
# ------------------------------------------------------------------------------

class RAGService:
    """
    Production-grade Retrieval-Augmented Generation pipeline.

    Pipeline:
        Query
          ↓
        Embedding
          ↓
        Vector Search
          ↓
        Context Augmentation
          ↓
        LLM Generation

    Features:
    - Async-compatible
    - Structured logging
    - Retrieval validation
    - Configurable top-k
    - Safer prompt construction
    - Timing metrics
    - Service reuse
    """

    def __init__(
        self,
        collection_name: str,
        top_k: int = 5,
        llm_model: str = "meta-llama/llama-3.3-70b-instruct",
    ) -> None:
        """
        Initialize RAG service.

        Args:
            collection_name: ChromaDB collection name.
            top_k: Number of retrieved chunks.
            llm_model: LLM model identifier.
        """

        logger.info(
            "Initializing RAG service | collection=%s",
            collection_name,
        )

        self.top_k = top_k
        self.llm_model = llm_model

        # Reuse services instead of recreating each request
        self.embedding_service = EmbeddingService()
        self.vector_db = ChromaDBService(collection_name)
        self.query_cache_db = ChromaDBService(
            f"{collection_name}_query_cache"
        )
        self.cache_hit_threshold = 0.15
        self.llm_service = LLMService()

        logger.info("RAG service initialized successfully")

    # --------------------------------------------------------------------------
    # Cache Helpers
    # --------------------------------------------------------------------------

    def _get_cached_answer(
        self,
        query_embedding: List[float],
    ) -> Optional[Dict[str, Any]]:
        """
        Return a cached answer when a semantically similar query exists.

        Args:
            query_embedding: Query vector embedding.

        Returns:
            Cached entry dictionary or None.
        """

        try:
            cache_results = self.query_cache_db.query_documents(
                query_embedding=query_embedding,
                n_results=1,
            )

            cached_documents = cache_results.get("documents", [[]])[0]
            cached_distances = cache_results.get("distances", [[]])[0]

            if (
                cached_documents
                and cached_documents[0]
                and cached_distances
                and cached_distances[0] is not None
            ):
                distance = float(cached_distances[0])

                if distance <= self.cache_hit_threshold:
                    return {
                        "answer": cached_documents[0],
                        "cache_distance": distance,
                    }

            return None

        except Exception:
            logger.exception(
                "Query cache lookup failed"
            )
            return None

    def _store_query_cache(
        self,
        query: str,
        query_embedding: List[float],
        answer: str,
    ) -> None:
        """
        Store a generated answer in the semantic query cache.

        Args:
            query: User query.
            query_embedding: Query embedding vector.
            answer: Generated answer text.
        """

        try:
            self.query_cache_db.add_documents(
                ids=[f"cache_{uuid.uuid4()}"],
                embeddings=[query_embedding],
                documents=[answer],
                metadatas=[
                    {
                        "query": query,
                        "cached_at": int(time.time()),
                        "source_collection": self.vector_db.collection_name,
                    }
                ],
            )

            logger.info("Cached answer stored successfully")

        except Exception:
            logger.exception(
                "Failed to store answer in query cache"
            )

    # --------------------------------------------------------------------------
    # Main QA Pipeline
    # --------------------------------------------------------------------------

    async def get_answer(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate answer using RAG pipeline.

        Args:
            query: User query.
            top_k: Override retrieval count.

        Returns:
            Dictionary containing answer and metadata.
        """

        start_time = time.time()

        try:
            # ------------------------------------------------------------------
            # Validate Input
            # ------------------------------------------------------------------

            if not query.strip():
                raise ValueError("Query cannot be empty")

            retrieval_count = top_k or self.top_k

            logger.info(
                "Starting RAG pipeline | query=%s",
                query[:100],
            )

            # ------------------------------------------------------------------
            # Step 1: Query Embedding
            # ------------------------------------------------------------------

            query_embedding = self.embedding_service.embed(
                sentences=[query]
            )[0]

            logger.info("Query embedding generated")

            # ------------------------------------------------------------------
            # Step 1.5: Semantic Query Cache
            # ------------------------------------------------------------------

            cached_entry = self._get_cached_answer(
                query_embedding=query_embedding,
            )

            if cached_entry is not None:
                elapsed_time = round(
                    time.time() - start_time,
                    2,
                )

                logger.info(
                    "Cache hit for semantic query | distance=%s",
                    cached_entry["cache_distance"],
                )

                return {
                    "status": "success",
                    "query": query,
                    "answer": cached_entry["answer"],
                    "retrieved_chunks": 0,
                    "cache_hit": True,
                    "cache_distance": cached_entry["cache_distance"],
                    "processing_time_seconds": elapsed_time,
                }

            # ------------------------------------------------------------------
            # Step 2: Vector Retrieval
            # ------------------------------------------------------------------

            retrieval_results = self.vector_db.query_documents(
                query_embedding=query_embedding,
                n_results=retrieval_count,
            )

            documents: List[List[str]] = retrieval_results.get(
                "documents",
                []
            )

            if not documents or not documents[0]:
                raise RetrievalError(
                    "No relevant documents found"
                )

            retrieved_chunks = documents[0]

            logger.info(
                "Document retrieval completed | chunks=%d",
                len(retrieved_chunks),
            )

            # ------------------------------------------------------------------
            # Step 3: Context Construction
            # ------------------------------------------------------------------

            relevant_context = "\n\n".join(retrieved_chunks)

            system_prompt = self._build_system_prompt(
                relevant_context
            )

            # ------------------------------------------------------------------
            # Step 4: LLM Generation
            # ------------------------------------------------------------------

            response = await asyncio.to_thread(
                self.llm_service.get_response,
                system_prompt,
                query,
                self.llm_model,
            )

            self._store_query_cache(
                query=query,
                query_embedding=query_embedding,
                answer=response,
            )

            elapsed_time = round(
                time.time() - start_time,
                2,
            )

            logger.info(
                "RAG pipeline completed successfully "
                "| duration=%ss",
                elapsed_time,
            )

            return {
                "status": "success",
                "query": query,
                "answer": response,
                "retrieved_chunks": len(retrieved_chunks),
                "cache_hit": False,
                "processing_time_seconds": elapsed_time,
            }

        except RetrievalError:
            logger.exception("Retrieval failed")
            raise

        except Exception as exc:
            logger.exception(
                "Unexpected error in RAG pipeline"
            )

            raise GenerationError(
                "Failed to generate answer"
            ) from exc

    # --------------------------------------------------------------------------
    # Prompt Builder
    # --------------------------------------------------------------------------

    @staticmethod
    def _build_system_prompt(context: str) -> str:
        """
        Construct safe system prompt.

        Args:
            context: Retrieved context.

        Returns:
            System prompt string.
        """

        return f"""
You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say:
"I could not find the answer in the provided documents."

Context:
{context}
""".strip()


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

async def main() -> None:

    try:
        rag_service = RAGService(
            collection_name="test_collection",
            top_k=5,
        )

        result = await rag_service.get_answer(
            query="What is Retrieval-Augmented Generation?"
        )

        print("\nAnswer:\n")
        print(result["answer"])

    except Exception:
        logger.exception("Application execution failed")


if __name__ == "__main__":
    asyncio.run(main())
