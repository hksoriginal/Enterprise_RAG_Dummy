import logging
from typing import Any, Dict, List

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    status,
)
from pydantic import (
    BaseModel,
    Field,
    validator,
)

from Services.chroma_db_service import (
    ChromaDBService,
    QueryExecutionError,
)
from Services.embedding_service import (
    EmbeddingGenerationError,
    EmbeddingService,
)


# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# ------------------------------------------------------------------------------
# Router Initialization
# ------------------------------------------------------------------------------

retrieval_router = APIRouter(
    prefix="/retrieval",
    tags=["Vector Retrieval"],
)


# ------------------------------------------------------------------------------
# Service Initialization (Singleton Style)
# ------------------------------------------------------------------------------

embedding_service = EmbeddingService()

chroma_db_service = ChromaDBService(
    collection_name="test_collection"
)


# ------------------------------------------------------------------------------
# Request Schema
# ------------------------------------------------------------------------------

class QueryPayload(BaseModel):
    """
    Request payload for semantic retrieval.
    """

    query_text: str = Field(
        ...,
        min_length=3,
        max_length=5000,
        description="Search query text",
        example="What is Retrieval-Augmented Generation?",
    )

    n_results: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of retrieval results",
    )

    @validator("query_text")
    def validate_query_text(
        cls,
        value: str,
    ) -> str:
        """
        Validate and sanitize query.
        """

        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError(
                "query_text cannot be empty"
            )

        return cleaned_value


# ------------------------------------------------------------------------------
# Response Schema
# ------------------------------------------------------------------------------

class RetrievalResponse(BaseModel):

    status: str
    query: str
    total_results: int
    documents: List[str]
    metadatas: List[Dict[str, Any]]
    distances: List[float]


# ------------------------------------------------------------------------------
# Health Check Endpoint
# ------------------------------------------------------------------------------

@retrieval_router.get(
    "/health",
    status_code=status.HTTP_200_OK,
)
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    """

    return {
        "status": "healthy",
        "service": "retrieval-api",
    }


# ------------------------------------------------------------------------------
# Retrieval Endpoint
# ------------------------------------------------------------------------------

@retrieval_router.post(
    "/retrieve",
    response_model=RetrievalResponse,
    status_code=status.HTTP_200_OK,
)
async def retrieve(
    query_payload: QueryPayload,
    request: Request,
) -> RetrievalResponse:
    """
    Semantic vector retrieval endpoint.
    """

    try:
        client_host = (
            request.client.host
            if request.client
            else "unknown"
        )

        logger.info(
            "Received retrieval request | client=%s",
            client_host,
        )

        # ----------------------------------------------------------------------
        # Generate Query Embedding
        # ----------------------------------------------------------------------

        embedding = embedding_service.embed(
            [query_payload.query_text]
        )[0]

        logger.info(
            "Query embedding generated successfully"
        )

        # ----------------------------------------------------------------------
        # Retrieve Documents
        # ----------------------------------------------------------------------

        result = chroma_db_service.query_documents(
            query_embedding=embedding,
            n_results=query_payload.n_results,
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        logger.info(
            "Vector retrieval completed successfully "
            "| results=%d",
            len(documents),
        )

        return RetrievalResponse(
            status="success",
            query=query_payload.query_text,
            total_results=len(documents),
            documents=documents,
            metadatas=metadatas,
            distances=distances,
        )

    except EmbeddingGenerationError as exc:

        logger.exception(
            "Embedding generation failed"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate query embedding",
        ) from exc

    except QueryExecutionError as exc:

        logger.exception(
            "Vector retrieval failed"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Vector database query failed",
        ) from exc

    except Exception as exc:

        logger.exception(
            "Unexpected retrieval API error"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from exc
