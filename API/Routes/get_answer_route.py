import logging
from typing import Any, Dict

from Guardrails.input_guardrails import InputGuardrails
from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    status,
)
from pydantic import BaseModel, Field, validator

from Utils.get_answer import RAGService
from Utils.get_answer import (
    GenerationError,
    RetrievalError,
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

get_answer_router = APIRouter(
    prefix="/rag",
    tags=["RAG QA"],
)


# ------------------------------------------------------------------------------
# Initialize Services Once
# ------------------------------------------------------------------------------

rag_service = RAGService(
    collection_name="test_collection",
    top_k=5,
)
input_guardrail = InputGuardrails()


# ------------------------------------------------------------------------------
# Request Schema
# ------------------------------------------------------------------------------

class QueryPayload(BaseModel):
    """
    Request payload for question-answering endpoint.
    """

    query: str = Field(
        ...,
        min_length=3,
        max_length=5000,
        description="User query",
        example="What is Retrieval-Augmented Generation?",
    )

    @validator("query")
    def validate_query(cls, value: str) -> str:
        """
        Validate and sanitize query input.
        """

        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError(
                "Query cannot be empty"
            )

        return cleaned_value


# ------------------------------------------------------------------------------
# Health Check Endpoint
# ------------------------------------------------------------------------------

@get_answer_router.get(
    "/health",
    status_code=status.HTTP_200_OK,
)
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    """

    logger.info("Health check requested")

    return {
        "status": "healthy",
        "service": "rag-api",
    }


# ------------------------------------------------------------------------------
# Main QA Endpoint
# ------------------------------------------------------------------------------

@get_answer_router.post(
    "/get-answer",
    status_code=status.HTTP_200_OK,
)
async def get_answer_route(
    query_payload: QueryPayload,
    request: Request,
) -> Dict[str, Any]:
    """
    RAG-based question-answering endpoint.
    """

    try:
        client_host = (
            request.client.host
            if request.client
            else "unknown"
        )

        logger.info(
            "Received QA request | client=%s",
            client_host,
        )

        input_query = query_payload.query
        clean_query, guardrail_triggered = input_guardrail.run_input_guardrails(
            input_query
        )

        answer = clean_query
        if not guardrail_triggered:

            # ---------------------------------------------------
            # Generate Answer
            # ---------------------------------------------------
            answer = await rag_service.get_answer(
                query=clean_query,
                top_k=5,
            )

        logger.info(
            "QA request completed successfully"
        )

        return answer

    except RetrievalError as exc:

        logger.exception(
            "Retrieval failure during QA request"
        )

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except GenerationError as exc:

        logger.exception(
            "Generation failure during QA request"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate response",
        ) from exc

    except Exception as exc:

        logger.exception(
            "Unexpected API error"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from exc
