import logging
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    HTTPException,
    Request,
    status,
)
from pydantic import (
    BaseModel,
    Field,
    validator,
)

from Services.ingestion_service import (
    DocumentIngestionError,
    IngestionService,
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

ingest_router = APIRouter(
    prefix="/documents",
    tags=["Document Ingestion"],
)


# ------------------------------------------------------------------------------
# Service Initialization
# ------------------------------------------------------------------------------

ingestion_service = IngestionService(
    collection_name="test_collection",
    chunk_size=500,
    chunk_overlap=100,
)


# ------------------------------------------------------------------------------
# Request Schema
# ------------------------------------------------------------------------------

class IngestPayload(BaseModel):
    """
    Request payload for document ingestion.
    """

    file_path: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="Path to PDF document",
        example="Documents/paper.pdf",
    )

    chunk_size: int = Field(
        default=500,
        ge=100,
        le=5000,
        description="Chunk size for text splitting",
    )

    chunk_overlap: int = Field(
        default=100,
        ge=0,
        le=1000,
        description="Chunk overlap size",
    )

    additional_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata fields",
    )

    # --------------------------------------------------------------------------
    # Validators
    # --------------------------------------------------------------------------

    @validator("file_path")
    def validate_file_path(cls, value: str) -> str:

        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError(
                "file_path cannot be empty"
            )

        path = Path(cleaned_value)

        if path.suffix.lower() != ".pdf":
            raise ValueError(
                "Only PDF files are supported"
            )

        return cleaned_value

    @validator("chunk_overlap")
    def validate_overlap(
        cls,
        overlap: int,
        values: Dict[str, Any],
    ) -> int:

        chunk_size = values.get("chunk_size")

        if (
            chunk_size is not None
            and overlap >= chunk_size
        ):
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        return overlap


# ------------------------------------------------------------------------------
# Response Schema
# ------------------------------------------------------------------------------

class IngestResponse(BaseModel):

    status: str
    message: str
    file_path: str


# ------------------------------------------------------------------------------
# Background Ingestion Task
# ------------------------------------------------------------------------------

def run_ingestion_task(
    ingest_payload: IngestPayload,
) -> None:
    """
    Execute ingestion task safely in background.
    """

    try:
        logger.info(
            "Starting background ingestion | file=%s",
            ingest_payload.file_path,
        )

        ingestion_service.ingest_document(
            file_path=ingest_payload.file_path,
            chunk_size=ingest_payload.chunk_size,
            chunk_overlap=ingest_payload.chunk_overlap,
            additional_metadata=(
                ingest_payload.additional_metadata
            ),
        )

        logger.info(
            "Background ingestion completed successfully"
        )

    except Exception:
        logger.exception(
            "Background ingestion task failed"
        )


# ------------------------------------------------------------------------------
# Health Check Endpoint
# ------------------------------------------------------------------------------

@ingest_router.get(
    "/health",
    status_code=status.HTTP_200_OK,
)
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    """

    return {
        "status": "healthy",
        "service": "document-ingestion-api",
    }


# ------------------------------------------------------------------------------
# Document Ingestion Endpoint
# ------------------------------------------------------------------------------

@ingest_router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def ingest_document(
    ingest_payload: IngestPayload,
    background_tasks: BackgroundTasks,
    request: Request,
) -> IngestResponse:
    """
    Ingest PDF document into vector database.
    """

    try:
        client_host = (
            request.client.host
            if request.client
            else "unknown"
        )

        logger.info(
            "Received ingestion request | client=%s | file=%s",
            client_host,
            ingest_payload.file_path,
        )

        # ----------------------------------------------------------------------
        # Validate File Existence
        # ----------------------------------------------------------------------

        file_path = Path(
            ingest_payload.file_path
        )

        if not file_path.exists():

            logger.warning(
                "Requested file does not exist | file=%s",
                ingest_payload.file_path,
            )

            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document file not found",
            )

        # ----------------------------------------------------------------------
        # Execute Ingestion in Background
        # ----------------------------------------------------------------------

        background_tasks.add_task(
            run_ingestion_task,
            ingest_payload,
        )

        logger.info(
            "Document ingestion task scheduled successfully"
        )

        return IngestResponse(
            status="accepted",
            message=(
                "Document ingestion started successfully"
            ),
            file_path=ingest_payload.file_path,
        )

    except HTTPException:
        raise

    except DocumentIngestionError as exc:

        logger.exception(
            "Document ingestion failed"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        logger.exception(
            "Unexpected error during ingestion request"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from exc
