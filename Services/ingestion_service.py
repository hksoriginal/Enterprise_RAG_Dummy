import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from Services.chroma_db_service import ChromaDBService
from Services.document_handler import DocumentHandler
from Services.embedding_service import EmbeddingService


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

class IngestionServiceError(Exception):
    """Base exception for ingestion service."""


class DocumentIngestionError(IngestionServiceError):
    """Raised when document ingestion fails."""


# ------------------------------------------------------------------------------
# Ingestion Service
# ------------------------------------------------------------------------------

class IngestionService:
    """
    Production-grade ingestion pipeline.

    Workflow:
        PDF -> Chunking -> Embeddings -> Vector DB

    Features:
    - Structured logging
    - Exception handling
    - Batch processing
    - Metadata enrichment
    - Input validation
    - UUID-based document IDs
    - Timing metrics
    """

    def __init__(
        self,
        collection_name: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        embedding_batch_size: int = 32,
    ) -> None:
        """
        Initialize ingestion pipeline.

        Args:
            collection_name: ChromaDB collection name.
            chunk_size: Default chunk size.
            chunk_overlap: Default chunk overlap.
            embedding_batch_size: Embedding batch size.
        """

        logger.info(
            "Initializing ingestion service | collection=%s",
            collection_name,
        )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_batch_size = embedding_batch_size

        self.document_handler = DocumentHandler()
        self.embedding_service = EmbeddingService()
        self.chroma_db_service = ChromaDBService(collection_name)

        logger.info("Ingestion service initialized successfully")

    # --------------------------------------------------------------------------
    # Document Ingestion
    # --------------------------------------------------------------------------

    def ingest_document(
        self,
        file_path: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a document into vector database.

        Args:
            file_path: Path to PDF document.
            chunk_size: Override default chunk size.
            chunk_overlap: Override default overlap.
            additional_metadata: Extra metadata fields.

        Returns:
            Dictionary containing ingestion summary.

        Raises:
            DocumentIngestionError
        """

        start_time = time.time()

        try:
            # ------------------------------------------------------------------
            # Validate Inputs
            # ------------------------------------------------------------------

            path = Path(file_path)

            if not path.exists():
                raise FileNotFoundError(
                    f"Document not found: {file_path}"
                )

            if path.suffix.lower() != ".pdf":
                raise ValueError(
                    "Only PDF files are supported"
                )

            chunk_size = chunk_size or self.chunk_size
            chunk_overlap = chunk_overlap or self.chunk_overlap

            logger.info(
                "Starting document ingestion | file=%s",
                file_path,
            )

            # ------------------------------------------------------------------
            # Chunk Generation
            # ------------------------------------------------------------------

            chunks = self.document_handler.get_chunks(
                file_path=file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            if not chunks:
                raise ValueError(
                    "No chunks generated from document"
                )

            logger.info(
                "Document chunking completed | chunks=%d",
                len(chunks),
            )

            # ------------------------------------------------------------------
            # Generate Embeddings
            # ------------------------------------------------------------------

            embeddings = self.embedding_service.embed(
                sentences=chunks,
                batch_size=self.embedding_batch_size,
            )

            logger.info(
                "Embedding generation completed | embeddings=%d",
                len(embeddings),
            )

            # ------------------------------------------------------------------
            # Generate IDs & Metadata
            # ------------------------------------------------------------------

            document_id = str(uuid.uuid4())

            ids: List[str] = [
                f"{document_id}_chunk_{i}"
                for i in range(len(chunks))
            ]

            metadatas: List[Dict[str, Any]] = []

            for chunk_index in range(len(chunks)):

                metadata = {
                    "document_id": document_id,
                    "file_name": path.name,
                    "file_path": str(path.resolve()),
                    "chunk_index": chunk_index,
                    "total_chunks": len(chunks),
                    "chunk_size": chunk_size,
                    "ingestion_timestamp": int(time.time()),
                }

                if additional_metadata:
                    metadata.update(additional_metadata)

                metadatas.append(metadata)

            # ------------------------------------------------------------------
            # Store in ChromaDB
            # ------------------------------------------------------------------

            self.chroma_db_service.add_documents(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
            )

            elapsed_time = round(time.time() - start_time, 2)

            logger.info(
                "Document ingestion completed successfully "
                "| file=%s | chunks=%d | duration=%ss",
                file_path,
                len(chunks),
                elapsed_time,
            )

            return {
                "status": "success",
                "document_id": document_id,
                "file_name": path.name,
                "total_chunks": len(chunks),
                "collection_name": self.chroma_db_service.collection_name,
                "processing_time_seconds": elapsed_time,
            }

        except (FileNotFoundError, ValueError) as exc:
            logger.exception(
                "Document ingestion validation failed"
            )

            raise DocumentIngestionError(
                f"Invalid ingestion request: {str(exc)}"
            ) from exc

        except Exception as exc:
            logger.exception(
                "Unexpected error during document ingestion"
            )

            raise DocumentIngestionError(
                "Document ingestion failed"
            ) from exc

    # --------------------------------------------------------------------------
    # Batch Ingestion
    # --------------------------------------------------------------------------

    def ingest_documents(
        self,
        file_paths: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Ingest multiple documents.

        Args:
            file_paths: List of PDF paths.

        Returns:
            List of ingestion results.
        """

        logger.info(
            "Starting batch ingestion | files=%d",
            len(file_paths),
        )

        results = []

        for file_path in file_paths:

            try:
                result = self.ingest_document(file_path)
                results.append(result)

            except Exception as exc:
                logger.error(
                    "Failed to ingest file | file=%s | error=%s",
                    file_path,
                    str(exc),
                )

                results.append(
                    {
                        "status": "failed",
                        "file_path": file_path,
                        "error": str(exc),
                    }
                )

        logger.info("Batch ingestion completed")

        return results


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":

    try:
        ingestion_service = IngestionService(
            collection_name="research_papers",
            chunk_size=500,
            chunk_overlap=100,
            embedding_batch_size=32,
        )

        result = ingestion_service.ingest_document(
            file_path="Documents/paper.pdf",
            additional_metadata={
                "source": "research_archive",
                "category": "machine_learning",
            },
        )

        print(result)

    except Exception:
        logger.exception("Application execution failed")
