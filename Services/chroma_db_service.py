import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.errors import ChromaError


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

class ChromaDBServiceError(Exception):
    """Base exception for ChromaDB service."""


class CollectionInitializationError(ChromaDBServiceError):
    """Raised when collection initialization fails."""


class DocumentInsertionError(ChromaDBServiceError):
    """Raised when document insertion fails."""


class QueryExecutionError(ChromaDBServiceError):
    """Raised when query execution fails."""


# ------------------------------------------------------------------------------
# ChromaDB Service
# ------------------------------------------------------------------------------

class ChromaDBService:
    """
    Production-grade wrapper around ChromaDB.

    Features:
    - Structured logging
    - Exception handling
    - Input validation
    - Type hints
    - Better maintainability
    """

    def __init__(
        self,
        collection_name: str,
        db_path: str = "./Database",
        distance_metric: str = "cosine",
    ) -> None:
        """
        Initialize ChromaDB persistent client and collection.

        Args:
            collection_name: Name of the Chroma collection.
            db_path: Persistent storage directory.
            distance_metric: HNSW similarity metric.
        """

        self.collection_name = collection_name
        self.db_path = db_path

        try:
            logger.info(
                "Initializing ChromaDB client | path=%s | collection=%s",
                db_path,
                collection_name,
            )

            Path(db_path).mkdir(parents=True, exist_ok=True)

            self.client = chromadb.PersistentClient(path=db_path)

            self.collection: Collection = (
                self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": distance_metric},
                )
            )

            logger.info(
                "ChromaDB collection initialized successfully | collection=%s",
                collection_name,
            )

        except Exception as exc:
            logger.exception(
                "Failed to initialize ChromaDB collection | collection=%s",
                collection_name,
            )
            raise CollectionInitializationError(
                f"Failed to initialize collection '{collection_name}'"
            ) from exc

    # --------------------------------------------------------------------------
    # Add Documents
    # --------------------------------------------------------------------------

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Insert documents into ChromaDB collection.

        Args:
            ids: Unique document IDs.
            embeddings: Vector embeddings.
            documents: Raw text documents.
            metadatas: Optional metadata dictionaries.

        Raises:
            DocumentInsertionError
        """

        try:
            self._validate_add_inputs(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            logger.info(
                "Adding documents to collection | collection=%s | count=%d",
                self.collection_name,
                len(ids),
            )

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            logger.info(
                "Documents added successfully | collection=%s | count=%d",
                self.collection_name,
                len(ids),
            )

        except (ValueError, ChromaError) as exc:
            logger.exception(
                "Document insertion failed | collection=%s",
                self.collection_name,
            )

            raise DocumentInsertionError(
                "Failed to insert documents into ChromaDB"
            ) from exc

        except Exception as exc:
            logger.exception(
                "Unexpected error during document insertion"
            )

            raise DocumentInsertionError(
                "Unexpected error during document insertion"
            ) from exc

    # --------------------------------------------------------------------------
    # Query Documents
    # --------------------------------------------------------------------------

    def query_documents(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query documents using vector similarity search.

        Args:
            query_embedding: Query vector embedding.
            n_results: Number of results to retrieve.
            where: Optional metadata filter.

        Returns:
            Query results dictionary.

        Raises:
            QueryExecutionError
        """

        try:
            if not query_embedding:
                raise ValueError("query_embedding cannot be empty")

            if n_results <= 0:
                raise ValueError("n_results must be greater than 0")

            logger.info(
                "Executing vector search | collection=%s | n_results=%d",
                self.collection_name,
                n_results,
            )

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
            )

            logger.info(
                "Query executed successfully | collection=%s",
                self.collection_name,
            )

            return results

        except (ValueError, ChromaError) as exc:
            logger.exception(
                "Query execution failed | collection=%s",
                self.collection_name,
            )

            raise QueryExecutionError(
                "Failed to query documents from ChromaDB"
            ) from exc

        except Exception as exc:
            logger.exception(
                "Unexpected error during query execution"
            )

            raise QueryExecutionError(
                "Unexpected error during query execution"
            ) from exc

    # --------------------------------------------------------------------------
    # Health Check
    # --------------------------------------------------------------------------

    def health_check(self) -> bool:
        """
        Verify ChromaDB connectivity.

        Returns:
            bool: True if healthy, else False
        """

        try:
            self.client.heartbeat()
            logger.info("ChromaDB health check passed")
            return True

        except Exception:
            logger.exception("ChromaDB health check failed")
            return False

    # --------------------------------------------------------------------------
    # Private Validators
    # --------------------------------------------------------------------------

    @staticmethod
    def _validate_add_inputs(
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]],
    ) -> None:
        """
        Validate insertion payload consistency.
        """

        if not ids:
            raise ValueError("ids cannot be empty")

        if not embeddings:
            raise ValueError("embeddings cannot be empty")

        if not documents:
            raise ValueError("documents cannot be empty")

        if not (
            len(ids)
            == len(embeddings)
            == len(documents)
        ):
            raise ValueError(
                "ids, embeddings, and documents must have equal length"
            )

        if metadatas and len(metadatas) != len(ids):
            raise ValueError(
                "metadatas length must match ids length"
            )
