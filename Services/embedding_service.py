import logging
from threading import Lock
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer


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

class EmbeddingServiceError(Exception):
    """Base exception for embedding service."""


class ModelInitializationError(EmbeddingServiceError):
    """Raised when embedding model loading fails."""


class EmbeddingGenerationError(EmbeddingServiceError):
    """Raised when embedding generation fails."""


# ------------------------------------------------------------------------------
# Embedding Service
# ------------------------------------------------------------------------------

class EmbeddingService:
    """
    Production-grade embedding service using Sentence Transformers.

    Features:
    - Lazy model loading
    - Thread-safe singleton model initialization
    - Structured logging
    - Input validation
    - Batch embedding support
    - Configurable normalization/device
    """

    _model = None
    _lock = Lock()

    def __init__(
        self,
        model_path: str = "Embedding_Model/paraphrase-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        """
        Initialize embedding service.

        Args:
            model_path: Path or HF model name.
            device: cpu / cuda / mps
        """

        self.model_path = model_path
        self.device = device

        self._initialize_model()

    # --------------------------------------------------------------------------
    # Model Initialization
    # --------------------------------------------------------------------------

    def _initialize_model(self) -> None:
        """
        Lazy-load embedding model safely.

        Raises:
            ModelInitializationError
        """

        try:
            if EmbeddingService._model is None:

                with EmbeddingService._lock:

                    if EmbeddingService._model is None:

                        logger.info(
                            "Loading embedding model | model=%s | device=%s",
                            self.model_path,
                            self.device,
                        )

                        EmbeddingService._model = SentenceTransformer(
                            self.model_path,
                            device=self.device,
                        )

                        logger.info(
                            "Embedding model loaded successfully"
                        )

        except Exception as exc:
            logger.exception(
                "Failed to initialize embedding model"
            )

            raise ModelInitializationError(
                f"Unable to load model: {self.model_path}"
            ) from exc

    # --------------------------------------------------------------------------
    # Generate Embeddings
    # --------------------------------------------------------------------------

    def embed(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        convert_to_list: bool = True,
    ) -> List[List[float]]:
        """
        Generate embeddings for input text.

        Args:
            sentences: Single sentence or list of sentences.
            batch_size: Batch size for encoding.
            normalize_embeddings: Whether to L2 normalize embeddings.
            convert_to_list: Convert numpy output to Python list.

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingGenerationError
        """

        try:
            # ------------------------------------------------------------------
            # Normalize Input
            # ------------------------------------------------------------------

            if isinstance(sentences, str):
                sentences = [sentences]

            if not isinstance(sentences, list):
                raise ValueError(
                    "sentences must be a string or list of strings"
                )

            if not sentences:
                raise ValueError("sentences cannot be empty")

            cleaned_sentences = []

            for sentence in sentences:

                if not isinstance(sentence, str):
                    raise ValueError(
                        "All inputs must be strings"
                    )

                cleaned_sentence = sentence.strip()

                if cleaned_sentence:
                    cleaned_sentences.append(cleaned_sentence)

            if not cleaned_sentences:
                raise ValueError(
                    "No valid non-empty sentences found"
                )

            logger.info(
                "Generating embeddings | count=%d | batch_size=%d",
                len(cleaned_sentences),
                batch_size,
            )

            # ------------------------------------------------------------------
            # Embedding Generation
            # ------------------------------------------------------------------

            embeddings = EmbeddingService._model.encode(
                cleaned_sentences,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            logger.info(
                "Embedding generation completed successfully"
            )

            # ------------------------------------------------------------------
            # Convert Output
            # ------------------------------------------------------------------

            if convert_to_list:
                return embeddings.tolist()

            return embeddings

        except ValueError as exc:
            logger.exception(
                "Embedding validation failed"
            )

            raise EmbeddingGenerationError(
                "Invalid embedding input"
            ) from exc

        except Exception as exc:
            logger.exception(
                "Unexpected error during embedding generation"
            )

            raise EmbeddingGenerationError(
                "Failed to generate embeddings"
            ) from exc

    # --------------------------------------------------------------------------
    # Utility Methods
    # --------------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(
        embedding_1: List[float],
        embedding_2: List[float],
    ) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            embedding_1: First embedding vector.
            embedding_2: Second embedding vector.

        Returns:
            Cosine similarity score.
        """

        try:
            vec1 = np.array(embedding_1)
            vec2 = np.array(embedding_2)

            similarity = np.dot(vec1, vec2) / (
                np.linalg.norm(vec1) * np.linalg.norm(vec2)
            )

            return float(similarity)

        except Exception as exc:
            logger.exception(
                "Failed to compute cosine similarity"
            )

            raise EmbeddingServiceError(
                "Cosine similarity computation failed"
            ) from exc


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":

    try:
        embedding_service = EmbeddingService(
            model_path="Embedding_Model/paraphrase-MiniLM-L6-v2",
            device="cpu",
        )

        sample_sentences = [
            "Machine learning is fascinating.",
            "Artificial intelligence is transforming industries.",
        ]

        embeddings = embedding_service.embed(sample_sentences)

        logger.info(
            "Generated embeddings successfully | vectors=%d",
            len(embeddings),
        )

        print(f"Embedding dimension: {len(embeddings[0])}")

    except Exception:
        logger.exception("Application execution failed")
