import logging
import re
from pathlib import Path
from typing import List

import pypdf
from pypdf.errors import PdfReadError


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

class DocumentHandlerError(Exception):
    """Base exception for document handler."""


class DocumentReadError(DocumentHandlerError):
    """Raised when PDF reading fails."""


class ChunkingError(DocumentHandlerError):
    """Raised when chunk generation fails."""


# ------------------------------------------------------------------------------
# Document Handler
# ------------------------------------------------------------------------------

class DocumentHandler:
    """
    Production-grade PDF document handler.

    Features:
    - Exception handling
    - Structured logging
    - Input validation
    - Safe PDF extraction
    - Memory-efficient processing
    """

    def __init__(self) -> None:
        logger.info("DocumentHandler initialized")

    # --------------------------------------------------------------------------
    # Read PDF Document
    # --------------------------------------------------------------------------

    def _read_document(self, file_path: str) -> str:
        """
        Read and extract text from a PDF document.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Extracted text as string.

        Raises:
            DocumentReadError
        """

        try:
            path = Path(file_path)

            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            if path.suffix.lower() != ".pdf":
                raise ValueError("Only PDF files are supported")

            logger.info("Reading PDF document | file=%s", file_path)

            reader = pypdf.PdfReader(file_path)

            if len(reader.pages) == 0:
                raise ValueError("PDF contains no pages")

            extracted_text: List[str] = []

            for page_number, page in enumerate(reader.pages, start=1):
                try:
                    page_text = page.extract_text()

                    if page_text:
                        extracted_text.append(page_text)

                except Exception as exc:
                    logger.warning(
                        "Failed to extract page | page=%d | error=%s",
                        page_number,
                        str(exc),
                    )

            final_text = " ".join(extracted_text)

            if not final_text.strip():
                raise ValueError("No extractable text found in PDF")

            logger.info(
                "PDF text extraction completed | pages=%d | characters=%d",
                len(reader.pages),
                len(final_text),
            )

            return final_text

        except (FileNotFoundError, PdfReadError, ValueError) as exc:
            logger.exception(
                "Failed to read PDF document | file=%s",
                file_path,
            )

            raise DocumentReadError(
                f"Unable to process PDF file: {file_path}"
            ) from exc

        except Exception as exc:
            logger.exception(
                "Unexpected error during PDF reading"
            )

            raise DocumentReadError(
                "Unexpected error while reading document"
            ) from exc

    # --------------------------------------------------------------------------
    # Clean Text
    # --------------------------------------------------------------------------

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean extracted text.

        Args:
            text: Raw extracted text.

        Returns:
            Cleaned text.
        """

        try:
            if not text:
                return ""

            logger.debug("Cleaning extracted text")

            # Remove excessive whitespace
            text = re.sub(r"\s+", " ", text)

            # Remove non-printable characters
            text = re.sub(r"[^\x20-\x7E]+", " ", text)

            # Final cleanup
            text = text.strip()

            return text

        except Exception:
            logger.exception("Text cleaning failed")
            return text

    # --------------------------------------------------------------------------
    # Chunk Generation
    # --------------------------------------------------------------------------

    def get_chunks(
        self,
        file_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> List[str]:
        """
        Split document into overlapping chunks.

        Args:
            file_path: PDF file path.
            chunk_size: Size of each chunk.
            chunk_overlap: Overlap between chunks.

        Returns:
            List of text chunks.

        Raises:
            ChunkingError
        """

        try:
            if chunk_size <= 0:
                raise ValueError("chunk_size must be greater than 0")

            if chunk_overlap < 0:
                raise ValueError("chunk_overlap cannot be negative")

            if chunk_overlap >= chunk_size:
                raise ValueError(
                    "chunk_overlap must be smaller than chunk_size"
                )

            logger.info(
                "Generating chunks | chunk_size=%d | overlap=%d",
                chunk_size,
                chunk_overlap,
            )

            text = self._read_document(file_path)
            text = self.clean_text(text)

            if not text:
                raise ValueError("Document text is empty after cleaning")

            chunks: List[str] = []

            step = chunk_size - chunk_overlap

            for i in range(0, len(text), step):
                chunk = text[i: i + chunk_size]

                if chunk.strip():
                    chunks.append(chunk)

            logger.info(
                "Chunk generation completed | total_chunks=%d",
                len(chunks),
            )

            return chunks

        except ValueError as exc:
            logger.exception("Chunking validation failed")

            raise ChunkingError(
                "Invalid chunking configuration"
            ) from exc

        except Exception as exc:
            logger.exception(
                "Unexpected error during chunk generation"
            )

            raise ChunkingError(
                "Failed to generate document chunks"
            ) from exc


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":

    try:
        document_handler = DocumentHandler()

        chunks = document_handler.get_chunks(
            file_path="Documents/paper.pdf",
            chunk_size=500,
            chunk_overlap=100,
        )

        logger.info("Printing generated chunks")

        for index, chunk in enumerate(chunks, start=1):
            print(f"\nChunk {index}")
            print("-" * 100)
            print(chunk)

    except Exception as exc:
        logger.exception("Application execution failed")
