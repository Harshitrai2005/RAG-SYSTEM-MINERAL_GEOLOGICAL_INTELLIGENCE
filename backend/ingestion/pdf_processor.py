"""
PDF Processor
Extracts text from geological survey reports (PDFs).
Handles multi-page PDFs, table extraction, and metadata capture.
Produces clean, chunked text ready for embedding.
"""

import hashlib
import os
import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from core.config import settings
from utils.logger import setup_logger
from utils.text_chunker import TextChunker

logger = setup_logger(__name__)


class PDFProcessor:
    """
    Processes PDF geological reports into embeddable text chunks.

    Strategy:
    - Extract text page-by-page (preserves page references for citations)
    - Clean OCR artifacts and formatting noise
    - Detect section headers for better semantic chunking
    - Split into overlapping chunks sized for the embedding model
    """

    def __init__(self):
        self.chunker = TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def process_file(self, file_path: str | Path) -> list[dict]:
        """
        Process a single PDF file.

        Returns:
            List of document dicts ready for vector store ingestion:
            [{id, text, metadata: {source, doc_type, page, section, ...}}]
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")
        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {file_path.name}")

        logger.info(f"Processing PDF: {file_path.name}")

        doc = fitz.open(str(file_path))
        all_chunks = []

        file_hash = self._compute_hash(file_path)
        total_pages = len(doc)

        for page_num in range(total_pages):
            page = doc[page_num]
            raw_text = page.get_text("text")

            if not raw_text.strip():
                logger.debug(f"  Page {page_num + 1} appears empty — skipping")
                continue

            cleaned_text = self._clean_text(raw_text)
            section = self._detect_section(cleaned_text)

            chunks = self.chunker.split(cleaned_text)

            for chunk_idx, chunk_text in enumerate(chunks):
                if len(chunk_text.strip()) < 50:
                    continue  # Skip near-empty chunks

                doc_id = f"{file_hash}_p{page_num+1}_c{chunk_idx}"
                all_chunks.append(
                    {
                        "id": doc_id,
                        "text": chunk_text,
                        "metadata": {
                            "source": file_path.name,
                            "source_path": str(file_path),
                            "doc_type": "geological_report",
                            "file_type": "pdf",
                            "page": page_num + 1,
                            "total_pages": total_pages,
                            "section": section or "Unknown",
                            "chunk_index": chunk_idx,
                            "file_hash": file_hash,
                        },
                    }
                )

        doc.close()
        logger.info(f"  Extracted {len(all_chunks)} chunks from {total_pages} pages")
        return all_chunks

    def process_directory(self, directory: str | Path) -> list[dict]:
        """Process all PDFs in a directory recursively."""
        directory = Path(directory)
        all_documents = []

        pdf_files = list(directory.rglob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")

        for pdf_file in pdf_files:
            try:
                chunks = self.process_file(pdf_file)
                all_documents.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")

        return all_documents

    def _clean_text(self, text: str) -> str:
        """
        Remove common PDF artifacts while preserving geological content.
        Geological numbers, coordinates, and chemical formulas must be preserved.
        """
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove page headers/footers patterns (page numbers, repeated headers)
        text = re.sub(r"\n\d+\s*\n", "\n", text)
        # Remove excessive punctuation runs (PDF table borders leaked as text)
        text = re.sub(r"[_\-=]{4,}", "", text)
        # Remove null bytes and control characters (except newline/tab)
        text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)
        # Collapse multiple newlines to paragraph breaks
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _detect_section(self, text: str) -> Optional[str]:
        """
        Identify common geological report section headers.
        Returns the section name if detected, otherwise None.
        """
        section_patterns = [
            (r"(?i)^(executive\s+summary)", "Executive Summary"),
            (r"(?i)(geological\s+setting|regional\s+geology)", "Geological Setting"),
            (r"(?i)(mineralization|mineral\s+zones?)", "Mineralization"),
            (r"(?i)(geochemist(?:ry|ical)\s+(?:data|survey|results))", "Geochemistry"),
            (r"(?i)(rock\s+types?|litholog(?:y|ies))", "Lithology"),
            (r"(?i)(structural\s+geology|tectonics?|fault)", "Structural Geology"),
            (r"(?i)(drilling\s+results?|borehole|drill\s+holes?)", "Drilling Results"),
            (r"(?i)(resource\s+estimate|mineral\s+resource)", "Resource Estimates"),
            (r"(?i)(hyperspectral|remote\s+sens)", "Remote Sensing"),
            (r"(?i)(recommendation|exploration\s+target)", "Recommendations"),
        ]
        for pattern, section_name in section_patterns:
            if re.search(pattern, text[:500]):
                return section_name
        return None

    def _compute_hash(self, file_path: Path) -> str:
        """Generate a short hash for document identity (deduplication)."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            hasher.update(f.read(65536))  # Read first 64KB for speed
        return hasher.hexdigest()[:12]
