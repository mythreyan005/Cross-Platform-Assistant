"""
Document Preprocessing and Chunking Pipeline.
Cleans software engineering text while preserving technical identifiers,
splits into overlapping chunks, and attaches comprehensive metadata.
"""

import re
import logging
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from .models_data import SourceRecord
except ImportError:
    from models_data import SourceRecord

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Handles text cleaning, token/character chunking, and deterministic ID generation for vector indexing.
    """

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", ", ", " "],
            keep_separator=True
        )

    @staticmethod
    def clean_text(raw_text: str) -> str:
        """
        Normalizes whitespace while carefully preserving technical symbols:
        commit hashes (7-40 chars), Jira keys (PROJ-1234), URLs, and HTTP endpoints.
        """
        if not raw_text:
            return ""

        # Normalize line endings
        text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse excess empty lines (max 2 consecutive newlines)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Collapse excessive spaces within lines without touching newlines
        lines = [re.sub(r'[ \t]{2,}', ' ', line).strip() for line in text.split('\n')]
        return "\n".join(lines).strip()

    def process_record(self, record: SourceRecord) -> List[Tuple[str, Document]]:
        """
        Processes a single SourceRecord into chunked LangChain Document objects with deterministic IDs.
        Returns a list of (deterministic_id, Document).
        """
        cleaned_content = self.clean_text(record.content)
        if not cleaned_content:
            logger.debug(f"Skipping empty record: {record.source_type} {record.source_id}")
            return []

        chunks = self.splitter.split_text(cleaned_content)
        if not chunks:
            chunks = [cleaned_content]

        processed: List[Tuple[str, Document]] = []
        safe_proj = re.sub(r'[^a-zA-Z0-9_-]', '_', record.project_id)
        safe_source_id = re.sub(r'[^a-zA-Z0-9_-]', '_', record.source_id)

        for idx, chunk_text in enumerate(chunks):
            # Deterministic chunk ID prevents duplicates during incremental re-indexing
            chunk_id = f"{safe_proj}:{record.source_type.value}:{safe_source_id}:c{idx}"

            metadata: Dict[str, Any] = {
                "chunk_id": chunk_id,
                "source_type": record.source_type.value if hasattr(record.source_type, 'value') else str(record.source_type),
                "source_id": record.source_id,
                "project_id": record.project_id,
                "title": record.title,
                "url": record.url or "",
                "author": record.author or "",
                "status": record.status or "",
                "timestamp": record.timestamp or "",
                "chunk_index": idx,
                "total_chunks": len(chunks)
            }

            # Copy relevant extra metadata without nesting
            for k in ("jira_keys", "components", "method", "path"):
                if k in record.metadata:
                    val = record.metadata[k]
                    if isinstance(val, list):
                        metadata[k] = ", ".join(str(v) for v in val)
                    else:
                        metadata[k] = str(val)

            doc = Document(page_content=chunk_text, metadata=metadata)
            processed.append((chunk_id, doc))

        return processed

    def process_records(self, records: List[SourceRecord]) -> List[Tuple[str, Document]]:
        """Batches processing over multiple records."""
        all_chunks: List[Tuple[str, Document]] = []
        for r in records:
            all_chunks.extend(self.process_record(r))
        return all_chunks
