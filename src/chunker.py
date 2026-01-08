from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import logging
import re
from transformers import AutoTokenizer

logger = logging.getLogger("policy_chunker")


@dataclass
class Chunk:
    text: str
    chunk_id: str
    document_id: str
    metadata: Dict
    token_start: int
    token_end: int


class SemanticChunker:
    def __init__(
        self,
        tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_tokens: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_tokens = min_chunk_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.sentence_splitter = re.compile(r"(?<=[.!?。！？])\s+")

    def chunk_document(self, text: str, document_id: str, metadata: Dict) -> List[Chunk]:
        sentences = self._split_sentences(text)

        chunks: List[Chunk] = []
        current_sentences: List[str] = []
        current_tokens = 0
        token_cursor = 0

        for sentence in sentences:
            sent_tokens = self._count_tokens(sentence)

            if current_tokens + sent_tokens > self.chunk_size and current_sentences:
                chunk_text = " ".join(current_sentences)
                chunk_token_len = self._count_tokens(chunk_text)

                chunks.append(
                    Chunk(
                        text=chunk_text,
                        chunk_id=f"{document_id}_chunk_{len(chunks)}",
                        document_id=document_id,
                        metadata=metadata,
                        token_start=token_cursor,
                        token_end=token_cursor + chunk_token_len,
                    )
                )

                overlap_text = self._get_overlap(current_sentences)
                overlap_tokens = self._count_tokens(overlap_text) if overlap_text else 0

                current_sentences = [overlap_text] if overlap_text else []
                current_tokens = overlap_tokens
                token_cursor += chunk_token_len - overlap_tokens

            current_sentences.append(sentence)
            current_tokens += sent_tokens

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            token_len = self._count_tokens(chunk_text)
            if token_len >= self.min_chunk_tokens:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        chunk_id=f"{document_id}_chunk_{len(chunks)}",
                        document_id=document_id,
                        metadata=metadata,
                        token_start=token_cursor,
                        token_end=token_cursor + token_len,
                    )
                )

        logger.info(f"Chunked {document_id}: {len(chunks)} chunks")
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        sentences = self.sentence_splitter.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _get_overlap(self, sentences: List[str]) -> str:
        overlap_tokens = 0
        overlap_sentences = []
        for sent in reversed(sentences):
            sent_tokens = self._count_tokens(sent)
            if overlap_tokens + sent_tokens > self.chunk_overlap:
                break
            overlap_sentences.insert(0, sent)
            overlap_tokens += sent_tokens
        return " ".join(overlap_sentences)


class HybridChunker(SemanticChunker):
    def __init__(self, skip_toc: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.skip_toc = skip_toc

        self.header_patterns = [
            re.compile(r"^\d+\.\s+[A-Z]"),      # "1. POLICY"
            re.compile(r"^[A-Z][A-Z\s]{6,}$"),  # "LEAVE POLICY"
            re.compile(r"^##\s+"),              # Markdown headers
        ]

    def chunk_document(self, text: str, document_id: str, metadata: Dict) -> List[Chunk]:
        sections = self._split_sections(text)
        all_chunks: List[Chunk] = []

        for idx, section in enumerate(sections):
            header = (section.get("header") or "").strip()
            header_l = header.lower()

            if self.skip_toc and (("table of contents" in header_l) or (header_l in {"contents", "index"})):
                logger.info(f"Skipping TOC section: '{header}'")
                continue

            section_metadata = {**metadata, "section_id": idx, "section_header": header}

            section_chunks = super().chunk_document(
                text=section["content"],
                document_id=f"{document_id}_sec{idx}",
                metadata=section_metadata,
            )
            all_chunks.extend(section_chunks)

        return all_chunks

    def _split_sections(self, text: str) -> List[Dict]:
        lines = text.splitlines()
        sections = []
        current = {"header": "", "content": ""}

        for line in lines:
            ls = line.strip()
            if any(p.match(ls) for p in self.header_patterns):
                if current["content"].strip():
                    sections.append(current)
                current = {"header": ls, "content": ""}
            else:
                current["content"] += line + "\n"

        if current["content"].strip():
            sections.append(current)

        return sections if sections else [{"header": "", "content": text}]
