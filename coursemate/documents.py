"""Document loading utilities for CourseMate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = {".md", ".txt"}


@dataclass(frozen=True)
class Document:
    """A source document from the course-material corpus."""

    doc_id: str
    title: str
    source_path: str
    text: str


def load_documents(data_dir: str | Path) -> list[Document]:
    """Load markdown/text documents from a directory tree."""

    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Course material directory does not exist: {root}")

    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        title = _extract_title(text, fallback=path.stem.replace("_", " ").title())
        doc_id = path.relative_to(root).with_suffix("").as_posix()
        documents.append(
            Document(
                doc_id=doc_id,
                title=title,
                source_path=path.as_posix(),
                text=text,
            )
        )

    if not documents:
        raise ValueError(f"No supported course material files found under {root}")
    return documents


def _extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        cleaned = line.strip()
        if cleaned.startswith("#"):
            return cleaned.lstrip("#").strip() or fallback
    return fallback
