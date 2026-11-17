"""Upload ingestion for CourseMate web documents."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from coursemate.documents import Document


SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".pptx", ".docx", ".txt", ".md"}


class DocumentIngestionError(ValueError):
    """Raised when an uploaded document cannot be parsed."""


@dataclass(frozen=True)
class UploadedDocument:
    """Parsed upload with a stable content hash."""

    filename: str
    content_hash: str
    document: Document


def parse_uploaded_file(filename: str, content: bytes) -> UploadedDocument:
    """Parse an uploaded PDF, PPTX, DOCX, TXT, or Markdown file."""

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_UPLOAD_EXTENSIONS))
        raise DocumentIngestionError(f"Unsupported file type '{suffix}'. Supported types: {supported}.")

    if suffix == ".pdf":
        text = _read_pdf(content)
    elif suffix == ".pptx":
        text = _read_pptx(content)
    elif suffix == ".docx":
        text = _read_docx(content)
    else:
        text = _read_text(content)

    cleaned = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not cleaned:
        raise DocumentIngestionError(f"No readable text found in {filename}.")

    digest = hashlib.sha256(content).hexdigest()[:16]
    title = Path(filename).stem.replace("_", " ").replace("-", " ").strip().title()
    doc_id = f"{Path(filename).stem}-{digest}"
    return UploadedDocument(
        filename=filename,
        content_hash=digest,
        document=Document(
            doc_id=doc_id,
            title=title,
            source_path=filename,
            text=cleaned,
        ),
    )


def _read_text(content: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentIngestionError("Could not decode text file with UTF-8, UTF-16, or Latin-1.")


def _read_pdf(content: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise DocumentIngestionError("Install pypdf to parse PDF files.") from exc

    try:
        reader = PdfReader(BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # pypdf raises several parser-specific exceptions
        raise DocumentIngestionError(f"Could not parse PDF: {exc}") from exc
    return "\n\n".join(pages)


def _read_docx(content: bytes) -> str:
    try:
        from docx import Document as DocxDocument
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise DocumentIngestionError("Install python-docx to parse DOCX files.") from exc

    try:
        document = DocxDocument(BytesIO(content))
        paragraphs = [paragraph.text for paragraph in document.paragraphs]
    except Exception as exc:
        raise DocumentIngestionError(f"Could not parse DOCX: {exc}") from exc
    return "\n".join(paragraphs)


def _read_pptx(content: bytes) -> str:
    try:
        from pptx import Presentation
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise DocumentIngestionError("Install python-pptx to parse PPTX files.") from exc

    try:
        presentation = Presentation(BytesIO(content))
        slide_text: list[str] = []
        for slide_number, slide in enumerate(presentation.slides, start=1):
            lines = [f"Slide {slide_number}"]
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    lines.append(shape.text)
            slide_text.append("\n".join(lines))
    except Exception as exc:
        raise DocumentIngestionError(f"Could not parse PPTX: {exc}") from exc
    return "\n\n".join(slide_text)
