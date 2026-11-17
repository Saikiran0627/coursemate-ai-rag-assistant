from coursemate.ingestion import DocumentIngestionError, parse_uploaded_file


def test_parse_text_upload_creates_document_metadata():
    uploaded = parse_uploaded_file(
        "lecture_notes.txt",
        b"Retrieval augmented generation uses retrieved context for answers.",
    )

    assert uploaded.filename == "lecture_notes.txt"
    assert uploaded.document.title == "Lecture Notes"
    assert "retrieved context" in uploaded.document.text
    assert uploaded.content_hash


def test_parse_unsupported_upload_type_raises_helpful_error():
    try:
        parse_uploaded_file("grades.xlsx", b"not supported")
    except DocumentIngestionError as exc:
        assert "Unsupported file type" in str(exc)
        assert ".pdf" in str(exc)
    else:
        raise AssertionError("Expected unsupported file type to raise")


def test_parse_docx_upload():
    from docx import Document
    from io import BytesIO

    buffer = BytesIO()
    document = Document()
    document.add_paragraph("CourseMate reads Word lecture notes.")
    document.save(buffer)

    uploaded = parse_uploaded_file("lecture.docx", buffer.getvalue())

    assert "Word lecture notes" in uploaded.document.text


def test_parse_pptx_upload():
    from io import BytesIO
    from pptx import Presentation

    buffer = BytesIO()
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    slide.shapes.title.text = "RAG Slide"
    presentation.save(buffer)

    uploaded = parse_uploaded_file("slides.pptx", buffer.getvalue())

    assert "RAG Slide" in uploaded.document.text


def test_parse_pdf_upload_with_text():
    uploaded = parse_uploaded_file("lecture.pdf", _make_pdf("PDF notes discuss ChromaDB retrieval."))

    assert "ChromaDB retrieval" in uploaded.document.text


def _make_pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream".encode("latin-1"),
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_number, payload in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{object_number} 0 obj\n".encode("ascii"))
        output.extend(payload)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode(
            "ascii"
        )
    )
    return bytes(output)
