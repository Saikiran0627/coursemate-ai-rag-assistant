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
