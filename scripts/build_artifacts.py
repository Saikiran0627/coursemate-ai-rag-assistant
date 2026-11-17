"""Build plain PDF submission artifacts from Markdown sources.

The generator intentionally uses only the Python standard library so the final
project can create PDFs in locked-down classroom environments.
"""

from __future__ import annotations

import textwrap
from pathlib import Path


PAGE_WIDTH = 72
LINES_PER_PAGE = 48


def main() -> None:
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    markdown_to_pdf(Path("docs/final_paper.md"), artifacts / "CourseMate_Final_Project_Paper.pdf")
    markdown_to_pdf(Path("docs/presentation_slides.md"), artifacts / "CourseMate_Presentation_Slides.pdf")
    print("Built PDF artifacts in artifacts/")


def markdown_to_pdf(source: Path, destination: Path) -> None:
    text = source.read_text(encoding="utf-8")
    pages = paginate(text)
    write_pdf(pages, destination)


def paginate(text: str) -> list[list[str]]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            lines.append("")
            continue
        prefix = ""
        stripped = raw_line.strip()
        if stripped.startswith("#"):
            prefix = ""
            stripped = stripped.lstrip("#").strip().upper()
        elif stripped.startswith("- "):
            prefix = "- "
            stripped = stripped[2:].strip()
        wrapped = textwrap.wrap(stripped, width=PAGE_WIDTH - len(prefix)) or [""]
        lines.extend(prefix + line if line else "" for line in wrapped)

    pages: list[list[str]] = []
    for index in range(0, len(lines), LINES_PER_PAGE):
        pages.append(lines[index : index + LINES_PER_PAGE])
    return pages or [[]]


def write_pdf(pages: list[list[str]], destination: Path) -> None:
    objects: list[bytes] = []
    catalog_id = 1
    pages_id = 2
    font_id = 3
    page_ids: list[int] = []

    for page_lines in pages:
        content = build_page_content(page_lines)
        content_id = len(objects) + 4
        page_id = content_id + 1
        objects.append(
            f"<< /Length {len(content.encode('latin-1', errors='replace'))} >>\nstream\n{content}\nendstream".encode(
                "latin-1", errors="replace"
            )
        )
        objects.append(
            (
                f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
                f"/Contents {content_id} 0 R >>"
            ).encode("ascii")
        )
        page_ids.append(page_id)

    all_objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{' '.join(f'{page_id} 0 R' for page_id in page_ids)}] /Count {len(page_ids)} >>".encode(
            "ascii"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        *objects,
    ]

    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_number, payload in enumerate(all_objects, start=1):
        offsets.append(len(output))
        output.extend(f"{object_number} 0 obj\n".encode("ascii"))
        output.extend(payload)
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(all_objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            "trailer\n"
            f"<< /Size {len(all_objects) + 1} /Root {catalog_id} 0 R >>\n"
            "startxref\n"
            f"{xref_offset}\n"
            "%%EOF\n"
        ).encode("ascii")
    )
    destination.write_bytes(output)


def build_page_content(lines: list[str]) -> str:
    commands = ["BT", "/F1 10 Tf", "50 750 Td", "14 TL"]
    for line in lines:
        commands.append(f"({escape_pdf(line)}) Tj")
        commands.append("T*")
    commands.append("ET")
    return "\n".join(commands)


def escape_pdf(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


if __name__ == "__main__":
    main()
