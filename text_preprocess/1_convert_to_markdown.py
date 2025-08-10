#!/usr/bin/env python3
"""
Convert every PDF in the current directory to Markdown.

Install docling first:
    pip install docling
"""

from pathlib import Path
from docling.document_converter import DocumentConverter

PDF_DIR    = Path.cwd() / "pdf"          # folder to scan
OUTPUT_DIR = Path.cwd() / "md"    # folder to store .md files
OUTPUT_DIR.mkdir(exist_ok=True)

converter = DocumentConverter()

for pdf in PDF_DIR.glob("*.pdf"):
    md_path = OUTPUT_DIR / f"{pdf.stem}.md"
    print(f"Processing {pdf.name} → {md_path.name}")
    try:
        result = converter.convert(str(pdf))          # SAME as sample
        md_path.write_text(
            result.document.export_to_markdown(),     # SAME as sample
            encoding="utf-8"
        )
    except Exception as e:
        print(f"  ❌  Error converting {pdf.name}: {e}")

print("Done!")
