"""Render PDF pages to PNG for visual checks: uv run python scripts/render_pdf.py file.pdf 1 2 3"""

import sys
from pathlib import Path

import pymupdf as fitz

pdf = Path(sys.argv[1])
has_out = len(sys.argv) > 2 and not sys.argv[2].isdigit()
out = Path(sys.argv[2]) if has_out else pdf.parent / "_render"
pages = [int(p) for p in sys.argv[2:] if p.isdigit()] or [1]
out.mkdir(parents=True, exist_ok=True)
doc = fitz.open(pdf)
print(f"{pdf.name}: {doc.page_count} pages")
for p in pages:
    path = out / f"{pdf.stem}_p{p}.png"
    doc[p - 1].get_pixmap(dpi=70).save(path)
    print(path)
