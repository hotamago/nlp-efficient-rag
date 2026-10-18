"""Pack the report and slides as Overleaf-uploadable zips (compiler: XeLaTeX) into data/overleaf/.

Only git-tracked files are included, so the gitignored course templates and PDFs never leave the
machine.
"""

import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    out_dir = ROOT / "data/overleaf"
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in ("report", "slide"):
        src = ROOT / "docs" / d
        files = subprocess.run(
            ["git", "ls-files"], cwd=src, capture_output=True, text=True, check=True
        ).stdout.split()
        files = [f for f in files if not f.endswith(".pptx")]  # PowerPoint template, not LaTeX
        out = out_dir / f"{d}.zip"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for f in files:
                z.write(src / f, f)
        print(f"{out.relative_to(ROOT)}: {len(files)} files")
