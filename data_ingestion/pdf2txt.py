#!/usr/bin/env python3
import sys
import shutil
import tempfile
import os
from pathlib import Path
import argparse

# ——————————————————————————————————————————————
# 1) VERIFY PYTHON DEPENDENCIES
# ——————————————————————————————————————————————
REQ_LIBS = [
    ("PyPDF2",      "PyPDF2"),
    ("pdf2image",   "pdf2image"),
    ("PIL",         "Pillow (PIL)"),
    ("pytesseract", "pytesseract"),
]

missing = []
for mod, name in REQ_LIBS:
    try:
        pkg = __import__(mod)
        version = getattr(pkg, "__version__", "unknown")
        print(f"✔️  {name} version {version} found")
    except ImportError:
        missing.append(name)

if missing:
    print("\n❌ Missing required Python packages:")
    for name in missing:
        print(f"   • {name}")
    print("\nInstall them via:")
    print("   python3 -m pip install --upgrade PyPDF2 pdf2image pillow pytesseract")
    sys.exit(1)

# ——————————————————————————————————————————————
# 2) VERIFY TESSERACT BINARY
# ——————————————————————————————————————————————
tess_path = shutil.which("tesseract")
if tess_path:
    try:
        import pytesseract
        tver = pytesseract.get_tesseract_version()
        print(f"✔️  Tesseract-OCR engine found: {tver}")
    except Exception:
        print("\n❌ Found Tesseract binary but could not query its version.")
        print("   Ensure you can run `tesseract --version` in your shell.")
        sys.exit(1)
else:
    print("\n❌ Tesseract executable not found in your PATH.")
    print("   On Windows, install it and add to your PATH so `tesseract --version` works.")
    sys.exit(1)

print("\nAll OCR dependencies OK — proceeding with extraction.\n")

# ——————————————————————————————————————————————
# 3) STANDARD IMPORTS
# ——————————————————————————————————————————————
from PyPDF2 import PdfReader, PdfWriter
from pdf2image import convert_from_path
import pytesseract

# ——————————————————————————————————————————————
# 4) ARGUMENTS
# ——————————————————————————————————————————————
parser = argparse.ArgumentParser(
    description="Extract all text from PDFs (with OCR fallback & batch-repair) into one TXT file."
)
parser.add_argument(
    "--input_folder", "-i",
    default="pdfs",
    help="Folder containing PDFs (default: ./pdfs)"
)
parser.add_argument(
    "--output_file", "-o",
    default="pdfs/extracted_texts.txt",
    help="Output TXT file (default: ./pdfs/extracted_texts.txt)"
)
parser.add_argument(
    "--poppler_path", "-p",
    "--poppler-path",
    dest="poppler_path",
    help="Path to Poppler 'bin' directory (where pdftoppm.exe lives)."
)
args = parser.parse_args()

input_folder = Path(args.input_folder)
output_file  = Path(args.output_file)

# ——————————————————————————————————————————————
# 5) POPPLER PATH DETECTION
# ——————————————————————————————————————————————
if args.poppler_path:
    poppler_path = args.poppler_path
else:
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm:
        poppler_path = os.path.dirname(pdftoppm)
    else:
        print("❌ Poppler ‘pdftoppm’ not found on PATH. Either install Poppler or use -p.")
        sys.exit(1)

# ——————————————————————————————————————————————
# 6) LOAD LIST OF ALREADY-PROCESSED PDFs
# ——————————————————————————————————————————————
processed = set()
if output_file.exists():
    for line in output_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("===== ") and line.rstrip().endswith(" ====="):
            # line is "===== filename.pdf ====="
            name = line.strip("= ").strip()
            processed.add(name)

# ——————————————————————————————————————————————
# 7) PREPARE OUTPUT FILE
# ——————————————————————————————————————————————
if not input_folder.is_dir():
    print(f"❌ Input folder not found: {input_folder}")
    sys.exit(1)

output_file.parent.mkdir(parents=True, exist_ok=True)
out_f = open(output_file, "a", encoding="utf-8")  # append mode

# ——————————————————————————————————————————————
# 8) PDF REPAIR FUNCTION
# ——————————————————————————————————————————————
def repair_pdf(src_path: Path) -> Path:
    """Attempt to rebuild PDF via PyPDF2 round-trip; returns path to repaired temp PDF."""
    tmp_fd, tmp_file = tempfile.mkstemp(suffix=".pdf")
    os.close(tmp_fd)
    writer = PdfWriter()
    reader = PdfReader(str(src_path))
    for page in reader.pages:
        writer.add_page(page)
    with open(tmp_file, "wb") as f_out:
        writer.write(f_out)
    return Path(tmp_file)

# ——————————————————————————————————————————————
# 9) PROCESS EACH PDF
# ——————————————————————————————————————————————
for pdf_path in sorted(input_folder.glob("*.pdf")):
    if pdf_path.name in processed:
        print(f"⏭️  Skipping already-processed {pdf_path.name}")
        continue

    print(f"🔄 Processing {pdf_path.name}...")
    out_f.write(f"\n===== {pdf_path.name} =====\n\n")

    # attempt to read; if fail, try repair
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        print(f"⚠️  Read error on {pdf_path.name}: {e}")
        print("    Attempting batch repair…")
        try:
            repaired = repair_pdf(pdf_path)
            reader = PdfReader(str(repaired))
            print("    ✔️  Repair successful; continuing on repaired PDF")
        except Exception as err2:
            print(f"    ❌ Repair failed: {err2}; skipping this file.")
            continue

    # extract pages
    num_pages = len(reader.pages)
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            out_f.write(text + "\n\n")
        else:
            # OCR fallback for blank page
            try:
                images = convert_from_path(
                    str(pdf_path),
                    poppler_path=poppler_path,
                    first_page=i,
                    last_page=i
                )
                ocr_text = pytesseract.image_to_string(images[0])
                out_f.write(ocr_text + "\n\n")
            except Exception as oe:
                print(f"  ✖ Page {i}: OCR failed ({oe})")

out_f.close()
print(f"\n✅ Extraction complete. All text written to '{output_file.resolve()}'")
