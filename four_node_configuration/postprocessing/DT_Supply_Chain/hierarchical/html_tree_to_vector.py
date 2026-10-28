"""
Convert the styled schematic tree.html (the card + 4-node-network look used in the
strat/decentr/... folders) into a VECTOR PDF for Illustrator.

Headless Edge/Chrome renders the DOM + inline SVG and prints to PDF; every box, edge,
schematic pipe/node and text label stays a vector path / editable text (not a raster).
The print page is sized to the tree's exact pixel dimensions so nothing is clipped or
paginated.

Cases (edit CASES): the two chosen readable trees
  - strat/supply  all  k=3  -> Central / Import-reliant / Local
  - decentr/A     robust k=4 -> 4 decentralisation levels

Run (PowerShell):
  python html_tree_to_vector.py
Out: same folder as each tree.html -> tree_styled.pdf   (+ copied into vector/)
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

# geometry constants MUST match render_driver_tree_html.py JS (LEAFW, VGAP, TOP, leaf height)
LEAFW, VGAP, TOP, LEAF_BLOCK = 216, 136, 42, 285

BASE = C.BASE_FIG_DIR
CASES = [
    (os.path.join(BASE, "strat", "supply", "all", "k03", "tree.html"), "supply_all_k3"),
    (os.path.join(BASE, "decentr", "A", "robust", "k04", "tree.html"), "decentrA_robust_k4"),
]
VECTOR_DIR = os.path.join(BASE, "vector")


def _browser():
    for b in BROWSERS:
        if os.path.exists(b):
            return b
    raise RuntimeError("No Chrome/Edge found for headless PDF printing.")


def _tree_dims(html):
    """Parse the embedded DATA json and compute the rendered stage width/height (px),
    replicating the JS layout() so the PDF page fits exactly."""
    blob = html.split("const DATA=", 1)[1].split(", FEAT=", 1)[0].strip()
    data = json.loads(blob)
    tree = data["tree"]

    def leaves(n):
        return 1 if n["type"] == "leaf" else leaves(n["left"]) + leaves(n["right"])

    def depth(n):
        return 0 if n["type"] == "leaf" else 1 + max(depth(n["left"]), depth(n["right"]))

    w = leaves(tree) * LEAFW
    h = TOP + depth(tree) * VGAP + LEAF_BLOCK
    return w, h


def _prep_html(src_html, pad=30):
    """Inject an @page sized to the content + make the stage fully visible (no scroll)."""
    html = open(src_html, encoding="utf-8").read()
    w, h = _tree_dims(html)
    # box-shadow / filter / opacity make Chrome rasterize those regions to JPEG on
    # print -> strip them so every element stays vector (Illustrator-editable).
    inject = (f"<style>@page{{size:{w + 2 * pad}px {h + 2 * pad}px;margin:0}}"
              f"html,body{{margin:0}}body{{padding:{pad}px !important}}"
              f"#stage{{overflow:visible !important}}"
              f"*{{box-shadow:none !important;filter:none !important}}"
              f".leaf,.split{{border:1.4px solid #b7bcc9 !important}}</style>")
    html = html.replace("</head>", inject + "</head>")
    return html, (w, h)


def convert(src_html, tag):
    html, (w, h) = _prep_html(src_html)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8")
    tmp.write(html); tmp.close()
    out_pdf = os.path.join(os.path.dirname(src_html), "tree_styled.pdf")
    url = "file:///" + tmp.name.replace("\\", "/")
    cmd = [_browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
           "--print-to-pdf-no-header", "--virtual-time-budget=4000",
           f"--print-to-pdf={out_pdf}", url]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    os.unlink(tmp.name)
    if not os.path.exists(out_pdf):
        raise RuntimeError(f"PDF not produced for {tag}\n{r.stderr}")
    # PDF -> SVG (PyMuPDF), text kept as editable text, not outlined
    out_svg = os.path.join(os.path.dirname(src_html), "tree_styled.svg")
    _pdf_to_svg(out_pdf, out_svg)
    os.makedirs(VECTOR_DIR, exist_ok=True)
    for ext, src in (("pdf", out_pdf), ("svg", out_svg)):
        shutil.copyfile(src, os.path.join(VECTOR_DIR, f"{tag}_styled.{ext}"))
    kb = os.path.getsize(out_svg) // 1024
    print(f"  {tag}: {w}x{h}px -> tree_styled.pdf + tree_styled.svg ({kb} KB) "
          f"+ copies in vector/")


def _pdf_to_svg(pdf_path, svg_path):
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    svg = doc[0].get_svg_image(text_as_path=False)  # keep real text (editable in Illustrator)
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg)
    doc.close()


def main():
    for src, tag in CASES:
        if not os.path.exists(src):
            print(f"  MISSING: {src}"); continue
        convert(src, tag)
    print(f"\nDone -> styled vector PDFs in each case folder + {VECTOR_DIR}")


if __name__ == "__main__":
    main()