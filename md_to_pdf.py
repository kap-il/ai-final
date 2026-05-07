"""Render a Markdown source file to PDF using fpdf2.

Handles: headings (#, ##, ###), code fences (```), pipe tables,
bullet lists, bold (**x**), italic (*x*), inline code (`x`),
horizontal rules (---), and plain paragraphs. Good enough for the
report drafts; not a full CommonMark parser.

Usage:
    python3 md_to_pdf.py input.md output.pdf
"""

import re
import sys
from fpdf import FPDF


PAGE_W = 210  # A4 mm
MARGIN = 18
USABLE = PAGE_W - 2 * MARGIN

UNICODE_FALLBACK = {
    "—": "--",   # em dash
    "–": "-",    # en dash
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
    "…": "...",
    "±": "+/-",
    "→": "->",
    "←": "<-",
    "≤": "<=",
    "≥": ">=",
    "≠": "!=",
    "×": "x",
    "·": ".",
    "•": "-",
    " ": " ",
}


def sanitize(text):
    for k, v in UNICODE_FALLBACK.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "replace").decode("latin-1")


class ReportPDF(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")
        self.set_text_color(0)


def write_inline(pdf, text, base_size=10.5):
    """Emit a line of text with **bold**, *italic*, and `code` markers."""
    parts = re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", text)
    for p in parts:
        if not p:
            continue
        if p.startswith("**") and p.endswith("**"):
            pdf.set_font("Helvetica", "B", base_size)
            pdf.write(5.5, p[2:-2])
        elif p.startswith("*") and p.endswith("*") and len(p) > 2:
            pdf.set_font("Helvetica", "I", base_size)
            pdf.write(5.5, p[1:-1])
        elif p.startswith("`") and p.endswith("`"):
            pdf.set_font("Courier", "", base_size - 0.5)
            pdf.write(5.5, p[1:-1])
        else:
            pdf.set_font("Helvetica", "", base_size)
            pdf.write(5.5, p)
    pdf.ln(6)


def render_table(pdf, rows):
    """Render a pipe table. `rows` is a list of lists of cell strings."""
    if not rows:
        return
    n_cols = max(len(r) for r in rows)
    rows = [r + [""] * (n_cols - len(r)) for r in rows]
    col_w = USABLE / n_cols
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(230, 230, 230)
    for cell in rows[0]:
        pdf.cell(col_w, 7, cell.strip(), border=1, fill=True)
    pdf.ln(7)
    pdf.set_font("Helvetica", "", 9)
    for row in rows[1:]:
        if all(re.fullmatch(r"\s*:?-+:?\s*", c) for c in row):
            continue
        for cell in row:
            pdf.cell(col_w, 6, cell.strip(), border=1)
        pdf.ln(6)
    pdf.ln(2)


def render_md(md_text, out_path):
    pdf = ReportPDF(format="A4")
    pdf.set_margins(MARGIN, MARGIN, MARGIN)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            j = i + 1
            code_lines = []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                code_lines.append(lines[j])
                j += 1
            pdf.set_fill_color(245, 245, 245)
            pdf.set_font("Courier", "", 8.5)
            for cl in code_lines:
                pdf.cell(0, 4.5, cl.rstrip(), fill=True, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
            i = j + 1
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"\|\s*:?-+", lines[i + 1].strip()):
            tbl = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c for c in lines[i].strip().strip("|").split("|")]
                tbl.append(cells)
                i += 1
            render_table(pdf, tbl)
            continue

        if re.match(r"^-{3,}\s*$", stripped):
            pdf.ln(1)
            y = pdf.get_y()
            pdf.set_draw_color(180, 180, 180)
            pdf.line(MARGIN, y, PAGE_W - MARGIN, y)
            pdf.set_draw_color(0)
            pdf.ln(3)
            i += 1
            continue

        if stripped.startswith("### "):
            pdf.ln(1)
            pdf.set_font("Helvetica", "B", 11.5)
            pdf.multi_cell(0, 6, stripped[4:])
            pdf.ln(0.5)
            i += 1
            continue
        if stripped.startswith("## "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(0, 7, stripped[3:])
            pdf.ln(1)
            i += 1
            continue
        if stripped.startswith("# "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 16)
            pdf.multi_cell(0, 9, stripped[2:])
            pdf.ln(1.5)
            i += 1
            continue

        m = re.match(r"^(\s*)([-*])\s+(.*)$", line)
        if m:
            indent = len(m.group(1)) // 2
            text = m.group(3)
            pdf.set_x(MARGIN + 4 + indent * 5)
            pdf.set_font("Helvetica", "", 10.5)
            pdf.write(5.5, "- ")
            write_inline(pdf, text)
            i += 1
            continue

        m = re.match(r"^(\s*)\[ \]\s+(.*)$", line)
        if m:
            pdf.set_x(MARGIN + 4)
            pdf.set_font("Helvetica", "", 10.5)
            pdf.write(5.5, "[ ] ")
            write_inline(pdf, m.group(2))
            i += 1
            continue

        if stripped.startswith("> "):
            pdf.set_font("Helvetica", "I", 10.5)
            pdf.set_text_color(80)
            pdf.set_x(MARGIN + 4)
            pdf.multi_cell(USABLE - 4, 5.5, stripped[2:])
            pdf.set_text_color(0)
            pdf.ln(1)
            i += 1
            continue

        if stripped == "":
            pdf.ln(2.5)
            i += 1
            continue

        write_inline(pdf, line)
        i += 1

    pdf.output(out_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python3 md_to_pdf.py input.md output.pdf", file=sys.stderr)
        sys.exit(1)
    with open(sys.argv[1], "r") as f:
        render_md(sanitize(f.read()), sys.argv[2])
    print(f"wrote {sys.argv[2]}")
