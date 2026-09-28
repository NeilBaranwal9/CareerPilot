"""
PDF/DOCX resume support:
- extract_resume_text : text from .pdf (pypdf), .docx (document.xml), .typ (markup stripped), .md/.txt
- render_resume_pdf   : renders a StructuredResumeSchema to a one-page PDF with fpdf2 (no Typst needed)
- check_tailored_resume: fabrication guard (employers, dates, name and skills must exist in the original)
"""

import logging
import os
import re
import zipfile
from pathlib import Path

from src.pipeline.schemas import StructuredResumeSchema

logger = logging.getLogger("recruiting-platform.utils.resume_pdf")

SUPPORTED_RESUME_EXTENSIONS = (".pdf", ".typ", ".docx", ".md", ".txt")

_FONT_CANDIDATES = [
    # (regular, bold, italic)
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/ariali.ttf"),
    ("C:/Windows/Fonts/calibri.ttf", "C:/Windows/Fonts/calibrib.ttf", "C:/Windows/Fonts/calibrii.ttf"),
    (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
    ),
    ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf", "/Library/Fonts/Arial Italic.ttf"),
    (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
    ),
]

_LATIN1_REPLACEMENTS = {
    "\u2022": "-", "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u20b9": "Rs.", "\u2192": "->", "\u2026": "...", "\u00a0": " ",
}


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def clean_extracted_text(text: str) -> str:
    """Repairs common PDF extraction artifacts ('V ellore' -> 'Vellore', stray replacement chars, spacing)."""
    text = text.replace("\r", "")
    text = re.sub(r"\b([A-Z]) ([a-z]{2,})", r"\1\2", text)  # kerning splits: "T echnology"
    text = text.replace("\ufffd", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def _extract_pdf(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_docx(path: str) -> str:
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8", errors="ignore")
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<w:tab/>", "\t", xml)
    return re.sub(r"<[^>]+>", "", xml)


def extract_resume_text(path: str | None) -> str:
    """Plain text of a resume in any supported format ('' if missing or unreadable)."""
    if not path or not os.path.exists(path):
        return ""
    ext = Path(path).suffix.lower()
    try:
        if ext == ".pdf":
            return clean_extracted_text(_extract_pdf(path))
        if ext == ".docx":
            return clean_extracted_text(_extract_docx(path))
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        if ext == ".typ":
            from src.utils.resume import typst_to_text

            return typst_to_text(raw, limit=20000)
        return clean_extracted_text(raw)
    except Exception as e:
        logger.warning(f"Could not extract text from resume {path}: {e}")
        return ""


def validate_resume_file(path: str) -> list[str]:
    """Problems that would make the file unusable as an attachment (empty list = OK)."""
    problems: list[str] = []
    if not os.path.exists(path):
        return [f"{path} does not exist"]
    ext = Path(path).suffix.lower()
    if ext not in SUPPORTED_RESUME_EXTENSIONS:
        problems.append(f"unsupported resume format {ext}")
    if os.path.getsize(path) == 0:
        problems.append("file is empty")
    if ext == ".pdf":
        with open(path, "rb") as f:
            if f.read(5) != b"%PDF-":
                problems.append("file does not look like a PDF")
    return problems


def structured_to_text(resume: StructuredResumeSchema) -> str:
    lines = [resume.name, resume.contact_line]
    for section in resume.sections:
        lines.append("")
        lines.append(section.heading)
        for entry in section.entries:
            head = " | ".join(x for x in (entry.title, entry.organization, entry.location, entry.dates) if x)
            lines.append(head)
            if entry.subtitle:
                lines.append(entry.subtitle)
            lines.extend(f"- {b}" for b in entry.bullets)
        lines.extend(f"- {item}" for item in section.items)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fabrication guard
# ---------------------------------------------------------------------------


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def check_tailored_resume(original_text: str, resume: StructuredResumeSchema) -> list[str]:
    """
    Returns violations if the tailored resume introduces facts that are not in the original:
    a different name, employers/institutions, dates/years or skills that never appear in the source.
    """
    errors: list[str] = []
    source = _norm(original_text)
    source_years = set(re.findall(r"(?:19|20)\d{2}", original_text))
    if _norm(resume.name) not in source:
        errors.append(f"name '{resume.name}' not in original resume")
    for section in resume.sections:
        for entry in section.entries:
            if entry.organization and _norm(entry.organization) not in source:
                errors.append(f"organization '{entry.organization}' not in original resume")
            for year in re.findall(r"(?:19|20)\d{2}", entry.dates or ""):
                if year not in source_years:
                    errors.append(f"date '{entry.dates}' not in original resume")
                    break
        if re.search(r"skill|technolog|tools", section.heading, re.IGNORECASE):
            for item in section.items:
                label, _, values = item.partition(":")
                for skill in re.split(r"[,;/|]", values or label):
                    if skill.strip() and _norm(skill) and _norm(skill) not in source:
                        errors.append(f"skill '{skill.strip()}' not in original resume")
    return errors


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _find_font() -> tuple[str, str, str] | None:
    for regular, bold, italic in _FONT_CANDIDATES:
        if os.path.exists(regular) and os.path.exists(bold):
            return regular, bold, italic if os.path.exists(italic) else regular
    return None


def _normalize_glyphs(text: str) -> str:
    """Maps characters many fonts lack (non-breaking/figure hyphens, narrow spaces) to safe equivalents."""
    for ch in ("‐", "‑", "‒", "−"):
        text = text.replace(ch, "-")
    for ch in (" ", " ", " ", "​"):
        text = text.replace(ch, " " if ch != "​" else "")
    return text


def _latin1(text: str) -> str:
    text = _normalize_glyphs(text)
    for src, dst in _LATIN1_REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _render(resume: StructuredResumeSchema, scale: float) -> tuple[bytes, int]:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=11)
    pdf.set_margins(13, 11, 13)
    pdf.add_page()

    font = _find_font()
    if font:
        pdf.add_font("Body", "", font[0])
        pdf.add_font("Body", "B", font[1])
        pdf.add_font("Body", "I", font[2])
        family, bullet = "Body", "\u2022"

        def t(text: str) -> str:
            return _normalize_glyphs(text)
    else:
        family, bullet = "Helvetica", "-"
        t = _latin1

    width = pdf.w - pdf.l_margin - pdf.r_margin
    line = 4.3 * scale

    pdf.set_font(family, "B", 17 * scale)
    pdf.cell(width, 8 * scale, t(resume.name), align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(family, "", 8.8 * scale)
    pdf.multi_cell(width, line, t(resume.contact_line), align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    for section in resume.sections:
        pdf.ln(1.6 * scale)
        pdf.set_font(family, "B", 11 * scale)
        pdf.cell(width, 5.6 * scale, t(section.heading.upper()), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        y = pdf.get_y()
        pdf.set_draw_color(120, 120, 120)
        pdf.line(pdf.l_margin, y, pdf.l_margin + width, y)
        pdf.ln(1.0 * scale)

        for entry in section.entries:
            dates = t(entry.dates or "")
            pdf.set_font(family, "", 9 * scale)
            dates_w = pdf.get_string_width(dates) + 2 if dates else 0
            pdf.set_font(family, "B", 9.6 * scale)
            pdf.cell(width - dates_w, line + 0.3, t(entry.title))
            pdf.set_font(family, "", 9 * scale)
            pdf.cell(dates_w, line + 0.3, dates, align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            meta = " | ".join(x for x in (entry.organization, entry.location) if x)
            if meta:
                pdf.set_font(family, "I", 9 * scale)
                pdf.multi_cell(width, line, t(meta), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            if entry.subtitle:
                pdf.set_font(family, "I", 8.8 * scale)
                pdf.multi_cell(width, line, t(entry.subtitle), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font(family, "", 9 * scale)
            for b in entry.bullets:
                pdf.set_x(pdf.l_margin + 2)
                pdf.cell(3.5, line, bullet)
                pdf.multi_cell(width - 5.5, line, t(b), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(0.8 * scale)

        pdf.set_font(family, "", 9 * scale)
        for item in section.items:
            label, sep, rest = item.partition(":")
            pdf.set_x(pdf.l_margin + 2)
            if sep and len(label) < 40:
                pdf.set_font(family, "B", 9 * scale)
                label_text = t(label.strip() + ": ")
                pdf.cell(pdf.get_string_width(label_text), line, label_text)
                pdf.set_font(family, "", 9 * scale)
                pdf.multi_cell(0, line, t(rest.strip()), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            else:
                pdf.cell(3.5, line, bullet)
                pdf.multi_cell(width - 5.5, line, t(item), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    return bytes(pdf.output()), pdf.pages_count


def render_resume_pdf(resume: StructuredResumeSchema, output_path: str, max_pages: int = 1) -> int:
    """Renders the resume, shrinking type slightly until it fits `max_pages`. Returns the page count."""
    data, pages = b"", 0
    for scale in (1.0, 0.95, 0.9, 0.85, 0.8):
        data, pages = _render(resume, scale)
        if pages <= max_pages:
            break
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(data)
    return pages
