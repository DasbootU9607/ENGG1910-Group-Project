from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer


PROJECT_DIR = Path(__file__).resolve().parent
REPORT_MD = PROJECT_DIR / "report_draft.md"
REPORT_PDF = PROJECT_DIR / "report_draft.pdf"
CHART_PATH = PROJECT_DIR / "outputs" / "risk_distribution.png"


def clean_inline(text: str) -> str:
    return text.replace("`", "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_pdf() -> None:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            alignment=1,
            textColor=colors.HexColor("#0B2545"),
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportHeading",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=12.5,
            leading=15,
            textColor=colors.HexColor("#2E74B5"),
            spaceBefore=10,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=14,
            firstLineIndent=0,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=11,
            alignment=1,
            textColor=colors.HexColor("#555555"),
            spaceAfter=8,
        )
    )

    story = []
    for line in REPORT_MD.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("# "):
            story.append(Paragraph(clean_inline(stripped[2:]), styles["ReportTitle"]))
        elif stripped.startswith("## Project Title:"):
            story.append(Paragraph(clean_inline(stripped[3:]), styles["ReportTitle"]))
        elif stripped.startswith("## "):
            story.append(Paragraph(clean_inline(stripped[3:]), styles["ReportHeading"]))
        elif stripped.startswith("!["):
            if CHART_PATH.exists():
                story.append(Spacer(1, 4))
                story.append(Image(str(CHART_PATH), width=5.8 * inch, height=3.26 * inch))
                story.append(Paragraph("Figure 1. Risk level distribution on the LIAR test set.", styles["Caption"]))
        else:
            story.append(Paragraph(clean_inline(stripped), styles["ReportBody"]))

    doc = SimpleDocTemplate(
        str(REPORT_PDF),
        pagesize=letter,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.85 * inch,
        title="AI-Powered Misinformation Risk Assistant",
    )
    doc.build(story)
    print(REPORT_PDF)


if __name__ == "__main__":
    build_pdf()
