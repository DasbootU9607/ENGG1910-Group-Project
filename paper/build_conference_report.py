from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


PAPER_DIR = Path(__file__).resolve().parent
REPORT_MD = PAPER_DIR / "report.md"
REPORT_PDF = PAPER_DIR / "report.pdf"
FIGURE_DIR = PAPER_DIR / "figures"


METRICS_TABLE = [
    ["Model", "Acc.", "Macro Prec.", "Macro Rec.", "Macro-F1", "Weighted-F1", "Macro AUC"],
    ["TF-IDF text-only", "0.422", "0.424", "0.427", "0.421", "0.421", "0.611"],
    ["BERT-base text-only", "0.496", "0.495", "0.489", "0.491", "0.495", "0.670"],
]


FIGURES = {
    "## III. Data Requirement and Preparation": [
        ("label_distribution.png", "Figure 1: Original LIAR label distribution on the official test split."),
    ],
    "## VI. Key Findings and Observations": [
        ("model_comparison.png", "Figure 2: Model comparison across multiclass classification metrics."),
        ("roc_curves.png", "Figure 3: One-vs-rest ROC curves for the fair text-only models."),
    ],
    "## VII. Discussion and Reflection": [
        ("risk_distribution.png", "Figure 4: Predicted three-level risk distribution from BERT-base."),
        ("baseline_confusion_matrix.png", "Figure 5: Three-class confusion matrix for TF-IDF text-only logistic regression."),
        ("transformer_confusion_matrix.png", "Figure 6: Three-class confusion matrix for BERT-base text-only fine-tuning."),
        ("score_distribution.png", "Figure 7: Distribution of continuous risk scores computed from Medium and High probabilities."),
    ],
}


def clean(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("`", "")


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "PaperTitle",
            parent=base["Title"],
            fontName="Times-Bold",
            fontSize=16,
            leading=18,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "authors": ParagraphStyle(
            "Authors",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=10,
            leading=12,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "heading": ParagraphStyle(
            "PaperHeading",
            parent=base["Heading1"],
            fontName="Times-Bold",
            fontSize=11,
            leading=13,
            spaceBefore=8,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "PaperBody",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=9.4,
            leading=11.2,
            alignment=TA_JUSTIFY,
            spaceAfter=4,
        ),
        "abstract": ParagraphStyle(
            "Abstract",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=9.4,
            leading=11.2,
            alignment=TA_JUSTIFY,
            leftIndent=0.12 * inch,
            rightIndent=0.12 * inch,
            spaceAfter=6,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=8.2,
            leading=9.5,
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
    }


def add_figure(story: list, filename: str, caption: str, style_map: dict[str, ParagraphStyle], width: float = 5.4) -> None:
    path = FIGURE_DIR / filename
    if not path.exists():
        return
    story.append(Spacer(1, 3))
    story.append(Image(str(path), width=width * inch, height=width * 0.56 * inch))
    story.append(Paragraph(clean(caption), style_map["caption"]))


def add_metrics_table(story: list, style_map: dict[str, ParagraphStyle]) -> None:
    table = Table(METRICS_TABLE, colWidths=[1.55 * inch, 0.51 * inch, 0.72 * inch, 0.72 * inch, 0.67 * inch, 0.76 * inch, 0.67 * inch])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.6),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(table)
    story.append(Paragraph("Table 1: Test-set comparison under the fair three-class LIAR protocol.", style_map["caption"]))


def build_pdf() -> None:
    style_map = styles()
    story: list = []
    lines = REPORT_MD.read_text(encoding="utf-8").splitlines()
    in_table = False

    story.append(Paragraph("AI-Powered Misinformation Risk Assistance for Social Media Users", style_map["title"]))
    story.append(Paragraph("ENGG1910 Course Project", style_map["authors"]))

    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("| Model |"):
            add_metrics_table(story, style_map)
            in_table = True
            continue
        if in_table and stripped.startswith("|"):
            continue
        in_table = False

        if stripped.startswith("## "):
            story.append(Paragraph(clean(stripped[3:]), style_map["heading"]))
            for filename, caption in FIGURES.get(stripped, []):
                add_figure(story, filename, caption, style_map)
        elif stripped == "We present a misinformation risk assistant for social media users.":
            story.append(Paragraph(clean(stripped), style_map["abstract"]))
        elif stripped.startswith("["):
            story.append(Paragraph(clean(stripped), style_map["body"]))
        elif stripped.startswith("# "):
            continue
        else:
            story.append(Paragraph(clean(stripped), style_map["body"]))

    doc = SimpleDocTemplate(
        str(REPORT_PDF),
        pagesize=letter,
        rightMargin=0.72 * inch,
        leftMargin=0.72 * inch,
        topMargin=0.72 * inch,
        bottomMargin=0.72 * inch,
        title="AI-Powered Misinformation Risk Assistance",
        author="ENGG1910 Group Project",
    )
    doc.build(story)
    print(REPORT_PDF)


if __name__ == "__main__":
    build_pdf()
