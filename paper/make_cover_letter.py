"""Render cover_letter.txt to cover_letter.pdf for the Pattern Recognition submission."""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

with open("cover_letter.txt") as f:
    raw = f.read()

paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]

styles = getSampleStyleSheet()
body_style = ParagraphStyle(
    "Body", parent=styles["Normal"], fontSize=11, leading=15, spaceAfter=10,
)

doc = SimpleDocTemplate(
    "cover_letter.pdf",
    pagesize=letter,
    leftMargin=1 * inch,
    rightMargin=1 * inch,
    topMargin=1 * inch,
    bottomMargin=1 * inch,
    title="Cover Letter",
)

story = []
for p in paragraphs:
    text = " ".join(line.strip() for line in p.splitlines())
    story.append(Paragraph(text, body_style))
    story.append(Spacer(1, 2))

doc.build(story)
print("Wrote cover_letter.pdf")
