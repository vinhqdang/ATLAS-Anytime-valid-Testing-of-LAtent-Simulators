"""Generate the Elsevier Highlights PDF for the ATLAS submission.

Elsevier requires a separate 'Highlights' file: 3-5 bullet points,
each at most 85 characters (including spaces), no author names.
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT

TITLE = ("ATLAS: Anytime-valid Testing of Latent Simulators — "
         "Sequential Certification of World-Model Faithfulness via E-Process Betting")

HIGHLIGHTS = [
    "ATLAS gives world models a running, anytime-valid faithfulness certificate.",
    "Per-horizon betting e-processes need no exchangeability or i.i.d. data.",
    "The faithfulness frontier yields a trust horizon that clips planning depth.",
    "Proves uniform validity, an anytime simulation lemma, and delay optimality.",
    "Real driving, drone, and neural world-model tests confirm the theory holds.",
]

assert 3 <= len(HIGHLIGHTS) <= 5, "Elsevier requires 3-5 highlights"
for b in HIGHLIGHTS:
    assert len(b) <= 85, f"Bullet exceeds 85 characters ({len(b)}): {b}"

styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    "TitleStyle", parent=styles["Title"], fontSize=13, leading=16, alignment=TA_LEFT,
)
heading_style = ParagraphStyle(
    "HeadingStyle", parent=styles["Heading1"], fontSize=14, spaceBefore=18, spaceAfter=10,
)
bullet_style = ParagraphStyle(
    "BulletStyle", parent=styles["Normal"], fontSize=12, leading=17,
    leftIndent=18, firstLineIndent=-18, spaceAfter=10,
)

doc = SimpleDocTemplate(
    "highlights.pdf",
    pagesize=letter,
    leftMargin=1 * inch,
    rightMargin=1 * inch,
    topMargin=1 * inch,
    bottomMargin=1 * inch,
    title="Highlights",
)

story = []
story.append(Paragraph(TITLE, title_style))
story.append(Spacer(1, 6))
story.append(Paragraph("Highlights", heading_style))

for b in HIGHLIGHTS:
    story.append(Paragraph(f"&#8226;&nbsp;&nbsp;{b}", bullet_style))

doc.build(story)
print("Wrote highlights.pdf")
