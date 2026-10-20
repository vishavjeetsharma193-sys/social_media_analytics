import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#6B7280"))
        
        if self._pageNumber > 1:
            self.drawString(54, 750, "Personalized Temporal Modeling of Social Media Addiction Risk")
            self.setStrokeColor(colors.HexColor("#E5E7EB"))
            self.setLineWidth(0.5)
            self.line(54, 742, 558, 742)
            
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, page_text)
        self.drawString(54, 36, "CONFIDENTIAL - Project Review & Viva Technical Documentation")
        self.setStrokeColor(colors.HexColor("#E5E7EB"))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)
        
        self.restoreState()

def build_pdf(filename="Social_Media_Analytics_Project_Review.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    PRIMARY = colors.HexColor("#1E3A8A")
    SECONDARY = colors.HexColor("#2563EB")
    ACCENT = colors.HexColor("#0D9488")
    TEXT_DARK = colors.HexColor("#1F2937")
    BG_LIGHT = colors.HexColor("#F8FAFC")
    BORDER_COLOR = colors.HexColor("#CBD5E1")
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=PRIMARY,
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#475569"),
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=PRIMARY,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=TEXT_DARK,
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    qa_q_style = ParagraphStyle(
        'QA_Q',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13.5,
        textColor=PRIMARY,
        spaceBefore=6,
        spaceAfter=2,
        keepWithNext=True
    )

    qa_a_style = ParagraphStyle(
        'QA_A',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=TEXT_DARK,
        spaceAfter=6
    )

    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13,
        textColor=PRIMARY
    )

    story = []

    # Title & Subtitle Banner
    story.append(Paragraph("Personalized Temporal Modeling of Social Media Addiction Risk", title_style))
    story.append(Paragraph("Using Behavioral Change Patterns and Explainable Machine Learning", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=PRIMARY, spaceBefore=0, spaceAfter=12))

    # Section 1: Research Abstract & Key Novelty
    story.append(Paragraph("1. Research Abstract & Methodological Novelty", h1_style))
    abstract_text = (
        "Traditional digital-trace approaches and psychological surveys assess social media addiction using static, "
        "cross-sectional usage snapshots (e.g. total screen time = 6 hrs). This project introduces a personalized temporal modeling "
        "system that measures how a user's behavioral structure evolves relative to their own personal baseline across 4 weeks. "
        "By tracking <b>Δ screen time, Δ session frequency, Δ average session duration, Δ night-time usage, Δ checking frequency, "
        "and Δ sleep-related usage</b>, the system detects trajectories toward problematic use rather than relying on one-off questionnaires."
    )
    story.append(Paragraph(abstract_text, body_style))
    story.append(Spacer(1, 6))

    # Core Pipeline Callout Box
    pipeline_html = Paragraph(
        "<b>Architectural Pipeline:</b><br/>"
        "Social Media Usage Data &rarr; Behavioral Feature Extraction &rarr; Personal Baseline Modeling &rarr; "
        "Behavioral Change Detection (&Delta;) &rarr; XGBoost Classifier &rarr; Addiction-Risk Score &rarr; "
        "SHAP Explanation + Early-Warning Alert & Personalized Addiction Risk Index (PARI).",
        callout_style
    )
    callout_table = Table([[pipeline_html]], colWidths=[504])
    callout_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BG_LIGHT),
        ('BOX', (0, 0), (-1, -1), 1, ACCENT),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(callout_table)
    story.append(Spacer(1, 10))

    # Section 2: Technical Stack & Architecture
    story.append(Paragraph("2. System Architecture & Technical Stack", h1_style))
    
    stack_data = [
        [Paragraph("<b>Component</b>", body_style), Paragraph("<b>Technology / Framework</b>", body_style), Paragraph("<b>Function & Responsibility</b>", body_style)],
        [Paragraph("Web Dashboard", body_style), Paragraph("Dash 2.x & Dash Bootstrap Components", body_style), Paragraph("Reactive web UI, multi-tab layout, theme management (FLATLY)", body_style)],
        [Paragraph("Visualizations", body_style), Paragraph("Plotly Express & Plotly Graph Objects", body_style), Paragraph("Interactive trajectory lines, SHAP bar charts, PARI category distributions", body_style)],
        [Paragraph("Machine Learning Engine", body_style), Paragraph("XGBoost Classifier & Scikit-Learn", body_style), Paragraph("Gradient boosted tree model trained on &Delta; behavioral change features", body_style)],
        [Paragraph("Explainable AI (XAI)", body_style), Paragraph("SHAP (TreeExplainer)", body_style), Paragraph("Local & global Shapley additive explanations for risk predictions", body_style)],
        [Paragraph("Risk Metric Engine", body_style), Paragraph("Personalized Addiction Risk Index (PARI)", body_style), Paragraph("PARI = current-risk prob + behavioral-change score + persistence score", body_style)],
    ]
    
    stack_table = Table(stack_data, colWidths=[110, 150, 244])
    stack_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    for i in range(3):
        stack_data[0][i].style.textColor = colors.white

    story.append(stack_table)
    story.append(Spacer(1, 10))

    # Section 3: Personalized Addiction Risk Index (PARI) Framework
    story.append(Paragraph("3. Personalized Addiction Risk Index (PARI) Framework", h1_style))
    pari_intro = (
        "To provide a dynamic risk-monitoring system rather than a static classifier, we formulate the <b>Personalized Addiction Risk Index (PARI)</b>:"
    )
    story.append(Paragraph(pari_intro, body_style))
    story.append(Paragraph("<b>PARI = Current Risk Probability + Behavioral Change Score + Persistence Score</b>", ParagraphStyle('Formula', parent=body_style, fontName='Helvetica-Bold', textColor=SECONDARY, leftIndent=10)))
    story.append(Spacer(1, 4))

    cat_data = [
        [Paragraph("<b>PARI Classification Category</b>", body_style), Paragraph("<b>Threshold & Criteria</b>", body_style), Paragraph("<b>System Response & Early-Warning Alert</b>", body_style)],
        [Paragraph("<font color='#10B981'><b>🟢 Stable / Low Risk</b></font>", body_style), Paragraph("PARI &lt; 35<br/>Minimal &Delta; escalation", body_style), Paragraph("Behavioral patterns remain within healthy baseline boundaries.", body_style)],
        [Paragraph("<font color='#F59E0B'><b>🟡 Increasing Risk</b></font>", body_style), Paragraph("35 &le; PARI &lt; 60<br/>Moderate &Delta; escalation", body_style), Paragraph("⚠️ Alert: Moderate behavioral escalation relative to personal baseline.", body_style)],
        [Paragraph("<font color='#F97316'><b>🟠 Persistent High Risk</b></font>", body_style), Paragraph("PARI &ge; 60<br/>High persistence score", body_style), Paragraph("🚨 Early Warning: Sustained elevated usage across consecutive weeks.", body_style)],
        [Paragraph("<font color='#EF4444'><b>🔴 Rapidly Increasing Risk</b></font>", body_style), Paragraph("PARI &ge; 70<br/>Steep positive &Delta; trajectory", body_style), Paragraph("🚨 Early Warning: Rapid night usage (+2.3h) & checking freq (+15/h) surge!", body_style)],
    ]
    cat_table = Table(cat_data, colWidths=[130, 140, 234])
    cat_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), SECONDARY),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    for i in range(3):
        cat_data[0][i].style.textColor = colors.white

    story.append(cat_table)
    story.append(Spacer(1, 10))

    # Section 4: Project Review & Viva Defense Questions
    story.append(Paragraph("4. Project Review & Viva Defense Q&A", h1_style))

    qa_list = [
        ("Q1: What is the primary novelty of this research project?",
         "<b>A:</b> The key novelty is modeling <i>behavioral change relative to personal baseline</i> (temporal Δ features over 4 weeks) rather than static cross-sectional usage hours. This allows the system to detect escalation trajectories early."),

        ("Q2: How does the XGBoost + SHAP framework improve explainability?",
         "<b>A:</b> XGBoost accurately captures non-linear risk interactions between Δ night usage and checking frequency. SHAP (SHapley Additive exPlanations) provides exact local feature attributions, explaining precisely which change metrics pushed an individual user's risk up."),

        ("Q3: What is the Personalized Addiction Risk Index (PARI)?",
         "<b>A:</b> PARI is a novel metric integrating model risk probability, normalized positive change magnitude, and multi-week persistence, categorizing users into 4 dynamic risk tiers: Stable, Increasing, Persistent High, and Rapidly Increasing Risk."),

        ("Q4: How does the dashboard deliver real-time interactive updates?",
         "<b>A:</b> Built with Python Dash and Plotly reactive callbacks, selecting a student dropdown instantly updates their 4-week trajectory graph, baseline comparison table, local SHAP explanation bar chart, and early-warning alert banner without page reload.")
    ]

    for q, a in qa_list:
        qa_block = [
            Paragraph(q, qa_q_style),
            Paragraph(a, qa_a_style)
        ]
        story.append(KeepTogether(qa_block))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully generated at: {filename}")
    return filename

if __name__ == "__main__":
    build_pdf()
