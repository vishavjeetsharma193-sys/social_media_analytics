"""
Implementation Document Generator
Social Media Addiction Risk Analytics Project
"""

import os
import sys
import textwrap

# ── Step 0: Make sure we run from the project root ──────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

# ── Step 1: Generate chart images for embedding ─────────────────────────────
print("Generating chart images for the document...")

from src.data_processing import generate_temporal_dataset, preprocess_data
from src.ml_engine import train_xgboost_and_compute_pari
from src.visualizations import (
    create_system_architecture_diagram,
    create_student_trajectory_chart,
    create_pari_category_distribution,
    create_behavioral_change_impact_plot,
    create_global_shap_importance_plot,
    create_local_shap_explanation_plot,
    create_platform_pari_chart,
)

PROCESSED_DATA_PATH = "data/students_behavioral_change_features.csv"
if not os.path.exists(PROCESSED_DATA_PATH):
    generate_temporal_dataset()

df_features = preprocess_data()
df_final = train_xgboost_and_compute_pari(df_features)

IMG_DIR = os.path.join(PROJECT_DIR, "screenshots")
os.makedirs(IMG_DIR, exist_ok=True)

chart_specs = {
    "architecture":    create_system_architecture_diagram(),
    "pari_dist":       create_pari_category_distribution(df_final),
    "change_impact":   create_behavioral_change_impact_plot(df_final),
    "global_shap":     create_global_shap_importance_plot(df_final),
    "platform_pari":   create_platform_pari_chart(df_final),
}

# Pick a student from each risk tier for the trajectory + local SHAP screenshots
sample_students = {}
for cat_label in df_final["PARI_Category"].unique():
    row = df_final[df_final["PARI_Category"] == cat_label].iloc[0]
    sample_students[cat_label] = row["Student_ID"]

# Pick two interesting students for trajectory/SHAP screenshots
high_risk_id = None
low_risk_id = None
for cat, sid in sample_students.items():
    if "Rapidly" in cat or "Persistent" in cat:
        high_risk_id = sid
    if "Stable" in cat:
        low_risk_id = sid
if high_risk_id is None:
    high_risk_id = df_final["Student_ID"].iloc[0]
if low_risk_id is None:
    low_risk_id = df_final["Student_ID"].iloc[1]

chart_specs["trajectory_high"] = create_student_trajectory_chart(df_final, high_risk_id)
chart_specs["trajectory_low"]  = create_student_trajectory_chart(df_final, low_risk_id)
chart_specs["local_shap_high"] = create_local_shap_explanation_plot(df_final, high_risk_id)
chart_specs["local_shap_low"]  = create_local_shap_explanation_plot(df_final, low_risk_id)

image_paths = {}
for name, fig in chart_specs.items():
    path = os.path.join(IMG_DIR, f"{name}.png")
    fig.write_image(path, width=900, height=420, scale=2)
    image_paths[name] = path
    print(f"  OK: {name}.png")

print("All chart images exported.\n")


# ── Step 2: Build the PDF document ──────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch, mm, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    KeepTogether, HRFlowable, Image, PageBreak, ListFlowable, ListItem
)
from reportlab.pdfgen import canvas


# ── Student / Course Info (placeholder – fill in your details) ──────────────
STUDENT_NAME   = "Aman Gupta"
STUDENT_NAME_2 = "Vishavjeet Sharma"
REG_NUMBER     = "24BCE0069"
REG_NUMBER_2   = "24BCE0077"
COURSE_CODE    = "BCSE302L"
COURSE_NAME    = "Social Media Analytics"
FACULTY_NAME   = "Dr. V. Ezhil Arasi"
GITHUB_LINK    = "https://github.com/YOUR_USERNAME/social-media-analytics"
SEMESTER       = "Fall 2026"
UNIVERSITY     = "VIT Vellore"


# ── Color Palette ───────────────────────────────────────────────────────────
C_PRIMARY   = colors.HexColor("#1a1a2e")
C_ACCENT    = colors.HexColor("#16213e")
C_BLUE      = colors.HexColor("#0f3460")
C_HIGHLIGHT = colors.HexColor("#e94560")
C_TEXT       = colors.HexColor("#222222")
C_GRAY      = colors.HexColor("#555555")
C_LIGHT_BG  = colors.HexColor("#f5f5f5")
C_TABLE_HEAD = colors.HexColor("#2c3e50")
C_TABLE_ALT  = colors.HexColor("#ecf0f1")
C_WHITE      = colors.white
C_LINK       = colors.HexColor("#2980b9")
C_GREEN      = colors.HexColor("#27ae60")
C_ORANGE     = colors.HexColor("#e67e22")
C_RED        = colors.HexColor("#c0392b")


# ── Page number + footer canvas ─────────────────────────────────────────────
class FooterCanvas(canvas.Canvas):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._pages = []

    def showPage(self):
        self._pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._pages)
        for state in self._pages:
            self.__dict__.update(state)
            self._draw_footer(total)
            super().showPage()
        super().save()

    def _draw_footer(self, total):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#888888"))
        w, h = A4
        # left side
        self.drawString(60, 30, f"{STUDENT_NAME} ({REG_NUMBER})  |  {STUDENT_NAME_2} ({REG_NUMBER_2})  |  {COURSE_CODE}")
        # right side
        self.drawRightString(w - 60, 30, f"Page {self._pageNumber} of {total}")
        self.restoreState()


# ── Styles ──────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

st_title = ParagraphStyle("DocTitle", parent=styles["Normal"],
    fontName="Helvetica-Bold", fontSize=22, leading=28,
    textColor=C_PRIMARY, alignment=TA_CENTER, spaceAfter=4)

st_subtitle = ParagraphStyle("DocSub", parent=styles["Normal"],
    fontName="Helvetica", fontSize=12, leading=16,
    textColor=C_GRAY, alignment=TA_CENTER, spaceAfter=6)

st_meta = ParagraphStyle("Meta", parent=styles["Normal"],
    fontName="Helvetica", fontSize=10, leading=14,
    textColor=C_GRAY, alignment=TA_CENTER, spaceAfter=3)

st_h1 = ParagraphStyle("H1", parent=styles["Normal"],
    fontName="Helvetica-Bold", fontSize=14, leading=18,
    textColor=C_PRIMARY, spaceBefore=14, spaceAfter=6, keepWithNext=True)

st_h2 = ParagraphStyle("H2", parent=styles["Normal"],
    fontName="Helvetica-Bold", fontSize=11, leading=15,
    textColor=C_BLUE, spaceBefore=10, spaceAfter=4, keepWithNext=True)

st_body = ParagraphStyle("Body", parent=styles["Normal"],
    fontName="Helvetica", fontSize=9.5, leading=13.5,
    textColor=C_TEXT, spaceAfter=5, alignment=TA_JUSTIFY)

st_body_small = ParagraphStyle("BodySm", parent=st_body,
    fontSize=8.5, leading=12, spaceAfter=3)

st_bullet = ParagraphStyle("Bullet", parent=st_body,
    leftIndent=18, firstLineIndent=-12, spaceAfter=3)

st_code = ParagraphStyle("Code", parent=styles["Normal"],
    fontName="Courier", fontSize=8, leading=11,
    textColor=C_TEXT, leftIndent=12, spaceAfter=4,
    backColor=C_LIGHT_BG)

st_caption = ParagraphStyle("Caption", parent=styles["Normal"],
    fontName="Helvetica-Oblique", fontSize=8, leading=10,
    textColor=C_GRAY, alignment=TA_CENTER, spaceAfter=8, spaceBefore=2)

st_tbl_hdr = ParagraphStyle("TblH", parent=styles["Normal"],
    fontName="Helvetica-Bold", fontSize=9, leading=12,
    textColor=C_WHITE)

st_tbl_cell = ParagraphStyle("TblC", parent=styles["Normal"],
    fontName="Helvetica", fontSize=8.5, leading=12,
    textColor=C_TEXT)


# ── Helper functions ────────────────────────────────────────────────────────
def heading(text):
    return Paragraph(text, st_h1)

def subheading(text):
    return Paragraph(text, st_h2)

def body(text):
    return Paragraph(text, st_body)

def bullet(text):
    return Paragraph(f"• {text}", st_bullet)

def code_block(text):
    lines = text.strip().split("\n")
    elements = []
    for ln in lines:
        elements.append(Paragraph(ln.replace(" ", "&nbsp;").replace("<", "&lt;").replace(">", "&gt;"), st_code))
    return elements

def chart_image(name, caption_text, w=6.0*inch, h=2.8*inch):
    """Return an Image + caption, keeping together."""
    img = Image(image_paths[name], width=w, height=h)
    cap = Paragraph(caption_text, st_caption)
    return KeepTogether([img, cap])

def make_table(header, rows, col_widths=None):
    """Build a styled Table from header list and list-of-lists."""
    hdr_cells = [Paragraph(f"<b>{h}</b>", st_tbl_hdr) for h in header]
    data = [hdr_cells]
    for row in rows:
        data.append([Paragraph(str(c), st_tbl_cell) for c in row])
    if col_widths is None:
        col_widths = [None] * len(header)
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_TABLE_HEAD),
        ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
        ("ALIGN",         (0, 0), (-1, -1), "LEFT"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#bdc3c7")),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_WHITE, C_TABLE_ALT]),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ]))
    return t


# ═══════════════════════════════════════════════════════════════════════════
#                         BUILD THE STORY
# ═══════════════════════════════════════════════════════════════════════════
story = []

# ── COVER PAGE ──────────────────────────────────────────────────────────────
story.append(Spacer(1, 1.8*inch))
story.append(Paragraph("Personalized Temporal Modeling of<br/>Social Media Addiction Risk", st_title))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "Using Behavioral Change Patterns and Explainable Machine Learning", st_subtitle))
story.append(Spacer(1, 8))
story.append(HRFlowable(width="60%", thickness=1.5, color=C_HIGHLIGHT,
                         spaceBefore=4, spaceAfter=12))

cover_data = [
    ["Student 1", f"{STUDENT_NAME}  ({REG_NUMBER})"],
    ["Student 2", f"{STUDENT_NAME_2}  ({REG_NUMBER_2})"],
    ["Course", f"{COURSE_CODE} -- {COURSE_NAME}"],
    ["Faculty", FACULTY_NAME],
    ["Semester", SEMESTER],
    ["University", UNIVERSITY],
    ["GitHub Repository", GITHUB_LINK],
]
cover_tbl = Table(
    [[Paragraph(f"<b>{r[0]}</b>", st_body), Paragraph(r[1], st_body)] for r in cover_data],
    colWidths=[2.0*inch, 4.2*inch]
)
cover_tbl.setStyle(TableStyle([
    ("ALIGN",     (0, 0), (0, -1), "RIGHT"),
    ("VALIGN",    (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING",(0, 0), (-1, -1), 5),
    ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ("LINEBELOW",(0, 0), (-1, -2), 0.3, colors.HexColor("#dddddd")),
]))
story.append(cover_tbl)
story.append(Spacer(1, 0.6*inch))
story.append(Paragraph(
    f"Submitted on: October 2026", st_meta))
story.append(PageBreak())

# ── TABLE OF CONTENTS ──────────────────────────────────────────────────────
story.append(heading("Table of Contents"))
story.append(Spacer(1, 6))
toc_items = [
    ("1", "Introduction and Problem Statement"),
    ("2", "Objectives"),
    ("3", "Technology Stack"),
    ("4", "System Architecture and Pipeline"),
    ("5", "Implementation Details"),
    ("6", "Personalized Addiction Risk Index (PARI)"),
    ("7", "Configuration and Setup"),
    ("8", "Output Screenshots"),
    ("9", "Key Results and Observations"),
    ("10", "Conclusion"),
    ("11", "GitHub Repository"),
    ("12", "References"),
]
for num, title in toc_items:
    story.append(Paragraph(
        f"<b>{num}.</b>&nbsp;&nbsp;{title}", st_body))
story.append(PageBreak())


# ── 1. INTRODUCTION ────────────────────────────────────────────────────────
story.append(heading("1.  Introduction and Problem Statement"))
story.append(body(
    "Social media usage among students has risen significantly over the past few years, "
    "and with it, the concern around compulsive usage patterns and digital addiction. "
    "Most existing studies rely on one-time surveys or static usage snapshots — for example, "
    "a student reports an average of 6 hours of daily screen time — but these single data points "
    "miss the bigger picture of how a student's behavior is actually changing over time."
))
story.append(body(
    "This project takes a different approach. Instead of asking whether a student uses their "
    "phone for a lot of hours, we ask: <i>is this student's screen time increasing relative to "
    "their own past behavior?</i> Are they checking their phone more often than they used to? "
    "Is their night-time usage creeping up week over week? By tracking these behavioral shifts "
    "across a 4-week observation window, we can catch early signs of escalation before they "
    "snowball into a full-blown problem."
))
story.append(body(
    "The core idea is personalized temporal modeling. Each student serves as their own control. "
    "We compute Week 1 usage as their personal baseline and then measure how much their "
    "behavior deviates from that baseline over the following weeks. These deviations — what we "
    "call delta features — become the primary inputs to an XGBoost machine learning model that "
    "predicts addiction risk. We then use SHAP explainability to show exactly which behavioral "
    "changes are driving a student's risk score, and we wrap the whole thing in an interactive "
    "Dash web dashboard where counselors or researchers can explore the data visually."
))

# ── 2. OBJECTIVES ──────────────────────────────────────────────────────────
story.append(heading("2.  Objectives"))
objectives = [
    "Simulate realistic longitudinal social media usage data for 800 students across 4 weeks, "
    "covering metrics like screen time, session frequency, night-time usage, and checking frequency.",
    "Compute personalized baseline profiles (Week 1) and delta features that capture how each "
    "student's behavior deviates from their own norm over time.",
    "Train an XGBoost gradient-boosted classifier on these delta features to predict addiction risk, "
    "achieving high discriminative performance.",
    "Use SHAP (TreeExplainer) to provide both global feature importance rankings and local, "
    "per-student explanations of what is driving their individual risk prediction.",
    "Formulate the Personalized Addiction Risk Index (PARI), a composite score combining model "
    "risk probability, behavioral change magnitude, and multi-week persistence into a single "
    "interpretable number between 0 and 100.",
    "Build an interactive Dash web dashboard with reactive filtering, student-level drill-down, "
    "trajectory visualizations, and real-time early-warning alert banners.",
]
for obj in objectives:
    story.append(bullet(obj))

# ── 3. TECHNOLOGY STACK ────────────────────────────────────────────────────
story.append(heading("3.  Technology Stack"))
story.append(body(
    "The project is implemented entirely in Python 3.12 and relies on the following "
    "libraries and frameworks:"
))

stack_rows = [
    ["Python 3.12",       "Programming Language",   "Core language used for all modules"],
    ["Pandas",            "Data Processing",        "DataFrame operations, CSV I/O, feature aggregation"],
    ["NumPy",             "Numerical Computing",    "Array operations, random seed control, statistical functions"],
    ["Scikit-Learn",      "ML Utilities",           "StandardScaler, GradientBoostingClassifier fallback, KMeans"],
    ["XGBoost",           "ML Model",               "XGBClassifier for gradient-boosted tree training and inference"],
    ["SHAP",              "Explainable AI",         "TreeExplainer for local and global Shapley value computation"],
    ["Plotly",            "Visualization",          "Plotly Express and Graph Objects for interactive charts"],
    ["Dash",              "Web Framework",          "Reactive web application framework with callback-driven updates"],
    ["Dash Bootstrap",    "UI Components",          "Responsive grid layout, cards, tabs, alerts, tables"],
    ["Statsmodels",       "Statistics",             "Time-series trend analysis support"],
    ["ReportLab",         "PDF Generation",         "Programmatic PDF document creation for this report"],
]
story.append(make_table(
    ["Library / Tool", "Category", "Purpose"],
    stack_rows,
    col_widths=[1.4*inch, 1.3*inch, 3.6*inch]
))
story.append(Spacer(1, 6))

# ── 4. SYSTEM ARCHITECTURE ─────────────────────────────────────────────────
story.append(heading("4.  System Architecture and Pipeline"))
story.append(body(
    "The system follows a linear data pipeline that transforms raw daily usage logs into "
    "actionable, explainable risk assessments. The overall flow is depicted in the diagram below."
))
story.append(chart_image("architecture",
    "Fig 1.  System architecture flowchart showing the end-to-end processing pipeline",
    w=6.2*inch, h=2.0*inch))

story.append(subheading("4.1  Pipeline Stages"))
stages = [
    ("<b>Raw Data Ingestion:</b> Longitudinal 4-week daily usage traces are generated (or loaded) for "
     "800 students. Each trace includes daily screen time, session frequency, average session duration, "
     "night-time usage, device checking frequency, and sleep-related usage hours."),
    ("<b>Behavioral Feature Extraction:</b> From the raw weekly traces, structured features are computed — "
     "both absolute current-week values and per-student baseline values from Week 1."),
    ("<b>Personal Baseline Modeling:</b> Week 1 is treated as each student's personal norm. This baseline "
     "captures their natural usage patterns before any potential escalation."),
    ("<b>Behavioral Change Detection (Delta Features):</b> The system computes the difference between "
     "Week 4 current values and Week 1 baseline for each metric. These deltas represent how much "
     "the student's behavior has shifted."),
    ("<b>XGBoost Classifier:</b> A gradient-boosted tree model is trained on the 13 extracted features "
     "(6 baseline, 6 delta, 1 variability) to predict a binary high-risk target."),
    ("<b>SHAP Explanation:</b> TreeExplainer generates Shapley values for each feature, enabling "
     "per-student local explanations and cohort-wide global feature importance ranking."),
    ("<b>PARI Score and Early-Warning Alerts:</b> The Personalized Addiction Risk Index combines model "
     "probability, change score, and persistence into a 0–100 index, and the system generates "
     "context-specific alert messages for at-risk students."),
]
for s in stages:
    story.append(bullet(s))

# ── 5. IMPLEMENTATION DETAILS ──────────────────────────────────────────────
story.append(PageBreak())
story.append(heading("5.  Implementation Details"))

story.append(subheading("5.1  Project Structure"))
struct_text = [
    "social_media_analytics/",
    "├── data/",
    "│   ├── students_temporal_usage.csv",
    "│   ├── students_behavioral_change_features.csv",
    "│   └── students_temporal_pari_cache.csv",
    "├── src/",
    "│   ├── data_processing.py",
    "│   ├── ml_engine.py",
    "│   ├── clustering.py",
    "│   └── visualizations.py",
    "├── screenshots/",
    "├── app.py",
    "├── requirements.txt",
    "└── README.md",
]
for ln in struct_text:
    story.append(Paragraph(ln.replace(" ", "&nbsp;"), st_code))
story.append(Spacer(1, 6))

story.append(subheading("5.2  Data Generation Module (data_processing.py)"))
story.append(body(
    "The data generation module creates synthetic but realistic longitudinal usage data for "
    "800 students. Each student is assigned to one of four behavioral archetypes that control "
    "how their usage evolves over the 4-week window:"
))
archetype_rows = [
    ["Stable / Low Risk (40%)",       "Low base usage, near-zero growth factor",
     "Student maintains healthy usage with minimal week-over-week change"],
    ["Increasing Risk (30%)",         "Moderate base usage, growth factor 0.3–0.7",
     "Gradual escalation in screen time and night usage over weeks"],
    ["Persistent High Risk (15%)",    "High base usage, slow growth factor",
     "Already heavy user who stays consistently elevated"],
    ["Rapidly Increasing (15%)",      "Low-to-moderate base, growth factor 0.9–1.5",
     "Dramatic spike from normal baseline to heavy usage within weeks"],
]
story.append(make_table(
    ["Archetype", "Configuration", "Behavioral Pattern"],
    archetype_rows,
    col_widths=[1.6*inch, 1.8*inch, 2.9*inch]
))
story.append(Spacer(1, 4))
story.append(body(
    "For each student, the module simulates 4 weekly metric values with controlled noise, "
    "computes the personal baseline from Week 1, calculates all delta features, derives a ground-truth "
    "risk probability using a logistic function, and exports both the raw temporal traces and the "
    "processed feature matrix to CSV files."
))

story.append(subheading("5.3  Machine Learning Engine (ml_engine.py)"))
story.append(body(
    "The ML engine trains an XGBoost classifier with the following configuration:"
))
ml_rows = [
    ["n_estimators", "100", "Number of boosting rounds"],
    ["max_depth", "4", "Maximum tree depth (prevents overfitting)"],
    ["learning_rate", "0.08", "Step size shrinkage for regularization"],
    ["eval_metric", "logloss", "Binary cross-entropy loss function"],
    ["random_state", "42", "Reproducibility seed"],
]
story.append(make_table(
    ["Hyperparameter", "Value", "Rationale"],
    ml_rows,
    col_widths=[1.5*inch, 1.0*inch, 3.8*inch]
))
story.append(Spacer(1, 4))
story.append(body(
    "The model takes 13 input features — 3 baseline metrics (screen time, night usage, checking "
    "frequency), 3 corresponding current-week values, 6 delta features (screen time, session freq, "
    "session duration, night usage, checking freq, sleep usage), and usage variability (standard "
    "deviation of screen time across weeks). After training, SHAP TreeExplainer is used to compute "
    "Shapley values for each prediction. A fallback mechanism using feature importance approximation "
    "is included in case SHAP initialization fails."
))

story.append(subheading("5.4  Visualization Module (visualizations.py)"))
story.append(body(
    "All charts are generated using Plotly Express and Plotly Graph Objects with the plotly_white "
    "template for a clean aesthetic. The module provides the following chart generators:"
))
viz_items = [
    "<b>System Architecture Flowchart</b> — Interactive Sankey-style pipeline diagram with hover tooltips",
    "<b>Student Trajectory Chart</b> — 4-week line chart showing screen time evolution vs personal baseline",
    "<b>PARI Category Distribution</b> — Color-coded bar chart of the four risk tiers",
    "<b>Behavioral Change Impact Plot</b> — Scatter plot of delta night usage vs PARI score, sized by screen time",
    "<b>Global SHAP Importance</b> — Horizontal bar chart comparing delta features vs static features",
    "<b>Local SHAP Explanation</b> — Per-student feature contribution waterfall chart",
    "<b>Platform Risk Breakdown</b> — Stacked bar chart of risk categories by social media platform",
]
for v in viz_items:
    story.append(bullet(v))

story.append(subheading("5.5  Web Application (app.py)"))
story.append(body(
    "The Dash application serves as the primary interactive interface. It uses Dash Bootstrap "
    "Components with the FLATLY theme and organizes the dashboard into three tabs:"
))
tab_items = [
    "<b>Tab 1 — Architecture and Trajectory:</b> Displays the pipeline flowchart, per-student "
    "4-week trajectory graphs, personal baseline comparison metrics table, and the early-warning "
    "alert banner that changes color based on risk level (green, blue, orange, red).",
    "<b>Tab 2 — Behavioral Change and PARI Index:</b> Shows the PARI category distribution, "
    "behavioral change impact scatter plot, and platform-wise risk stacked bar chart. All three "
    "charts react to global filters.",
    "<b>Tab 3 — XGBoost and SHAP Explainable AI:</b> Contains the global SHAP feature importance "
    "chart, local per-student SHAP explanation chart, and a methodology summary card.",
]
for t in tab_items:
    story.append(bullet(t))
story.append(body(
    "The left sidebar provides interactive dropdown filters for student selection, risk level, "
    "academic level, and social media platform. All charts update reactively through Dash callbacks "
    "without page reloads. The application automatically opens in the default browser on launch."
))

# ── 6. PARI FRAMEWORK ─────────────────────────────────────────────────────
story.append(PageBreak())
story.append(heading("6.  Personalized Addiction Risk Index (PARI)"))
story.append(body(
    "The PARI is a composite metric designed to capture not just the current predicted risk "
    "but also the trajectory and consistency of behavioral escalation. It is computed as follows:"
))
story.append(Spacer(1, 3))
formula_text = Paragraph(
    "<b>PARI = 0.40 × Current Risk Probability + 0.35 × Behavioral Change Score + "
    "0.25 × Persistence Score</b>",
    ParagraphStyle("Formula", parent=st_body, fontName="Helvetica-Bold",
                   textColor=C_BLUE, alignment=TA_CENTER, fontSize=10, leading=14))
story.append(formula_text)
story.append(Spacer(1, 6))

story.append(subheading("6.1  Component Definitions"))
comp_items = [
    "<b>Current Risk Probability (weight 0.40):</b> The raw output probability from the XGBoost "
    "classifier, ranging from 0.0 to 1.0. This reflects the model's assessment based on all 13 features.",
    "<b>Behavioral Change Score (weight 0.35):</b> A normalized composite of positive delta escalations — "
    "specifically, 35% from delta screen time, 40% from delta night usage, 15% from delta checking "
    "frequency, and 10% from usage variability. This component is clipped to [0, 1].",
    "<b>Persistence Score (weight 0.25):</b> Measures whether the student's screen time has been "
    "consistently increasing across weeks (W1 ≤ W2 ≤ W3 ≤ W4) and whether later weeks exceed "
    "a 4.5-hour threshold. This catches students who maintain elevated usage over time.",
]
for c in comp_items:
    story.append(bullet(c))

story.append(subheading("6.2  Risk Classification Thresholds"))
cat_rows = [
    ["Stable / Low Risk",        "PARI < 35",                          "Minimal deviation from baseline",
     "Healthy patterns within personal norms"],
    ["Increasing Risk",          "35 ≤ PARI < 60",                     "Moderate positive delta escalation",
     "Behavioral trends show initial escalation"],
    ["Persistent High Risk",     "PARI ≥ 60 or (prob ≥ 0.65 & persist ≥ 0.60)", "Sustained high usage",
     "Elevated usage across consecutive weeks"],
    ["Rapidly Increasing Risk",  "PARI ≥ 70 & change ≥ 0.60",         "Steep trajectory surge",
     "Dramatic behavioral spike from baseline"],
]
story.append(make_table(
    ["Category", "Threshold", "Trigger Condition", "Alert Summary"],
    cat_rows,
    col_widths=[1.4*inch, 1.7*inch, 1.4*inch, 1.8*inch]
))

# ── 7. CONFIGURATION ──────────────────────────────────────────────────────
story.append(heading("7.  Configuration and Setup"))
story.append(subheading("7.1  Prerequisites"))
prereqs = [
    "Python 3.10 or higher installed on the system",
    "pip package manager available",
    "A modern web browser (Chrome, Firefox, Edge) for the dashboard",
    "Approximately 50 MB of disk space for dependencies and data files",
]
for p in prereqs:
    story.append(bullet(p))

story.append(subheading("7.2  Installation Steps"))
install_code = [
    "# Clone the repository",
    "git clone " + GITHUB_LINK,
    "cd social-media-analytics",
    "",
    "# Install required packages",
    "pip install -r requirements.txt",
    "",
    "# Launch the dashboard",
    "python app.py",
]
for ln in install_code:
    story.append(Paragraph(
        ln.replace(" ", "&nbsp;").replace("<", "&lt;").replace(">", "&gt;") if ln else "&nbsp;",
        st_code))
story.append(Spacer(1, 4))

story.append(subheading("7.3  Requirements File Contents"))
story.append(body("The requirements.txt file includes the following packages:"))
req_lines = ["pandas", "numpy", "scikit-learn", "plotly", "dash",
             "dash-bootstrap-components", "statsmodels", "xgboost", "shap", "reportlab"]
for r in req_lines:
    story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;{r}", st_code))
story.append(Spacer(1, 4))

story.append(subheading("7.4  Runtime Configuration"))
config_rows = [
    ["Dashboard Port",     "8050",   "Configurable in app.py (app.run port argument)"],
    ["Debug Mode",         "True",   "Hot-reloading enabled during development"],
    ["Dataset Size",       "800",    "Number of students in generate_temporal_dataset()"],
    ["Random Seed",        "42",     "Ensures reproducibility across runs"],
    ["Observation Window", "4 weeks","Longitudinal trace duration per student"],
    ["Bootstrap Theme",    "FLATLY", "Dash Bootstrap Components theme"],
    ["Cache Files",        "CSV",    "PARI scores cached to avoid recomputation"],
]
story.append(make_table(
    ["Parameter", "Default Value", "Notes"],
    config_rows,
    col_widths=[1.5*inch, 1.0*inch, 3.8*inch]
))

# ── 8. SCREENSHOTS ────────────────────────────────────────────────────────
story.append(PageBreak())
story.append(heading("8.  Output Screenshots"))
story.append(body(
    "The following screenshots show the actual outputs generated by the system, including "
    "the interactive dashboard charts and analytical visualizations."
))
story.append(Spacer(1, 4))

story.append(subheading("8.1  PARI Risk Category Distribution"))
story.append(chart_image("pari_dist",
    "Fig 2.  Distribution of students across the four PARI risk categories"))

story.append(subheading("8.2  Behavioral Change Impact Analysis"))
story.append(chart_image("change_impact",
    "Fig 3.  Scatter plot showing how delta night usage drives PARI risk scores"))

story.append(PageBreak())
story.append(subheading("8.3  Student Temporal Trajectory — High Risk Example"))
story.append(chart_image("trajectory_high",
    f"Fig 4.  4-week screen time trajectory for a high-risk student ({high_risk_id})"))

story.append(subheading("8.4  Student Temporal Trajectory — Low Risk Example"))
story.append(chart_image("trajectory_low",
    f"Fig 5.  4-week screen time trajectory for a stable/low-risk student ({low_risk_id})"))

story.append(PageBreak())
story.append(subheading("8.5  Global SHAP Feature Importance"))
story.append(chart_image("global_shap",
    "Fig 6.  Global SHAP analysis — behavioral change features vs static usage metrics"))

story.append(subheading("8.6  Local SHAP Explanation — High Risk Student"))
story.append(chart_image("local_shap_high",
    f"Fig 7.  Per-student SHAP explanation for {high_risk_id} showing individual risk drivers"))

story.append(PageBreak())
story.append(subheading("8.7  Local SHAP Explanation — Low Risk Student"))
story.append(chart_image("local_shap_low",
    f"Fig 8.  Per-student SHAP explanation for {low_risk_id} showing protective factors"))

story.append(subheading("8.8  Platform-wise PARI Risk Breakdown"))
story.append(chart_image("platform_pari",
    "Fig 9.  PARI risk severity distribution across different social media platforms"))


# ── 9. KEY RESULTS ────────────────────────────────────────────────────────
story.append(PageBreak())
story.append(heading("9.  Key Results and Observations"))

story.append(subheading("9.1  Model Performance"))
story.append(body(
    "The XGBoost classifier achieves strong discriminative performance on the behavioral change "
    "feature set, accurately separating high-risk and low-risk trajectories. The model benefits "
    "from the temporal delta features, which provide richer signal than static usage hours alone."
))

story.append(subheading("9.2  Feature Importance Insights"))
story.append(body(
    "One of the most interesting findings from the SHAP analysis is that the behavioral change "
    "features — particularly delta night usage and delta screen time — contribute more to the "
    "model's predictions than the absolute static values. This validates the core hypothesis of "
    "the project: it is not how much you use social media that matters most, but how much more "
    "you are using it compared to your own past behavior."
))

story.append(subheading("9.3  PARI Distribution"))
story.append(body(
    "The PARI framework successfully stratifies the student population into meaningful risk "
    "tiers. The distribution roughly follows the archetype proportions used in data generation, "
    "but with some crossover due to noise and the multi-factor nature of the PARI formula. "
    "Students classified as Rapidly Increasing Risk tend to show the steepest week-over-week "
    "escalation in night-time usage, while Persistent High Risk students maintain consistently "
    "elevated absolute usage levels."
))

story.append(subheading("9.4  Platform Differences"))
story.append(body(
    "The platform-wise analysis shows variation in risk category distribution across social "
    "media platforms. While the synthetic data does not model platform-specific addiction "
    "mechanisms, the visualization demonstrates the dashboard's ability to slice and compare "
    "risk profiles across categorical dimensions — a feature that would be valuable when applied "
    "to real-world data."
))

# ── 10. CONCLUSION ────────────────────────────────────────────────────────
story.append(heading("10.  Conclusion"))
story.append(body(
    "This project demonstrates a practical approach to modeling social media addiction risk that "
    "goes beyond static, one-time assessments. By computing personalized baselines and tracking "
    "how each student's behavior changes over time, the system captures early warning signs that "
    "would be invisible to traditional cross-sectional analysis."
))
story.append(body(
    "The combination of XGBoost classification with SHAP explainability ensures that the risk "
    "predictions are not just accurate but also interpretable — counselors and researchers can "
    "see exactly which behavioral changes are driving a particular student's risk score. The PARI "
    "framework adds a layer of nuance by incorporating persistence and change magnitude alongside "
    "the raw model probability."
))
story.append(body(
    "The interactive Dash dashboard ties everything together into a usable tool, with real-time "
    "filtering, student-level drill-down, and visual alerts that make the system accessible to "
    "non-technical users. While the current implementation uses synthetic data, the architecture "
    "is designed to be directly applicable to real-world digital trace data collected through "
    "screen time APIs or usage logging applications."
))

# ── 11. GITHUB REPOSITORY ─────────────────────────────────────────────────
story.append(heading("11.  GitHub Repository"))
story.append(body(
    f"The complete source code, data files, and documentation for this project are available at:"
))
story.append(Spacer(1, 4))
story.append(Paragraph(
    f'<font color="#2980b9"><b>{GITHUB_LINK}</b></font>',
    ParagraphStyle("Link", parent=st_body, alignment=TA_CENTER, fontSize=11, leading=15)))
story.append(Spacer(1, 6))
story.append(body(
    "The repository includes the application source code, data generation scripts, cached "
    "datasets, the interactive dashboard, and this implementation document."
))

# ── 12. REFERENCES ─────────────────────────────────────────────────────────
story.append(heading("12.  References"))
refs = [
    "Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model "
    "Predictions. <i>Advances in Neural Information Processing Systems</i>, 30.",
    "Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. "
    "<i>Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery "
    "and Data Mining</i>, 785–794.",
    "Andreassen, C. S. (2015). Online Social Network Site Addiction: A Comprehensive Review. "
    "<i>Current Addiction Reports</i>, 2(2), 175–184.",
    "Plotly Technologies Inc. (2015). Collaborative Data Science. Montreal, QC. "
    "https://plotly.com",
    "Dash by Plotly. (2017). Analytical Web Applications for Python. "
    "https://dash.plotly.com",
    "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. "
    "<i>Journal of Machine Learning Research</i>, 12, 2825–2830.",
    "Kuss, D. J., & Griffiths, M. D. (2017). Social Networking Sites and Addiction: "
    "Ten Lessons Learned. <i>International Journal of Environmental Research and "
    "Public Health</i>, 14(3), 311.",
]
for i, r in enumerate(refs, 1):
    story.append(Paragraph(f"[{i}]&nbsp;&nbsp;{r}", st_body_small))


# ═══════════════════════════════════════════════════════════════════════════
#                         GENERATE PDF
# ═══════════════════════════════════════════════════════════════════════════
OUTPUT_PATH = os.path.join(PROJECT_DIR, "Implementation_Document.pdf")

doc = SimpleDocTemplate(
    OUTPUT_PATH,
    pagesize=A4,
    leftMargin=60,
    rightMargin=60,
    topMargin=50,
    bottomMargin=50,
)

doc.build(story, canvasmaker=FooterCanvas)
print(f"\n{'='*60}")
print(f"PDF generated successfully: {OUTPUT_PATH}")
print(f"{'='*60}")
