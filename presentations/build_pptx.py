"""Build presentations/stakeholder_deck.pptx — mirrors stakeholder_deck.html.

Run:  python presentations/build_pptx.py
Requires: python-pptx (pip install python-pptx)
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures"
PRES_FIG = ROOT / "presentations" / "figures"
OUT = ROOT / "presentations" / "stakeholder_deck.pptx"

# academic-pptx palette: navy primary, mid-blue accent, white bg
NAVY = RGBColor(0x1F, 0x4E, 0x79)
BLUE = RGBColor(0x2E, 0x75, 0xB6)
INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTE = RGBColor(0x59, 0x59, 0x59)
FAINT = RGBColor(0x8A, 0x8A, 0x8A)
LINE = RGBColor(0xD9, 0xD9, 0xD9)
SOFT = RGBColor(0xF2, 0xF6, 0xFA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
WARN = RGBColor(0x9A, 0x62, 0x00)
BAD = RGBColor(0xA6, 0x1B, 0x1B)
GOOD = RGBColor(0x1E, 0x6B, 0x3A)
FONT = "Calibri"

SW, SH = Inches(13.333), Inches(7.5)
ML, MR, MT, MB = Inches(0.55), Inches(0.55), Inches(0.45), Inches(0.4)
CONTENT_TOP = MT + Inches(1.15)

prs = Presentation()
prs.slide_width, prs.slide_height = SW, SH
BLANK = prs.slide_layouts[6]


def _set(run, text, size=18, bold=False, color=INK, italic=False):
    run.text = text
    f = run.font
    f.name, f.size, f.bold, f.italic, f.color.rgb = FONT, Pt(size), bold, italic, color


def box(slide, x, y, w, h, fill=None, line=None, line_w=0.75, shape=MSO_SHAPE.RECTANGLE, radius=False):
    sp = slide.shapes.add_shape(shape, x, y, w, h)
    if fill is None:
        sp.fill.background()
    else:
        sp.fill.solid(); sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line; sp.line.width = Pt(line_w)
    sp.shadow.inherit = False
    if radius and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        try:
            sp.adjustments[0] = 0.08
        except Exception:
            pass
    return sp


def text(slide, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, space_after=6):
    """runs: list of paragraphs; each paragraph is list of (text,size,bold,color,italic) tuples."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        for t, sz, b, c, it in para:
            _set(p.add_run(), t, sz, b, c, it)
    return tb


def bullets(slide, x, y, w, h, items, size=17, gap=8):
    paras = []
    for it in items:
        if isinstance(it, tuple):
            head, rest = it
            paras.append([("▪ ", size, True, BLUE, False), (head, size, True, INK, False), ("  " + rest, size, False, MUTE, False)])
        else:
            paras.append([("▪ ", size, True, BLUE, False), (it, size, False, INK, False)])
    return text(slide, x, y, w, h, paras, space_after=gap)


def header(slide, kicker, title, num=None):
    text(slide, ML, Inches(0.32), SW - ML - MR, Inches(0.3),
         [[(kicker.upper(), 12, True, BLUE, False)]])
    text(slide, ML, Inches(0.62), SW - ML - MR, Inches(0.85),
         [[(title, 25, True, NAVY, False)]])
    if num:
        text(slide, SW - MR - Inches(1.2), SH - Inches(0.42), Inches(1.2), Inches(0.3),
             [[(f"{num} / 28", 11, False, FAINT, False)]], align=PP_ALIGN.RIGHT)
    text(slide, ML, SH - Inches(0.42), Inches(6), Inches(0.3),
         [[("credit_risk_modeling · stakeholder briefing", 10, False, FAINT, False)]])


def notes(slide, s):
    slide.notes_slide.notes_text_frame.text = s


def table(slide, x, y, w, rows, col_ws=None, font=14, header_font=13, row_h=0.34,
          aligns=None, bold_first=False):
    n_r, n_c = len(rows), len(rows[0])
    gfx = slide.shapes.add_table(n_r, n_c, x, y, w, Inches(row_h * n_r)).table
    if col_ws:
        total = sum(col_ws)
        for i, cw in enumerate(col_ws):
            gfx.columns[i].width = Emu(int(w * cw / total))
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = gfx.cell(r, c)
            cell.margin_left = Inches(0.08); cell.margin_right = Inches(0.08)
            cell.margin_top = Inches(0.02); cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            if r == 0:
                cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
            else:
                cell.fill.solid(); cell.fill.fore_color.rgb = WHITE if r % 2 else SOFT
            tf = cell.text_frame; tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = (aligns[c] if aligns else PP_ALIGN.LEFT)
            _set(p.add_run(), str(val), header_font if r == 0 else font,
                 bold=(r == 0 or (bold_first and c == 0)),
                 color=WHITE if r == 0 else INK)
    return gfx


def pic(slide, path, x, y, w=None, h=None):
    kw = {}
    if w: kw["width"] = w
    if h: kw["height"] = h
    return slide.shapes.add_picture(str(path), x, y, **kw)


def kpi(slide, x, y, w, h, value, label, vsize=34, lsize=13, vcolor=NAVY):
    box(slide, x, y, w, h, fill=WHITE, line=LINE, radius=True)
    text(slide, x + Inches(0.15), y + Inches(0.12), w - Inches(0.3), Inches(0.55),
         [[(value, vsize, True, vcolor, False)]])
    text(slide, x + Inches(0.15), y + h - Inches(0.42), w - Inches(0.3), Inches(0.35),
         [[(label, lsize, False, MUTE, False)]])


def card(slide, x, y, w, h, title, body_paras, tcolor=NAVY, fill=WHITE):
    box(slide, x, y, w, h, fill=fill, line=LINE, radius=True)
    tb = slide.shapes.add_textbox(x + Inches(0.18), y + Inches(0.14), w - Inches(0.36), h - Inches(0.28))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; _set(p.add_run(), title, 16, True, tcolor)
    p.space_after = Pt(6)
    for para in body_paras:
        p = tf.add_paragraph(); p.space_after = Pt(4)
        for t, sz, b, c, it in para:
            _set(p.add_run(), t, sz, b, c, it)


def src(slide, s):
    text(slide, ML, SH - Inches(0.62), SW - ML - MR, Inches(0.25),
         [[(s, 10, False, FAINT, False)]])


def callout(slide, x, y, w, h, paras, accent=BLUE):
    bar = box(slide, x, y, Inches(0.07), h, fill=accent)
    box(slide, x + Inches(0.07), y, w - Inches(0.07), h, fill=SOFT, line=None)
    text(slide, x + Inches(0.25), y + Inches(0.1), w - Inches(0.45), h - Inches(0.2), paras)


N = [0]
def new_slide(kicker=None, title=None, num=True):
    s = prs.slides.add_slide(BLANK)
    N[0] += 1
    if kicker is not None:
        header(s, kicker, title or "", N[0] if num else None)
    return s


# ---------------------------------------------------------------- 1 Cover
s = prs.slides.add_slide(BLANK); N[0] += 1
box(s, 0, 0, SW, SH, fill=NAVY)
text(s, Inches(0.9), Inches(2.0), Inches(11.5), Inches(0.4),
     [[("STAKEHOLDER BRIEFING · CREDIT RISK MODELING", 14, True, RGBColor(0x9D, 0xC3, 0xE6), False)]])
text(s, Inches(0.9), Inches(2.5), Inches(11.5), Inches(1.8),
     [[("Two validated probability-of-default models,", 44, True, WHITE, False)],
      [("ready for pilot", 44, True, WHITE, False)]])
text(s, Inches(0.9), Inches(4.6), Inches(11), Inches(1.4),
     [[("Scenario A scores loan applicants before funding. Scenario B monitors the funded book. "
        "Both are CatBoost models evaluated out-of-time, calibrated, thresholded on expected cost, "
        "and shipped as reproducible pipelines with drift monitoring.", 18, False, RGBColor(0xD6, 0xE4, 0xF0), False)]])
text(s, Inches(0.9), Inches(6.6), Inches(11), Inches(0.4),
     [[("credit_risk_modeling · reports/scenario_{a,b}_metrics.json · experiments/reports/", 13, False, RGBColor(0x9D, 0xC3, 0xE6), False)]])
notes(s, "Frame: results briefing, not a tutorial. Two models, one shared engine, honest limits.")

# ------------------------------------------------- 2 Business problem
s = new_slide("Context", "Lenders need calibrated default probabilities at two decision points — not just a ranking")
card(s, ML, CONTENT_TOP, Inches(5.9), Inches(2.3), "Decision 1 — Approve or reject (before funding)",
     [[("Every applicant needs a probability of default before money moves. A score that only ranks well "
        "is not enough: pricing, limits, and fair-lending review all need probabilities that mean what they say.", 15, False, MUTE, False)]])
card(s, ML + Inches(6.35), CONTENT_TOP, Inches(5.9), Inches(2.3), "Decision 2 — Monitor the funded book (after origination)",
     [[("Once funded, pricing and loan-structure fields exist. The book needs re-scoring as vintages age, "
        "with drift detection when the world changes.", 15, False, MUTE, False)]])
callout(s, ML, CONTENT_TOP + Inches(2.7), SW - ML - MR, Inches(0.9),
        [[("What this project delivers: ", 16, True, NAVY, False),
          ("two models answering those two questions, evaluated on genuinely unseen future loans, "
           "with every claim traceable to a committed report.", 16, False, INK, False)]])
src(s, "Situation")
notes(s, "Keep it plain: ranking vs probability. Calibration is the business-critical property.")

# ------------------------------------------------------- 3 Two models
s = new_slide("Architecture", "Scenario A answers “should we fund this applicant?” Scenario B answers “how risky is this funded loan?”")
card(s, ML, CONTENT_TOP, Inches(5.9), Inches(4.6), "Scenario A · application-time",
     [[("Approve / reject", 18, True, NAVY, False)],
      [("14 features — bureau, demographic, derived (credit age, DTI, LTI)", 15, False, INK, False)],
      [("Uses only what an applicant tells you and the bureau reports", 15, False, MUTE, False)],
      [("Hard-blocks funding, repayment, and collection fields", 15, False, MUTE, False)],
      [("OOT AUC 0.644   τ = 0.288", 15, True, BLUE, False)]])
card(s, ML + Inches(6.35), CONTENT_TOP, Inches(5.9), Inches(4.6), "Scenario B · post-origination",
     [[("Funded-book monitoring", 18, True, NAVY, False)],
      [("28 features — Scenario A set + grade, int_rate, installment, funded_amnt", 15, False, INK, False)],
      [("Scoring entry point rejects payloads containing repayment/collection fields", 15, False, MUTE, False)],
      [("Same shared engine (pipeline/core.py), separate feature contract", 15, False, MUTE, False)],
      [("OOT AUC 0.669   τ = 0.282", 15, True, BLUE, False)]])
src(s, "Source: reports/scenario_{a,b}_model_card.md · pipeline/scenario_{a,b}/features.py")
notes(s, "One engine, two contracts. Leakage prevention is enforced in code, not by convention.")

# ------------------------------------------------------------- 4 Data
s = new_slide("Data", "117,841 mature loans, labeled GOOD/BAD by a 33-month performance window and split forward in time")
kpis = [("466k", "raw loan records (2007–2014)"), ("117,841", "usable after maturity filter"),
        ("3", "temporal splits, never shuffled"), ("~15%", "bad (default) rate")]
kw = (SW - ML - MR - Inches(0.3) * 3) / 4
for i, (v, l) in enumerate(kpis):
    kpi(s, ML + i * (kw + Inches(0.1)), CONTENT_TOP, kw, Inches(1.15), v, l)
table(s, ML, CONTENT_TOP + Inches(1.5), SW - ML - MR,
      [["Split", "Period", "Loans", "Bad rate", "Role"],
       ["Train", "2007-08 → 2012-11", "84,764", "15.3%", "fit models"],
       ["Validation", "2012-11 → 2012-12", "10,596", "14.8%", "choose calibration & τ"],
       ["OOT test", "2013Q1", "22,481", "13.9%", "evaluated once"]],
      col_ws=[1, 2.4, 1, 1, 1.6], font=15, row_h=0.42,
      aligns=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.LEFT])
src(s, "Source: reports/scenario_a_model_card.md §Data & splits · src/performance_window.py (M=33)")
notes(s, "The OOT quarter is genuinely future data — the honest evaluation most projects skip.")

# ---------------------------------------------------------- 5 Pipeline
s = new_slide("What was built", "A four-stage pipeline turns raw loan tapes into scored, monitored probabilities of default")
stages = [("Raw tape", "466k loan records\nCSV → interim pickles"),
          ("prepare", "clean · performance\nwindow · temporal split"),
          ("train", "CatBoost / XGB / LGBM\ncalibration · τ on valid"),
          ("score", "batch scorer\nPD + decision + intervals"),
          ("monitor", "PSI drift report\nStable / Moderate / Significant")]
nw = (SW - ML - MR - Inches(0.4) * 4) / 5
x = ML
for i, (t, b) in enumerate(stages):
    hl = i == 2
    sp = box(s, x, CONTENT_TOP, nw, Inches(1.5), fill=SOFT if hl else WHITE,
             line=BLUE if hl else LINE, radius=True, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    tf = sp.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    _set(p.add_run(), t, 17, True, NAVY if hl else INK)
    for line in b.split("\n"):
        p = tf.add_paragraph(); p.alignment = PP_ALIGN.CENTER
        _set(p.add_run(), line, 12, False, MUTE)
    if i < 4:
        text(s, x + nw + Inches(0.02), CONTENT_TOP + Inches(0.5), Inches(0.38), Inches(0.5),
             [[("→", 24, False, FAINT, False)]], align=PP_ALIGN.CENTER)
    x += nw + Inches(0.4)
cards = [("Reproducible", "DVC-tracked data & models, manifest.json env+hashes, MLflow runs, 48-test suite."),
         ("Unified inference", "pipeline/infer.py routes one record through both scenarios."),
         ("Explainable", "pipeline/explain.py: SHAP importance, cost curves, LR coefficients.")]
cw = (SW - ML - MR - Inches(0.6) * 2) / 3
for i, (t, b) in enumerate(cards):
    card(s, ML + i * (cw + Inches(0.3)), CONTENT_TOP + Inches(2.1), cw, Inches(1.7), t,
         [[(b, 14, False, MUTE, False)]])
src(s, "Source: docs/method.md · pipeline/core.py · pipeline/{scenario_a,scenario_b}/")
notes(s, "Emphasize: one shared engine, scenario wrappers pass a spec — new scenarios extend, not copy.")

# ------------------------------------------------------------ 6 Leakage
s = new_slide("Trust", "Forbidden features are blocked in code at scoring time — not by analyst discipline")
card(s, ML, CONTENT_TOP, Inches(5.9), Inches(4.2), "What never enters a model",
     [[("Repayment outcomes: ", 15, True, INK, False), ("recoveries, out_prncp, total_pymnt, last_pymnt_*", 15, False, INK, False)],
      [("Collection fields: ", 15, True, INK, False), ("collection_recovery_fee, pymnt_plan", 15, False, INK, False)],
      [("Scenario A additionally excludes all funding/pricing fields", 15, False, MUTE, False)]])
card(s, ML + Inches(6.35), CONTENT_TOP, Inches(5.9), Inches(4.2), "How it is enforced",
     [[("FORBIDDEN_FEATURES lists live in each scenario's features.py", 15, False, INK, False)],
      [("Scenario B scorer hard-rejects payloads containing blocked columns", 15, False, INK, False)],
      [("pipeline/infer.py strips them per scenario when routing one record", 15, False, INK, False)],
      [("Calibration and τ are chosen on validation — the OOT test set is touched once", 15, False, INK, False)]])
callout(s, ML, CONTENT_TOP + Inches(4.5), SW - ML - MR, Inches(0.6),
        [[("This is the difference between a notebook that looks good and a model that survives production.", 15, True, NAVY, False)]], GOOD)
src(s, "Source: docs/method.md §The two scenarios · pipeline/scenario_b/score.py")
notes(s, "Stress: leakage is the #1 way credit models lie. Here it is structural.")

# ----------------------------------------------------- 7 Leaderboard
s = new_slide("Model selection", "CatBoost won both scenarios — and the alternatives were close enough that ensembles added nothing")
table(s, ML, CONTENT_TOP, SW - ML - MR,
      [["Model", "A · valid AUC", "A · OOT AUC", "B · valid AUC", "B · OOT AUC"],
       ["CatBoost", "0.6468", "0.6442", "0.6684", "0.6687"],
       ["XGBoost", "0.6442", "0.6433", "0.6666", "0.6674"],
       ["LightGBM", "0.6456", "0.6436", "0.6658", "0.6687"]],
      col_ws=[1.4, 1, 1, 1, 1], font=16, header_font=13, row_h=0.5,
      aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.RIGHT] * 4)
# bold CatBoost row manually
for c in range(5):
    cell = s.shapes[-1].table.cell(1, c)
    for p in cell.text_frame.paragraphs:
        for r in p.runs:
            r.font.bold = True; r.font.color.rgb = BLUE if c else NAVY
cards = [("Calibration: raw wins", "CatBoost probabilities needed no recalibration — sigmoid/isotonic tested on validation Brier, raw kept."),
         ("Threshold: expected cost", "τ minimizes expected loss (FN cost ≈ 8.25M vs FP ≈ 2.7M IDR), not accuracy."),
         ("Ensembles: tested, rejected", "Soft-vote and stacking evaluated — differences within noise, single CatBoost shipped.")]
cw = (SW - ML - MR - Inches(0.6) * 2) / 3
for i, (t, b) in enumerate(cards):
    card(s, ML + i * (cw + Inches(0.3)), CONTENT_TOP + Inches(2.6), cw, Inches(1.9), t,
         [[(b, 14, False, MUTE, False)]])
src(s, "Source: reports/scenario_{a,b}_leaderboard.csv · reports/scenario_{a,b}_metrics.json")
notes(s, "Key point: differences are ~0.002-0.003 AUC — honest reporting, no overselling.")

# ------------------------------------------- 8 Scenario A results
s = new_slide("Results · Scenario A", "The application-time model holds up out-of-time: AUC 0.644, Gini 0.288, well-calibrated")
table(s, ML, CONTENT_TOP, Inches(6.6),
      [["Metric", "Valid", "OOT (2013Q1)"],
       ["ROC AUC", "0.6468", "0.6442"], ["PR AUC", "0.2375", "0.2138"],
       ["Gini", "0.2936", "0.2885"], ["KS statistic", "0.2149", "0.2141"],
       ["Log loss", "0.4024", "0.3897"], ["Brier score", "0.1217", "0.1169"]],
      col_ws=[1.6, 1, 1], font=15, row_h=0.44,
      aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT])
rx = ML + Inches(7.1)
card(s, rx, CONTENT_TOP, SW - rx - MR, Inches(1.9), "Decision threshold τ = 0.2883",
     [[("Expected-cost optimum on validation. Approved-book bad rate at τ: 13.5% vs 14.8% overall.", 14, False, MUTE, False)]])
card(s, rx, CONTENT_TOP + Inches(2.2), SW - rx - MR, Inches(1.9), "Reproduces the notebook baseline",
     [[("Pipeline output matches nb4/autoresearch CatBoost to six decimals — the .py pipeline is verified "
        "against the research code.", 14, False, MUTE, False)]])
src(s, "Source: reports/scenario_a_metrics.json · reports/scenario_a_model_card.md")
notes(s, "Modest AUC is honest — application-time features are limited. Stability valid→OOT is the story.")

# ------------------------------------------- 9 Scenario B results
s = new_slide("Results · Scenario B", "Adding funding and pricing fields lifts OOT AUC to 0.669 — the strongest model in the project")
table(s, ML, CONTENT_TOP, Inches(6.6),
      [["Metric", "Valid", "OOT (2013Q1)"],
       ["ROC AUC", "0.6684", "0.6687"], ["PR AUC", "0.2587", "0.2364"],
       ["Gini", "0.3367", "0.3373"], ["KS statistic", "0.2527", "0.2483"],
       ["Log loss", "0.3968", "0.3841"], ["Brier score", "0.1202", "0.1155"]],
      col_ws=[1.6, 1, 1], font=15, row_h=0.44,
      aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT])
rx = ML + Inches(7.1)
card(s, rx, CONTENT_TOP, SW - rx - MR, Inches(1.9), "τ = 0.2823",
     [[("Expected-cost threshold, same methodology as Scenario A.", 14, False, MUTE, False)]])
callout(s, rx, CONTENT_TOP + Inches(2.2), SW - rx - MR, Inches(1.9),
        [[("OOT AUC higher than validation — the model generalizes, not degrades, on the future quarter.", 14, True, NAVY, False)]], GOOD)
src(s, "Source: reports/scenario_b_metrics.json · reports/scenario_b_model_card.md")
notes(s, "Grade/int_rate/installment carry signal — that's why B beats A. This is the monitoring model.")

# ------------------------------------------------- 10 Rolling OOT
s = new_slide("Robustness", "Rolling-origin evaluation shows gradual drift — a scheduled retraining cadence is sufficient")
pic(s, PRES_FIG / "rolling_oot_auc.png", ML, CONTENT_TOP, w=Inches(8.6))
bullets(s, ML + Inches(9.0), CONTENT_TOP + Inches(0.2), SW - ML - MR - Inches(9.0), Inches(4.8), [
    ("Method: ", "refit on expanding window, score the next quarter — 16 windows per scenario"),
    ("Expectation: ", "OOT AUC ~0.64–0.67 on future vintages; the 2010 peak is not recoverable"),
    ("Drift: ", "bad rates move with Lending Club's book growth — PSI monitoring is designed for this"),
    ("Cost: ", "full run < 5 min per scenario — cheap gate before promoting any retrained bundle"),
], size=15)
src(s, "Source: experiments/reports/rolling_oot_evaluation.md · rolling_oot_{a,b}.csv")
notes(s, "This answers 'is 0.644 a lucky quarter?' — no, it's the honest range across many quarters.")

# ------------------------------------------------------ 11 Calibration
s = new_slide("Calibration", "Raw CatBoost probabilities are already well-calibrated — extra recalibration was tested and rejected")
kpis = [("0.897 / 0.881", "calibration slope, valid / OOT (1.0 = perfect)"),
        ("0.013 / 0.019", "ECE, valid / OOT (expected calibration error)"),
        ("raw", "winning calibration — sigmoid & isotonic lost on Brier")]
kw = (SW - ML - MR - Inches(0.6) * 2) / 3
for i, (v, l) in enumerate(kpis):
    kpi(s, ML + i * (kw + Inches(0.3)), CONTENT_TOP, kw, Inches(1.7), v, l, vsize=30, lsize=12)
callout(s, ML, CONTENT_TOP + Inches(2.3), SW - ML - MR, Inches(2.1),
        [[("Per-vintage recalibration hurts. ", 17, True, NAVY, False),
          ("16 quarterly windows tested: refit calibrators on the adjusted tail worsened ECE and Brier — "
           "raw ECE ≈ 0.02 is already good. Recalibrate only at scheduled retraining, not continuously.", 17, False, INK, False)]])
src(s, "Source: experiments/reports/calibration_slope.{csv,md} · recalibration_drift_{a,b}.csv")
notes(s, "Business meaning: a predicted PD of 15% corresponds to ~15% observed defaults. That's what pricing needs.")

# -------------------------------------------------------- 12 Threshold
s = new_slide("Decision threshold", "τ is statistically stable (95% CI ±0.05) — but the real uncertainty is population bias, not sampling noise")
table(s, ML, CONTENT_TOP, Inches(7.2),
      [["Scenario", "τ deployed", "Bootstrap 95% CI", "Regret (mean)"],
       ["A", "0.288", "[0.244, 0.292]", "< 0.5% of cost"],
       ["B", "0.282", "[0.238, 0.302]", "< 0.5% of cost"]],
      col_ws=[1, 1, 1.8, 1.6], font=15, row_h=0.55,
      aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT])
rx = ML + Inches(7.7)
card(s, rx, CONTENT_TOP, SW - rx - MR, Inches(1.7), "Cost curve is flat near τ",
     [[("±0.05 deviation changes expected cost by ~2%; p90 regret ~1.2% of minimum.", 14, False, MUTE, False)]])
callout(s, rx, CONTENT_TOP + Inches(2.0), SW - rx - MR, Inches(2.2),
        [[("Governance implication: ", 15, True, WARN, False),
          ("τ needs policy control, not statistical control. The binding constraint is reject-inference bias "
           "(next slide) — an order of magnitude larger effect.", 15, False, INK, False)]], WARN)
src(s, "Source: experiments/reports/threshold_ci.md · threshold_ci.csv (500 bootstrap resamples)")
notes(s, "500 bootstraps. Deployed τ sits inside the CI — keep it, revisit at retraining cadence.")

# ------------------------------------------------- 13 Reject inference
s = new_slide("Known bias · selection", "Trained only on funded loans, the model likely overstates applicant quality — bounded, not hidden")
kpis = [("1.4–1.9×", "plausible true applicant-pool PD vs funded-book PD"),
        ("0.12–0.18", "oracle τ under plausible assumptions vs deployed 0.288"),
        ("none", "reject labels exist — strict reject inference is impossible")]
kw = (SW - ML - MR - Inches(0.6) * 2) / 3
for i, (v, l) in enumerate(kpis):
    kpi(s, ML + i * (kw + Inches(0.3)), CONTENT_TOP, kw, Inches(1.9), v, l, vsize=28, lsize=12)
callout(s, ML, CONTENT_TOP + Inches(2.4), SW - ML - MR, Inches(2.2),
        [[("What this means: ", 16, True, BAD, False),
          ("absolute PDs are conditional on funding. A predicted 15% means “15% of similar funded loans default” — "
           "the true applicant pool defaults more often. Direction and rough magnitude are quantified; exact "
           "calibration needs reject-level data.", 16, False, INK, False)]], BAD)
src(s, "Source: experiments/reports/reject_inference.md · reject_inference_results.csv")
notes(s, "Most important caveat in the deck. Sensitivity analysis, not an estimate — say it plainly.")

# --------------------------------------------------------- 14 Fairness
s = new_slide("Fairness", "Adverse-impact screening flagged 60-month terms, small_business, and top-LTI borrowers — income is clean")
table(s, ML, CONTENT_TOP, SW - ML - MR,
      [["Screen", "Flag", "Detail", "Action"],
       ["Adverse impact (AIR < 0.80)", "60-mo term · small_business · top LTI tercile", "Approval-rate disparity vs reference", "Monitor; document"],
       ["Calibration parity", "small_business under-predicts ~5pp", "Predicted PD too low vs observed", "Segment review"],
       ["Calibration parity", "60-mo term over-predicts ~6pp OOT", "Predicted PD too high vs observed", "Segment review"],
       ["Income", "—", "No adverse impact detected", "None"]],
      col_ws=[1.5, 2.2, 2, 1.2], font=14, header_font=12, row_h=0.62,
      aligns=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT])
text(s, ML, CONTENT_TOP + Inches(3.6), SW - ML - MR, Inches(0.8),
     [[("AIR tests whether the decision treats segments equally; calibration parity tests whether a predicted PD "
        "means the same thing inside every segment. 73 segment rows across 9 groupings, valid + OOT.", 14, False, MUTE, True)]])
src(s, "Source: experiments/reports/adverse_impact.csv · calibration_parity.{csv,md}")
notes(s, "Both tests, not just one. Flags are monitoring items, not proven discrimination — be precise.")

# ------------------------------------------------------------ 15 SHAP
s = new_slide("Explainability", "SHAP explanations work at portfolio and individual-loan level — decline reasons are defensible")
pic(s, FIG / "shap_summary_global.png", ML, CONTENT_TOP, w=Inches(8.0))
bullets(s, ML + Inches(8.5), CONTENT_TOP + Inches(0.3), SW - ML - MR - Inches(8.5), Inches(4.8), [
    ("Global: ", "feature importance for the whole book"),
    ("Per-loan: ", "waterfall plots give each applicant a reason list"),
    ("Dependence: ", "how revol_util, LTI etc. move PD"),
    ("Interpretability track preserved: ", "WoE logistic-regression scorecard (nb3) remains for adverse-action explanations"),
], size=15)
src(s, "Source: reports/figures/shap_*.png · src/shap_interpretability.py · pipeline/explain.py")
notes(s, "Point at the figure: high revol_util and delinquencies push PD up; thin-file and high-DTI too.")

# --------------------------------------------------------- 16 Conformal
s = new_slide("Uncertainty", "Venn-Abers conformal intervals give each loan a defensible PD range for risk committees")
card(s, ML, CONTENT_TOP, Inches(5.9), Inches(4.4), "How it works",
     [[("Isotonic calibration fit twice on the 10,596-loan validation set — label forced 0, then 1", 15, False, INK, False)],
      [("Interval [min(p0,p1), max(p0,p1)] brackets a calibrated probability per loan", 15, False, INK, False)],
      [("Wired into scoring: --intervals on score.py and infer.py", 15, False, INK, False)],
      [("Distribution-free coverage guarantees under exchangeability", 15, False, INK, False)]])
card(s, ML + Inches(6.35), CONTENT_TOP, Inches(5.9), Inches(4.4), "Why it matters",
     [[("Near-threshold loans get visibly wider intervals → manual review queue", 15, False, INK, False)],
      [("Risk committees see uncertainty, not false precision", 15, False, INK, False)],
      [("Complements the bootstrap τ analysis: score-level and threshold-level uncertainty both quantified", 15, False, INK, False)]])
src(s, "Source: experiments/reports/conformal_coverage.md · src/conformal.py")
notes(s, "Novel-ish capability most credit shops don't have. Brief — technical audience gets it.")

# ------------------------------------------------------------- 17 PSI
s = new_slide("Monitoring", "Every scoring batch emits a PSI drift report — per-feature and score-distribution, with a verdict")
card(s, ML, CONTENT_TOP, Inches(5.9), Inches(4.4), "What is monitored",
     [[("Characteristic PSI per feature vs reference distribution", 15, False, INK, False)],
      [("Score-distribution PSI — is the book's risk mix shifting?", 15, False, INK, False)],
      [("Overall verdict: Stable / Moderate / Significant", 15, False, INK, False)],
      [("Natural-sort bin ordering (audit fix) so drift localizes correctly", 15, False, INK, False)]])
card(s, ML + Inches(6.35), CONTENT_TOP, Inches(5.9), Inches(4.4), "Why it matters now",
     [[("Rolling evaluation proved drift is gradual — exactly what PSI catches early", 15, False, INK, False)],
      [("Triggers the scheduled retraining cadence before AUC degrades visibly", 15, False, INK, False)],
      [("Regression test: shifted-feature fixture produces material PSI + non-stable verdict", 15, False, INK, False)]])
src(s, "Source: src/population_stability_index.py · pipeline/scenario_{a,b}/score.py --psi · docs/method.md")
notes(s, "Monitoring is a feature, not an afterthought — it closes the loop with the drift finding.")

# ---------------------------------------------------- 18 Reproducibility
s = new_slide("Engineering", "DVC-tracked artifacts, a manifest of environment and hashes, MLflow runs, and a 48-test suite")
cards = [("DVC", "Data, interim frames, and model bundles versioned; local remote pushed."),
         ("manifest.json", "Environment versions, split config, artifact hashes — any bundle is auditable."),
         ("MLflow", "Experiment tracking with sqlite backend; runs comparable across retrains."),
         ("48 tests", "pytest green: pipelines, metrics, PSI, conformal, reject-inference, threshold math.")]
cw = (SW - ML - MR - Inches(0.45) * 3) / 4
for i, (t, b) in enumerate(cards):
    card(s, ML + i * (cw + Inches(0.15)), CONTENT_TOP, cw, Inches(1.8), t,
         [[(b, 13, False, MUTE, False)]])
callout(s, ML, CONTENT_TOP + Inches(2.3), SW - ML - MR, Inches(1.5),
        [[("Scenario A pipeline reproduces the research notebook's CatBoost to six decimals — "
           "production code and research code are provably the same model.", 16, True, NAVY, False)]], GOOD)
src(s, "Source: README.md · docs/reproduce.md · artifacts/manifest.json · tests/")
notes(s, "For the technical half of the audience: this is what 'production-ready' means concretely.")

# ------------------------------------------------------------ 19 Hazard
s = new_slide("v2 candidate", "The survival model produces PD-as-a-function-of-time — valuable, but it trails on discrimination")
pic(s, PRES_FIG / "hazard_pd_t.png", ML, CONTENT_TOP, w=Inches(8.4))
bullets(s, ML + Inches(8.9), CONTENT_TOP + Inches(0.3), SW - ML - MR - Inches(8.9), Inches(4.8), [
    ("Model: ", "discrete-time hazard h(t|x) → survival curve → PD(t) per loan"),
    ("Performance: ", "OOT AUC 0.637 vs CatBoost 0.644 — documented, not promoted"),
    ("Outputs: ", "expected-loss term structure peaks around month 11; median time-to-event 17mo"),
    ("Use case: ", "right tool if the business needs lifetime EL, cross-sell windows, or horizon-specific triggers"),
], size=15)
src(s, "Source: experiments/reports/hazard_model.md · hazard_pd_curves.md · hazard_pd_curves_hgb.csv")
notes(s, "Show the curve: PD at 6/12/24/33 months per decile. This is what binary models cannot express.")

# ------------------------------------------------------- 20 Limitations
s = new_slide("Honesty", "Five known limitations — each with a mitigation already in place or scoped")
lims = [("Single OOT quarter (2013Q1)", "Mitigated by rolling-origin evaluation across 16 quarters."),
        ("Funded-loan selection bias", "Bounded by sensitivity analysis; needs reject-level data to resolve."),
        ("Modest application-time AUC (~0.64)", "Ceiling of pre-funding features; Scenario B reaches 0.669 with pricing data."),
        ("Vintage drift", "Gradual, monitored by PSI, handled by retraining cadence."),
        ("Synthetic-data tests", "Unit tests use fixtures; no live-portfolio shadow validation yet."),
        ("Segment exceptions", "small_business under-prediction and 60-mo over-prediction flagged for review.")]
cw = (SW - ML - MR - Inches(0.4) * 2) / 3
ch = Inches(1.75)
for i, (t, b) in enumerate(lims):
    x = ML + (i % 3) * (cw + Inches(0.2))
    y = CONTENT_TOP + (i // 3) * (ch + Inches(0.25))
    card(s, x, y, cw, ch, t, [[(b, 13, False, MUTE, False)]])
notes(s, "Naming limits builds credibility. Each has a next step — see next-steps slide.")

# ------------------------------------------------- 21 Rejected paths
s = new_slide("Negative results", "Class imbalance fixes, alternative representations, and ensembles were all tried — none beat raw CatBoost")
table(s, ML, CONTENT_TOP, SW - ML - MR,
      [["Experiment", "Result", "Decision"],
       ["SMOTENC oversampling", "AUC −0.006 to −0.020, monotonic harm", "Rejected"],
       ["Random undersampling", "RUS-30 wrecks log loss; RUS-20 +0.0003 (noise)", "Rejected"],
       ["Count-feature re-expression", "+0.0018 valid AUC — within noise margin", "Discarded"],
       ["mths_since_last_* + LR interactions", "+0.0014 valid, flat OOT; interactions nil", "Rejected"],
       ["Soft-vote / stacking ensembles", "Differences within noise", "Single CatBoost"],
       ["Per-vintage recalibration", "Worsens ECE and Brier", "Recal only at retrain"]],
      col_ws=[2.2, 2.6, 1.4], font=14, header_font=12, row_h=0.55,
      aligns=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT])
text(s, ML, CONTENT_TOP + Inches(4.5), SW - ML - MR, Inches(0.5),
     [[("Autoresearch loop: ~15s per trial, protocol and keep/discard rule committed in autoresearch/program.md.", 14, False, MUTE, True)]])
src(s, "Source: experiments/reports/class_imbalance_sampling.md · feature_representation.md · recalibration_drift.md · autoresearch/results/trials.jsonl")
notes(s, "Negative results are a feature: the champion survived real competition, documented in trials.jsonl.")

# ------------------------------------------------------ 22 Recommendation
s = new_slide("Recommendation", "Deploy Scenario A for approve/reject decisions and Scenario B for funded-book monitoring — as a pilot")
card(s, ML, CONTENT_TOP, Inches(5.9), Inches(4.4), "Scenario A → origination",
     [[("Score every applicant; τ = 0.288 decision boundary", 15, False, INK, False)],
      [("SHAP waterfall per decline → adverse-action reasons", 15, False, INK, False)],
      [("Venn-Abers interval flags borderline cases for review", 15, False, INK, False)],
      [("Report PDs with the documented funding-conditional caveat", 15, False, INK, False)]], tcolor=BLUE)
card(s, ML + Inches(6.35), CONTENT_TOP, Inches(5.9), Inches(4.4), "Scenario B → portfolio management",
     [[("Re-score funded book on each reporting cycle", 15, False, INK, False)],
      [("PSI report per batch → retraining trigger", 15, False, INK, False)],
      [("Segment dashboards watch the three flagged groups", 15, False, INK, False)],
      [("Hazard model available for horizon-specific EL questions", 15, False, INK, False)]], tcolor=BLUE)
callout(s, ML, CONTENT_TOP + Inches(4.7), SW - ML - MR, Inches(0.8),
        [[("Both bundles reproduce with python -m pipeline.scenario_{a,b}.train; unified scoring via pipeline/infer.py.", 15, True, NAVY, False)]], GOOD)
notes(s, "Concrete deployment shape. Pilot, not big-bang — shadow mode alongside existing process first.")

# ---------------------------------------------------------- 23 Decisions
s = new_slide("Asks", "Stakeholders must decide: how to handle reject-data bias, τ governance, and retraining cadence")
asks = [("1 · Reject-level data",
         "Accept funded-conditional PDs with documented bounds (1.4–1.9×, oracle τ 0.12–0.18), or invest in "
         "collecting/scoring rejected applicants to close the gap."),
        ("2 · τ governance",
         "Approve the expected-cost τ policy (C_FN ≈ 8.25M vs C_FP ≈ 2.7M IDR) and its review cadence — "
         "statistical uncertainty is small; policy risk is not."),
        ("3 · Retraining cadence",
         "Approve scheduled retraining (~semi-annual, PSI-triggered early) — rolling evaluation shows "
         "gradual drift, not abrupt breaks.")]
y = CONTENT_TOP
for t, b in asks:
    card(s, ML, y, SW - ML - MR, Inches(1.5), t, [[(b, 15, False, MUTE, False)]])
    y += Inches(1.75)
notes(s, "Each ask maps to a documented finding. Don't ask for blanket approval — ask for three decisions.")

# --------------------------------------------------------- 24 Next steps
s = new_slide("Next steps", "Four scoped next steps: reject data, live shadow validation, dashboard, retraining automation")
steps = [("1 · Reject-level data collection", "Highest-value gap: turns bounded bias into measured bias. Reject-inference module already accepts augmented files."),
         ("2 · Shadow-mode pilot", "Score live applicants without acting; compare model PDs to realized outcomes — the missing live-portfolio validation."),
         ("3 · Streamlit decision dashboard", "Scaffold exists; wire to pipeline/infer.py + SHAP for a business-facing UI."),
         ("4 · Retraining automation", "Rolling-OOT harness is the promotion gate; PSI triggers; MLflow tracks each retrain.")]
cw = (SW - ML - MR - Inches(0.45) * 2) / 2
for i, (t, b) in enumerate(steps):
    x = ML + (i % 2) * (cw + Inches(0.3))
    y = CONTENT_TOP + (i // 2) * (Inches(1.85) + Inches(0.25))
    card(s, x, y, cw, Inches(1.85), t, [[(b, 14, False, MUTE, False)]])
src(s, "Source: README.md §Future directions · dashboard/app.py · autoresearch/program.md")
notes(s, "Ordered by value: reject data first — everything else is engineering.")

# --------------------------------------------------------- 25 Conclusions
s = prs.slides.add_slide(BLANK); N[0] += 1
box(s, 0, 0, SW, SH, fill=NAVY)
text(s, Inches(0.9), Inches(0.9), Inches(11), Inches(0.4),
     [[("CONCLUSIONS", 14, True, RGBColor(0x9D, 0xC3, 0xE6), False)]])
text(s, Inches(0.9), Inches(1.5), Inches(11.5), Inches(1.6),
     [[("Two validated models.", 46, True, WHITE, False)],
      [("Honest about their limits. Ready for pilot.", 46, True, WHITE, False)]])
concl = [("Scenario A", "AUC 0.644 OOT · τ 0.288 · approve/reject"),
         ("Scenario B", "AUC 0.669 OOT · τ 0.282 · book monitoring"),
         ("Trust layer", "Calibration · SHAP · conformal · PSI · fairness · reproducibility")]
cw = (SW - Inches(1.8) - Inches(0.6) * 2) / 3
for i, (t, b) in enumerate(concl):
    x = Inches(0.9) + i * (cw + Inches(0.3))
    sp = box(s, x, Inches(4.0), cw, Inches(1.5), fill=RGBColor(0x24, 0x5A, 0x8C), line=None, radius=True, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    tf = sp.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; _set(p.add_run(), t, 18, True, WHITE)
    p = tf.add_paragraph(); _set(p.add_run(), b, 13, False, RGBColor(0xC9, 0xD9, 0xE8))
text(s, Inches(0.9), Inches(6.1), Inches(11), Inches(0.4),
     [[("Every number traces to a committed report or test · questions welcome", 15, False, RGBColor(0x9D, 0xC3, 0xE6), False)]])
notes(s, "This slide stays on screen during Q&A. If asked for depth: appendix has full metrics + repro commands.")

# ------------------------------------------------------ 26 Appendix divider
s = prs.slides.add_slide(BLANK); N[0] += 1
box(s, 0, 0, SW, SH, fill=SOFT)
text(s, Inches(0.9), Inches(2.6), Inches(11), Inches(0.4),
     [[("APPENDIX", 14, True, BLUE, False)]])
text(s, Inches(0.9), Inches(3.1), Inches(11.5), Inches(1.4),
     [[("Details & reproduction", 54, True, NAVY, False)]])
text(s, Inches(0.9), Inches(4.7), Inches(11), Inches(0.8),
     [[("Full metrics tables, experiment log, and exact commands to rebuild every artifact.", 20, False, MUTE, False)]])
notes(s, "Skip unless asked. Everything here is committed in the repo.")

# ------------------------------------------------- 27 Appendix metrics
s = new_slide("Appendix · metrics", "Full OOT metrics for both production bundles")
rx = ML + Inches(6.9)
table(s, ML, CONTENT_TOP, Inches(6.2),
      [["Scenario A — CatBoost · raw · τ=0.2883", "Valid", "OOT"],
       ["ROC AUC", "0.646804", "0.644239"], ["PR AUC", "0.237474", "0.213800"],
       ["Log loss", "0.402443", "0.389742"], ["Brier", "0.121717", "0.116929"],
       ["Gini", "0.293607", "0.288478"], ["KS", "0.214866", "0.214102"]],
      col_ws=[2, 1, 1], font=13, header_font=12, row_h=0.4,
      aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT])
table(s, rx, CONTENT_TOP, SW - rx - MR,
      [["Scenario B — CatBoost · raw · τ=0.2823", "Valid", "OOT"],
       ["ROC AUC", "0.668363", "0.668670"], ["PR AUC", "0.258677", "0.236444"],
       ["Log loss", "0.396757", "0.384100"], ["Brier", "0.120207", "0.115477"],
       ["Gini", "0.336726", "0.337340"], ["KS", "0.252660", "0.248295"]],
      col_ws=[2, 1, 1], font=13, header_font=12, row_h=0.4,
      aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT])
src(s, "Source: reports/scenario_{a,b}_metrics.json — full precision values")
notes(s, "Exact decimals for the technical audience.")

# ------------------------------------------------ 28 Appendix reproduce
s = new_slide("Appendix · reproduce", "Every artifact rebuilds from committed code and DVC-tracked data")
cmds = ("# pipelines\npython -m pipeline.scenario_a.train\npython -m pipeline.scenario_b.train\n\n"
        "# scoring + drift\npython -m pipeline.scenario_a.score batch.csv -o scores.csv --psi --intervals\n\n"
        "# unified inference\npython -m pipeline.infer input.csv -o out.csv\n\n"
        "# experiments (~15s/trial)\npython autoresearch/experiment.py configs/cat_raw.json")
box(s, ML, CONTENT_TOP, Inches(6.6), Inches(4.7), fill=RGBColor(0x24, 0x2B, 0x33), line=None, radius=True, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
tb = s.shapes[-1].text_frame; tb.word_wrap = True
for i, line in enumerate(cmds.split("\n")):
    p = tb.paragraphs[0] if i == 0 else tb.add_paragraph()
    _set(p.add_run(), line if line else " ", 12, False,
         RGBColor(0x9F, 0xD3, 0xFF) if not line.startswith("#") else RGBColor(0x6B, 0x7A, 0x8D))
rx = ML + Inches(7.1)
card(s, rx, CONTENT_TOP, SW - rx - MR, Inches(1.5), "Environment",
     [[("conda env credit-risk_VENV, Python 3.13 · environment.yml · docs/reproduce.md", 13, False, MUTE, False)]])
card(s, rx, CONTENT_TOP + Inches(1.7), SW - rx - MR, Inches(1.5), "Reports index",
     [[("reports/scenario_*_{metrics.json,leaderboard.csv,model_card.md} · experiments/reports/*.md + *.csv", 13, False, MUTE, False)]])
card(s, rx, CONTENT_TOP + Inches(3.4), SW - rx - MR, Inches(1.3), "Figures",
     [[("reports/figures/ (SHAP, KS, cost curves) · presentations/figures/ (rolling OOT, hazard PD(t))", 13, False, MUTE, False)]])
notes(s, "Leave this up if they want to verify anything live.")

prs.save(OUT)
print(f"wrote {OUT} — {N[0]} slides")
