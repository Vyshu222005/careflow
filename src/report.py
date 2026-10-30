"""PDF report generator for the selected patients.

`collect_report_data(ctx, ...)` gathers everything a report needs from a dashboard context; `build_pdf(data)` turns it
into PDF bytes. The two steps are separate so the PDF code can be tested with plain tables.

Text is kept to characters the built-in PDF fonts can draw (no arrows or emoji); `ascii_text` converts them.
"""
import io
from dataclasses import dataclass, field
from datetime import datetime
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402

import bundle as bundle_module  # noqa: E402
import conformance as conformance_module  # noqa: E402
import handover as handover_module  # noqa: E402
import insights as insights_module  # noqa: E402
import rework as rework_module  # noqa: E402
from bottlenecks import clinical_steps, er_waits, format_hours, transition_stats  # noqa: E402
from load_data import ACTIVITY, CASE_ID  # noqa: E402
from variants import shorten, variant_table  # noqa: E402

BLUE = "#2a78d6"
NAVY = colors.HexColor("#0f2d52")
MUTED = colors.HexColor("#667085")
LINE = colors.HexColor("#d0d5dd")
SOFT = colors.HexColor("#eaf2fd")
LIMITATIONS = [
    "The data is a public research dataset from one hospital, with timestamps entered retrospectively. Results show where delays happen, not why.",
    "Times to antibiotics are measured from ER registration, because the log has no time of sepsis recognition. The true share treated in time is probably higher.",
    "The expected pathway used for conformance checking and the care-bundle scorecard are adapted for this project. They are not official clinical measures.",
    "Department codes are anonymised; roles such as Laboratory or Ward are inferred from the steps each department records.",
    "Model results and what-if scenarios are for research demonstration. They are not clinical advice and not evidence of real-world benefit.",
]


def ascii_text(text) -> str:
    """Replace characters the built-in PDF fonts cannot draw."""
    replacements = {"→": " to ", "←": " from ", "–": "-", "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"', "≥": ">=", "≤": "<=", "×": "x", "·": "-", "…": "..."}
    out = str(text)
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def para(text, style) -> Paragraph:
    return Paragraph(escape(ascii_text(text)).replace("\n", "<br/>"), style)


@dataclass
class ReportData:
    title: str
    filters_text: str
    counts_text: str
    kpis: list  # [(label, value, note)]
    insights: list = field(default_factory=list)
    pathways: pd.DataFrame | None = None
    waits: pd.DataFrame | None = None
    scorecard: pd.DataFrame | None = None
    deviations: pd.DataFrame | None = None
    rework_lines: list = field(default_factory=list)
    handover_pairs: pd.DataFrame | None = None
    model: dict | None = None  # stage_label, verdict, best_model, metrics (DataFrame), importance (DataFrame), base_rate
    simulation: dict | None = None  # scenario_text, summary (DataFrame), newly_within_one_hour, patients
    limitations: list = field(default_factory=lambda: list(LIMITATIONS))


# ---------------------------------------------------------------- data collection
def collect_report_data(ctx, metrics: dict, insight_list: list, model: dict | None = None, simulation: dict | None = None) -> ReportData:
    ids = set(ctx.cases[CASE_ID])
    cases = ctx.cases
    antibiotics = ctx.antibiotics
    kpis = [
        ("Patients", f"{len(cases):,}", f"of {len(ctx.cases_all):,}"),
        ("Cleaned events", f"{ctx.cleaned_events_selected:,}", "recorded steps"),
        ("Median journey", f"{cases['duration_days'].median():.1f} days", "first to last event"),
        ("Median time to antibiotics", format_hours(antibiotics["median_hours"]) if ctx.has_abx else "n/a", "from registration"),
        ("Antibiotics within 1 hour", f"{antibiotics['within_1h_pct']:.0f}%" if ctx.has_abx else "n/a", "of treated patients"),
        ("Intensive care", f"{cases['reached_icu'].mean() * 100:.0f}%", "of patients"),
        ("Returned to the ER", f"{cases['returned_to_er'].mean() * 100:.0f}%", "of patients"),
        ("No discharge recorded", f"{metrics['missing_discharge_pct']:.0f}%", "of patients"),
        ("Follow expected pathway", f"{metrics['conformance_pct']:.0f}%", f"average fitness {metrics['avg_fitness']:.2f}"),
        ("Complete care bundle", f"{metrics['bundle_complete_bundle_pct']:.0f}%", "all applicable elements"),
    ]
    table = variant_table(ctx.cases, "milestone_variant").head(8)
    pathways = pd.DataFrame({"Pathway": table["variant"].map(shorten), "Patients": table["cases"], "Share %": table["share_pct"].round(1),
                             "Median days": table["median_duration_days"].round(1)})
    waits = er_waits(transition_stats(clinical_steps(ctx.events)), 30).head(8)
    waits = pd.DataFrame({"Step": [f"{a} to {b}" for a, b in zip(waits["from_step"], waits["to_step"])], "Patients": waits["cases"],
                          "Median wait": waits["median_hours"].map(format_hours), "90th percentile": waits["p90_hours"].map(format_hours),
                          "median_hours": waits["median_hours"]})
    bt = ctx.bundle_all[ctx.bundle_all[CASE_ID].isin(ids)]
    conf = ctx.conformance_all[ctx.conformance_all[CASE_ID].isin(ids)]
    log = ctx.clean_log[ctx.clean_log[CASE_ID].isin(ids)]
    departments = handover_module.department_table(log)
    pairs = handover_module.handover_pairs(handover_module.handover_events(log), departments.set_index("department")["role"]).head(6)
    rework_lines = [
        f"{metrics['repeated_ward_patients']:,} patients ({metrics['repeated_ward_pct']:.0f}%) have more than one ward admission record; "
        f"{metrics['ward_transfer_share_pct']:.0f}% of the repeated records are entered by a different unit than the previous one, which looks like transfers.",
        f"{metrics['returned']:,} of {metrics['released']:,} discharged patients return to the ER; {metrics['return_30d']:,} ({metrics['return_30d_pct']:.0f}%) within 30 days.",
        f"{len(departments)} departments recorded events; {int((departments['role'] == 'Ward').sum())} are ward units.",
    ]
    counts = ctx.counts
    return ReportData(
        title="CareFlow 2.0: Clinical Pathway Report", filters_text=ctx.filters_text, counts_text=counts.explanation, kpis=kpis, insights=insight_list,
        pathways=pathways, waits=waits, scorecard=bundle_module.scorecard(bt),
        deviations=conformance_module.deviation_summary(conf).query("kind == 'rule'"), rework_lines=rework_lines, handover_pairs=pairs, model=model, simulation=simulation,
    )


# ---------------------------------------------------------------- charts
def _png(fig) -> io.BytesIO:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return buffer


def _bar_chart(labels, values, title, xlabel, value_format="{:.0f}", color=BLUE, height=None) -> io.BytesIO:
    labels = [ascii_text(label) for label in labels][::-1]
    values = list(values)[::-1]
    fig, ax = plt.subplots(figsize=(7.2, height or max(1.8, 0.42 * len(labels) + 0.9)))
    ax.barh(labels, values, color=color, height=0.62)
    span = max(values) if values else 1
    for index, value in enumerate(values):
        ax.text(value + span * 0.012, index, value_format.format(value), va="center", fontsize=8, color="#475467")
    ax.set_xlim(0, span * 1.18 if span else 1)
    ax.set_title(ascii_text(title), loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel(ascii_text(xlabel), fontsize=8)
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#e4e7ec", linewidth=0.7)
    ax.set_axisbelow(True)
    return _png(fig)


# ---------------------------------------------------------------- PDF
def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=22, textColor=NAVY, alignment=TA_LEFT, spaceAfter=4),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], fontSize=10, textColor=MUTED, spaceAfter=2),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=14, textColor=NAVY, spaceBefore=12, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11, textColor=NAVY, spaceBefore=6, spaceAfter=3),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontSize=9.5, leading=13, spaceAfter=4),
        "small": ParagraphStyle("small", parent=base["BodyText"], fontSize=8, leading=10.5, textColor=MUTED),
        "cell": ParagraphStyle("cell", parent=base["BodyText"], fontSize=8.5, leading=10.5),
        "bullet": ParagraphStyle("bullet", parent=base["BodyText"], fontSize=9.5, leading=13, leftIndent=12, bulletIndent=2, spaceAfter=2),
    }


def _table(frame: pd.DataFrame, styles: dict, col_widths=None, align_right=()) -> Table:
    head = [para(c, ParagraphStyle("th", parent=styles["cell"], fontName="Helvetica-Bold", textColor=colors.white)) for c in frame.columns]
    rows = [head] + [[para(v if isinstance(v, str) else f"{v:,.1f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else v, styles["cell"]) for v in row]
                     for row in frame.itertuples(index=False)]
    table = Table(rows, colWidths=col_widths, repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("GRID", (0, 0), (-1, -1), 0.4, LINE), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fb")]),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    table.setStyle(TableStyle(style))
    return table


def _kpi_grid(kpis: list, styles: dict) -> Table:
    cells = []
    for label, value, note in kpis:
        cells.append([para(label.upper(), ParagraphStyle("kl", parent=styles["small"], fontSize=7)),
                      para(value, ParagraphStyle("kv", parent=styles["body"], fontName="Helvetica-Bold", fontSize=15, textColor=NAVY, leading=17)),
                      para(note, styles["small"])])
    per_row = 5
    rows = [cells[i:i + per_row] for i in range(0, len(cells), per_row)]
    for row in rows:
        row += [""] * (per_row - len(row))  # empty cells to complete the last row
    grid = Table(rows, colWidths=[3.45 * cm] * per_row)
    grid.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.4, LINE), ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE), ("BACKGROUND", (0, 0), (-1, -1), SOFT),
                              ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return grid


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.2 * cm, "CareFlow 2.0 - research and education use only; not clinical advice")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(data: ReportData) -> bytes:
    """Render the report. Returns the PDF as bytes."""
    styles = _styles()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=2 * cm,
                            title=ascii_text(data.title), author="CareFlow 2.0")
    story = [para(data.title, styles["title"]),
             para(f"Generated {datetime.now():%d %b %Y %H:%M}. Patients: {data.filters_text}.", styles["subtitle"]),
             para(data.counts_text, styles["small"]), Spacer(1, 8), para("Key figures", styles["h1"]), _kpi_grid(data.kpis, styles)]

    if data.insights:
        story += [para("Key findings and recommendations", styles["h1"]),
                  para("Each finding comes from a named rule with a visible threshold. Suggestions are about reviewing the process and are not clinical advice.", styles["small"])]
        for item in data.insights:
            block = [para(f"{item.severity.upper()} - {item.title}", styles["h2"]), para(item.finding, styles["body"]),
                     para(f"Rule: {item.rule}", styles["small"]), para(f"What to review: {item.action}", styles["small"]), Spacer(1, 4)]
            story.append(KeepTogether(block))

    story.append(PageBreak())
    story.append(para("Patient pathways", styles["h1"]))
    if data.pathways is not None and len(data.pathways):
        story += [para("The most common clinical pathways (repeated lab tests are left out so the main route stays visible).", styles["body"]),
                  _table(data.pathways, styles, col_widths=[9.6 * cm, 1.9 * cm, 1.9 * cm, 2.2 * cm])]
    story.append(para("Bottleneck analysis", styles["h1"]))
    if data.waits is not None and len(data.waits):
        story += [Image(_bar_chart(list(data.waits["Step"]), list(data.waits["median_hours"]), "Median wait between ER steps", "Hours", "{:.1f} h"), width=16 * cm,
                        height=16 * cm * (max(1.8, 0.42 * len(data.waits) + 0.9) / 7.2)),
                  _table(data.waits.drop(columns="median_hours"), styles, col_widths=[8 * cm, 2.2 * cm, 3 * cm, 3 * cm])]

    story.append(PageBreak())
    story.append(para("Quality and conformance", styles["h1"]))
    if data.scorecard is not None and len(data.scorecard):
        card = data.scorecard.dropna(subset=["met_pct"])
        story.append(Image(_bar_chart(list(card["label"]), list(card["met_pct"]), "Care-bundle scorecard (adapted, not an official measure)", "Share of patients the element applies to (%)", "{:.0f}%"),
                           width=16 * cm, height=16 * cm * (max(1.8, 0.42 * len(card) + 0.9) / 7.2)))
    if data.deviations is not None and len(data.deviations):
        dev = data.deviations
        story.append(Image(_bar_chart(list(dev["deviation"]), list(dev["share_pct"]), "Deviations from the expected pathway", "Share of patients (%)", "{:.0f}%"),
                           width=16 * cm, height=16 * cm * (max(1.8, 0.42 * len(dev) + 0.9) / 7.2)))

    story.append(para("Rework, returns and department flow", styles["h1"]))
    story += [Paragraph("&bull; " + escape(ascii_text(line)), styles["bullet"]) for line in data.rework_lines]
    if data.handover_pairs is not None and len(data.handover_pairs):
        pairs = data.handover_pairs
        view = pd.DataFrame({"From": pairs["from_department"], "To": pairs["to_department"], "Handovers": pairs["handovers"],
                             "Median wait": pairs["median_wait_hours"].map(format_hours)})
        story += [Spacer(1, 4), para("Most frequent handovers between roles", styles["h2"]), _table(view, styles, col_widths=[4.6 * cm, 4.6 * cm, 3 * cm, 3.4 * cm])]

    if data.model:
        m = data.model
        story += [PageBreak(), para("Predictive model summary", styles["h1"]),
                  para(f"Question: will antibiotics be given more than 1 hour after registration? Prediction point: {m['stage_label']}. Best model: {m['best_model']}. "
                       f"Verdict: {m['verdict']}.", styles["body"]),
                  para(f"{m['base_rate'] * 100:.0f}% of the modelled patients were delayed. Metrics below are from 5-fold patient-level cross-validation; the first row is a do-nothing baseline.", styles["small"]),
                  _table(m["metrics"].round(3), styles)]
        if m.get("importance") is not None:
            top = m["importance"].head(8)
            story.append(Image(_bar_chart(list(top["label"]), list(top["share_pct"]), "What drives the model (SHAP)", "Share of the model's total effect (%)", "{:.0f}%"),
                               width=16 * cm, height=16 * cm * (max(1.8, 0.42 * len(top) + 0.9) / 7.2)))
        story.append(para("Weak performance means delays are mostly not explained by the recorded patient and arrival information. The model explains itself, not causes.", styles["small"]))

    if data.simulation:
        s = data.simulation
        story += [Spacer(1, 6), para("What-if scenario (hypothetical)", styles["h1"]),
                  para("HYPOTHETICAL: this is arithmetic on recorded waiting times. It does not predict real outcomes.", ParagraphStyle("warn", parent=styles["body"], textColor=colors.HexColor("#b42318"))),
                  para(s["scenario_text"], styles["body"]), _table(s["summary"].round(1), styles),
                  para(f"{s['newly_within_one_hour']:,} more of {s['patients']:,} treated patients would be within 1 hour under these assumptions.", styles["small"])]

    story += [para("Limitations", styles["h1"])] + [Paragraph("&bull; " + escape(ascii_text(line)), styles["bullet"]) for line in data.limitations]
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()
