"""
Spamurai - five trained models slicing spam from email they have never seen
ML Assignment 2, M.Tech AIML, BITS Pilani WILP
Built by Naren (2026AJ05871)

The app only LOADS the models saved by the training notebook (model/*.pkl)
and scores the test emails. Nothing is retrained here.
"""
import ast
import base64
import io
import json
import textwrap
import time
from datetime import date
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from scipy.stats import chi2
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

APP_NAME = "Spamurai"
APP_TAGLINE = "Spam gets the sword. Real mail gets the inbox."
APP_SUBLINE = "Spam detection, tested on emails the models never saw during training."
AUTHOR = "Naren (2026AJ05871)"

HERE = Path(__file__).parent
MODEL_DIR = HERE / "model"
SAMPLE_CSV = HERE / "test_data.csv"
MASCOT_GIF = HERE / "assets" / "spamurai.gif"

# ---------------------------------------------------------------- palette (taken from the logo)
PAPER, INK, MUTED = "#F6F7F9", "#17243A", "#5B6577"
CORAL, CORAL_DARK = "#F15A4A", "#B8382A"          # spam / mistakes
TEAL, TEAL_DARK = "#2BB7A5", "#1D7C70"            # legitimate / correct / final model
GOLD = "#E3A72F"
RULE, GREY_BAR = "#DDE1E8", "#BAC2CF"
# names used by the chart helpers
POSTAL_BLUE, AIRMAIL_RED, SORTED_GREEN = INK, CORAL, TEAL

SHORT = {"Logistic Regression": "LR", "Decision Tree": "DT", "KNN": "KNN", "Naive Bayes": "NB", "SVM": "SVM"}
MODEL_NOTES = {
    "Logistic Regression": "Weighs the evidence from every feature, adds it up and turns the total into a spam score.",
    "Decision Tree": "Asks a chain of yes/no questions about the email, like a flowchart.",
    "KNN": "Finds the most similar emails it saw in training and lets them vote.",
    "Naive Bayes": "Treats each feature as independent evidence and multiplies the odds together.",
    "SVM": "Draws a curved boundary between spam and real mail, using many features at once.",
}
METRIC_MEANING = {
    "Accuracy":  "Share of all emails sorted correctly",
    "AUC":       "How well spam is ranked above real mail (1 = perfect, 0.5 = guessing)",
    "Precision": "When it says spam, how often it is right",
    "Recall":    "Share of the real spam it catches",
    "F1":        "Balance between precision and recall",
    "MCC":       "Overall agreement with the truth, fair even with uneven classes (-1 to 1)",
}

st.set_page_config(page_title=f"{APP_NAME} | Spam detection", page_icon="⚔️", layout="wide")

# ---------------------------------------------------------------- styles
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800&family=Source+Sans+3:wght@400;600;700&display=swap');

.block-container {{ padding-top: 3.6rem; max-width: 1250px; }}
:where([data-testid="stMarkdownContainer"]) p, :where([data-testid="stMarkdownContainer"]) li {{
    font-family: 'Source Sans 3', sans-serif; font-size: 1.02rem; line-height: 1.5; }}
h1, h2, h3 {{ font-family: 'Archivo', sans-serif !important; color: {INK}; letter-spacing: -0.01em; }}

/* ---- header: logo text + mascot */
.hero-head {{ display: flex; align-items: center; justify-content: space-between; gap: 1.5rem; flex-wrap: wrap; }}
.logo {{ font-family: Verdana, 'DejaVu Sans', 'Segoe UI', sans-serif; font-weight: 700;
    font-size: 4.4rem; line-height: 1.1; margin: 0; padding-bottom: .22em; letter-spacing: -0.01em; }}
.logo .spam {{ color: {CORAL}; position: relative; display: inline-block; }}
.logo .spam::after {{ content: ""; position: absolute; left: -.04em; right: .02em; bottom: -.1em;
    height: .1em; border-radius: 99px; background: {TEAL}; }}
.logo .urai {{ color: {INK}; }}
.tagline {{ font-family: 'Archivo', sans-serif; font-weight: 700; color: {INK}; font-size: 1.35rem;
    line-height: 1.3; margin: .9rem 0 .35rem 0 !important; max-width: 48ch; }}
.subline {{ font-family: 'Source Sans 3', sans-serif; color: {MUTED}; font-size: 1.05rem; margin: 0 0 1rem 0 !important; }}
.jump {{ display: inline-block; font-family: 'Source Sans 3', sans-serif; font-weight: 700; font-size: .98rem;
    color: {INK} !important; text-decoration: none !important; border: 2px solid {INK}; border-radius: 99px;
    padding: .35rem 1rem; }}
.jump:hover {{ background: {INK}; color: #FFFFFF !important; }}
.mascot {{ width: 340px; max-width: 100%; height: auto; padding-top: 12px; }}
#compare {{ scroll-margin-top: 4.5rem; }}

/* ---- verdict hero */
.verdict {{ background: #FFFFFF; border: 1px solid {RULE}; border-radius: 12px; padding: 1.2rem 1.4rem;
    border-top: 5px solid {TEAL}; }}
.verdict-top {{ display: flex; align-items: center; gap: 1rem; margin-bottom: .9rem; }}
.verdict-top h3 {{ margin: 0; font-size: 1.45rem; }}
.verdict-top p {{ margin: .15rem 0 0 0; color: {MUTED}; font-family: 'Source Sans 3', sans-serif; }}
.belt {{ font-family: 'Archivo', sans-serif; font-weight: 800; color: #FFFFFF; background: {TEAL_DARK};
    border-radius: 8px; padding: .5rem .75rem; text-align: center; line-height: 1.05; font-size: .8rem; }}
.belt b {{ display: block; font-size: 1.55rem; }}
.stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: .6rem; }}
.stat {{ background: {PAPER}; border-radius: 9px; padding: .75rem .85rem; }}
.stat .n {{ font-family: 'Archivo', sans-serif; font-weight: 800; font-size: 2.1rem; color: {INK}; line-height: 1; }}
.stat .l {{ font-family: 'Source Sans 3', sans-serif; color: {MUTED}; font-size: .92rem; margin-top: .3rem; line-height: 1.3; }}
.stat.good .n {{ color: {TEAL_DARK}; }}
.reasons {{ margin: .9rem 0 0 0; padding-left: 1.1rem; font-family: 'Source Sans 3', sans-serif; color: {INK}; }}
.reasons li {{ margin: .15rem 0; }}

/* ---- evidence pack: the whole column becomes a dark card */
[data-testid="stColumn"]:has(.evidence), [data-testid="column"]:has(.evidence) {{
    background: {INK}; border-radius: 12px; padding: 1.2rem 1.3rem 1.1rem; }}
.evidence {{ color: #FFFFFF; font-family: 'Source Sans 3', sans-serif; }}
.evidence .tag {{ display: inline-block; background: {GOLD}; color: {INK}; font-weight: 700; font-size: .82rem;
    border-radius: 99px; padding: .15rem .6rem; }}
.evidence h3 {{ color: #FFFFFF !important; margin: .55rem 0 .35rem 0; font-size: 1.35rem; }}
.evidence p {{ color: #D9DEE7; margin: 0 0 .5rem 0; }}
.evidence ul {{ margin: 0 0 .8rem 0; padding-left: 1.1rem; color: #D9DEE7; }}
.evidence li {{ margin: .1rem 0; }}
[data-testid="stColumn"]:has(.evidence) [data-testid="stDownloadButton"] button,
[data-testid="column"]:has(.evidence) [data-testid="stDownloadButton"] button {{
    width: 100%; background: {CORAL}; border: none; border-radius: 9px; padding: .75rem 1rem; }}
[data-testid="stColumn"]:has(.evidence) [data-testid="stDownloadButton"] button p,
[data-testid="column"]:has(.evidence) [data-testid="stDownloadButton"] button p {{
    color: #FFFFFF; font-weight: 700; font-size: 1.05rem; }}
[data-testid="stColumn"]:has(.evidence) [data-testid="stDownloadButton"] button:hover,
[data-testid="column"]:has(.evidence) [data-testid="stDownloadButton"] button:hover {{ background: {CORAL_DARK}; }}

/* ---- how to use: three steps */
.howto {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: .8rem; margin: 1.3rem 0 .4rem 0; }}
.step {{ display: flex; gap: .75rem; align-items: flex-start; background: #FFFFFF; border: 1px solid {RULE};
    border-radius: 10px; padding: .8rem .95rem; font-family: 'Source Sans 3', sans-serif; }}
.step .num {{ flex-shrink: 0; width: 2rem; height: 2rem; border-radius: 50%; background: {TEAL}; color: #FFFFFF;
    font-family: 'Archivo', sans-serif; font-weight: 800; display: flex; align-items: center; justify-content: center; }}
.step b {{ font-family: 'Archivo', sans-serif; color: {INK}; font-size: 1.02rem; }}
.step span {{ display: block; color: {MUTED}; font-size: .93rem; line-height: 1.35; margin-top: .1rem; }}

/* ---- results header with a hanko-style seal */
.res-head {{ display: flex; align-items: center; justify-content: space-between; gap: 1rem; margin: 1.8rem 0 .4rem 0; }}
.res-head h2 {{ margin: 0; font-size: 1.7rem; }}
.res-head p {{ margin: .2rem 0 0 0; color: {MUTED}; font-family: 'Source Sans 3', sans-serif; }}
.hanko {{ flex-shrink: 0; width: 96px; height: 96px; border: 4px solid {CORAL}; border-radius: 14px;
    color: {CORAL}; text-align: center; font-family: 'Archivo', sans-serif; display: flex; flex-direction: column;
    justify-content: center; transform: rotate(-8deg); animation: stamp .6s cubic-bezier(.2, .9, .3, 1.25) .1s both; }}
.hanko b {{ font-size: 1.5rem; font-weight: 800; line-height: 1; }}
.hanko span {{ font-size: .78rem; font-weight: 700; margin-top: .25rem; }}
@keyframes stamp {{
    0%   {{ transform: rotate(-8deg) scale(1.8); opacity: 0; }}
    55%  {{ transform: rotate(-8deg) scale(.92); opacity: 1; }}
    100% {{ transform: rotate(-8deg) scale(1);   opacity: 1; }} }}
@media (prefers-reduced-motion: reduce) {{ .hanko {{ animation: none; }} }}

.section-title {{ font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 1.3rem;
    color: {INK}; margin: 1.2rem 0 .2rem 0 !important; }}
.section-note {{ font-family: 'Source Sans 3', sans-serif; color: {MUTED}; margin: 0 0 .8rem 0 !important; }}

/* ---- confusion matrix drawn as four trays */
.trays {{ display: grid; grid-template-columns: 8.5rem 1fr 1fr; gap: 6px; font-family: 'Source Sans 3', sans-serif; }}
.trays .head {{ font-weight: 600; color: {MUTED}; font-size: .92rem; align-self: end; padding: 0 .4rem .2rem; }}
.trays .side {{ font-weight: 600; color: {MUTED}; font-size: .92rem; align-self: center; }}
.tray {{ border-radius: 8px; padding: .9rem 1rem; min-height: 6.2rem; }}
.tray .n {{ display: block; font-family: 'Archivo', sans-serif; font-weight: 800; font-size: 2.3rem; line-height: 1; }}
.tray .t {{ display: block; margin-top: .35rem; font-size: .95rem; }}
.tray.ok  {{ background: #E2F4F1; color: #17665C; border-left: 5px solid {TEAL}; }}
.tray.bad {{ background: #FDE8E5; color: #A3291C; border-left: 5px solid {CORAL}; }}

/* ---- scorecard */
.score {{ display: grid; grid-template-columns: 6.2rem 4.6rem 1fr 5.6rem; gap: .35rem .8rem;
    align-items: center; font-family: 'Source Sans 3', sans-serif; }}
.score .m {{ font-weight: 700; color: {INK}; }}
.score .v {{ font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 1.15rem; color: {INK}; }}
.score .bar {{ background: {RULE}; border-radius: 3px; height: 9px; position: relative; }}
.score .bar span {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 3px; background: {INK}; }}
.score .r {{ color: {MUTED}; font-size: .9rem; text-align: right; white-space: nowrap; }}
.score .d {{ grid-column: 1 / -1; color: {MUTED}; font-size: .88rem; margin: -.2rem 0 .45rem 0; }}

/* ---- tabs styled as folder tabs so they read as clickable */
.explore {{ font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 1.7rem; color: {INK}; margin: 2.2rem 0 .1rem 0 !important; }}
.explore-note {{ font-family: 'Source Sans 3', sans-serif; color: {MUTED}; margin: 0 0 .6rem 0 !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; flex-wrap: wrap; border-bottom: 2px solid {INK}; }}
.stTabs [data-baseweb="tab"] {{ height: auto; background: #E9ECF1; border: 1.5px solid {RULE};
    border-bottom: none; border-radius: 9px 9px 0 0; padding: .65rem 1.15rem; margin-bottom: 0; }}
.stTabs [data-baseweb="tab"] p {{ font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 1.02rem; color: {INK}; }}
.stTabs [data-baseweb="tab"]:hover {{ background: #FFFFFF; border-color: {INK}; }}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{ background: {INK}; border-color: {INK}; }}
.stTabs [data-baseweb="tab"][aria-selected="true"] p {{ color: #FFFFFF; }}
.stTabs [data-baseweb="tab"]:focus-visible {{ outline: 3px solid {CORAL}; outline-offset: 2px; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}
.stTabs [data-baseweb="tab-panel"] {{ background: #FFFFFF; border: 1px solid {RULE}; border-top: none;
    border-radius: 0 0 10px 10px; padding: 1.2rem 1.4rem 1.4rem; }}

/* ---- data-scope labels in the Why tab */
.phase {{ font-family: 'Source Sans 3', sans-serif; color: {INK}; margin: 1.3rem 0 .5rem 0 !important;
    padding-top: .9rem; border-top: 1px solid {RULE}; }}
.scope {{ display: inline-block; font-weight: 700; font-size: .85rem; border-radius: 6px;
    padding: .12rem .55rem; margin-right: .55rem; }}
.scope.train {{ background: {GOLD}; color: {INK}; }}
.scope.test {{ background: {TEAL_DARK}; color: #FFFFFF; }}

/* ---- mistake cards */
.cards2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: .4rem 0 1rem 0; }}
.mcard {{ border-radius: 10px; padding: 1rem 1.2rem; font-family: 'Source Sans 3', sans-serif; }}
.mcard .n {{ font-family: 'Archivo', sans-serif; font-weight: 800; font-size: 2.6rem; line-height: 1; }}
.mcard .h {{ font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 1.1rem; margin: .2rem 0; }}
.mcard.fa {{ background: #FDE8E5; color: #A3291C; }}
.mcard.ms {{ background: #FCF1DC; color: #7A4A06; }}

/* ---- sidebar model card */
.profile {{ background: #FFFFFF; border: 1px solid {RULE}; border-radius: 10px; padding: .85rem .95rem;
    font-family: 'Source Sans 3', sans-serif; color: {INK}; font-size: .95rem; }}
.profile b.t {{ font-family: 'Archivo', sans-serif; font-size: 1.05rem; }}
.profile .final {{ display: inline-block; background: {TEAL_DARK}; color: #FFFFFF; font-size: .75rem; font-weight: 700;
    border-radius: 99px; padding: .05rem .5rem; margin-left: .3rem; vertical-align: 2px; }}
.profile ul {{ margin: .3rem 0 .4rem 0; padding-left: 1.05rem; }}
.profile li {{ margin: .12rem 0; }}
.profile .tech {{ color: {MUTED}; font-size: .82rem; }}
.footer {{ margin-top: 2.5rem; padding-top: .8rem; border-top: 1px solid {RULE};
    font-family: 'Source Sans 3', sans-serif; color: {MUTED}; font-size: .9rem; }}
@media (max-width: 900px) {{ .stats {{ grid-template-columns: repeat(2, 1fr); }} .howto {{ grid-template-columns: 1fr; }} }}
@media (max-width: 760px) {{ .cards2 {{ grid-template-columns: 1fr; }} .logo {{ font-size: 3.2rem; }} }}
</style>
""", unsafe_allow_html=True)


# ================================================================ loading and scoring
@st.cache_resource
def load_bundle():
    with open(MODEL_DIR / "metadata.json") as f:
        meta = json.load(f)
    models = {name: joblib.load(MODEL_DIR / fname) for name, fname in meta["model_files"].items()}
    summary_path = MODEL_DIR / "training_summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else None
    return meta, models, summary


def score_all_metrics(y, pred, proba):
    auc = roc_auc_score(y, proba) if len(np.unique(y)) == 2 else np.nan
    return {
        "Accuracy":  accuracy_score(y, pred),
        "AUC":       auc,
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall":    recall_score(y, pred, zero_division=0),
        "F1":        f1_score(y, pred, zero_division=0),
        "MCC":       matthews_corrcoef(y, pred),
    }


def check_file(df, features, target):
    problems = []
    missing = [c for c in features if c not in df.columns]
    if missing:
        shown = ", ".join(missing[:6]) + (f" and {len(missing) - 6} more" if len(missing) > 6 else "")
        problems.append(f"{len(missing)} feature column(s) are missing: {shown}.")
    if target not in df.columns:
        problems.append(f"The label column '{target}' is missing. Add it with 1 for spam and 0 for legitimate.")
    if not problems:
        bad = [c for c in features if pd.to_numeric(df[c], errors="coerce").isna().any()]
        if bad:
            problems.append(f"Non-numeric or empty values found in: {', '.join(bad[:6])}.")
        labels = set(pd.to_numeric(df[target], errors="coerce").dropna().unique())
        if not labels.issubset({0, 1}) or df[target].isna().any():
            problems.append(f"The '{target}' column must contain only 0 and 1.")
    return problems


@st.cache_data(show_spinner="Sorting the emails with all five models...")
def sort_mail(csv_bytes):
    meta, models, _ = load_bundle()
    df = pd.read_csv(io.BytesIO(csv_bytes))
    problems = check_file(df, meta["features"], meta["target"])
    if problems:
        return df, problems, None
    X = df[meta["features"]].astype(float)
    y = df[meta["target"]].astype(int).to_numpy()
    runs = {}
    for name, model in models.items():
        t0 = time.perf_counter()
        pred = model.predict(X)
        proba = model.predict_proba(X)[:, 1]
        secs = time.perf_counter() - t0
        runs[name] = {"pred": pred, "proba": proba, "metrics": score_all_metrics(y, pred, proba),
                      "ms_per_1000": secs / len(X) * 1_000_000}
    return df, [], {"y": y, "runs": runs}


def mcnemar(correct_a, correct_b):
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    if a_only + b_only == 0:
        return a_only, b_only, 1.0
    stat = (abs(a_only - b_only) - 1) ** 2 / (a_only + b_only)
    return a_only, b_only, float(chi2.sf(stat, df=1))


# ================================================================ shared charts (app + PDF)
def style_axes(ax, grid_axis="x"):
    ax.set_facecolor("white")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=INK, labelsize=10)
    ax.grid(axis=grid_axis, color=RULE, lw=0.8)
    ax.set_axisbelow(True)


def new_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("white")
    return fig, ax


def fmt_p(p):
    """A p-value is never exactly zero, so very small values are shown as p < 0.001."""
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


SCOPE_STYLE = {
    "train": (GOLD, INK),        # model selection
    "test": (TEAL_DARK, "#FFFFFF"),  # final evaluation
    "both": (INK, "#FFFFFF"),
}


def chart_title(ax, text, scope=None, n_test=None):
    """Title plus a coloured badge saying which data the chart is based on."""
    ax.set_title(text, loc="left", fontsize=12, fontweight="bold", color=INK, pad=8)
    if scope:
        n_txt = f"{n_test:,}" if n_test else "the"
        badge = {"train": "Model selection: cross-validation on training data",
                 "test": f"Final evaluation: results on {n_txt} held-out test emails",
                 "both": f"Training data vs {n_txt} held-out test emails"}[scope]
        bg, fg = SCOPE_STYLE[scope]
        ax.annotate(badge, xy=(0, 1), xycoords="axes fraction", xytext=(0, 27), textcoords="offset points",
                    ha="left", va="bottom", fontsize=9, fontweight="bold", color=fg,
                    bbox=dict(boxstyle="round,pad=0.35", fc=bg, ec="none"))


def draw_mistakes(ax, errors, final, n_test):
    order = pd.Series(errors).sort_values(ascending=False)
    ax.barh(order.index, order.values,
            color=[SORTED_GREEN if n == final else GREY_BAR for n in order.index], height=0.6)
    for i, v in enumerate(order.values):
        ax.text(v, i, f"  {v}", va="center", fontsize=10, color=INK)
    ax.set_xlim(0, order.max() * 1.15)
    ax.set_xlabel("mistakes (lower is better)", color=MUTED)
    chart_title(ax, "Total mistakes", "test", n_test)
    style_axes(ax)


def draw_metric(ax, all_metrics, metric, final, title, n_test):
    order = all_metrics[metric].sort_values()
    ax.barh(order.index, order.values,
            color=[SORTED_GREEN if n == final else GREY_BAR for n in order.index], height=0.6)
    for i, v in enumerate(order.values):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=10, color=INK)
    ax.set_xlim(max(0, order.min() - 0.1), 1.02)
    ax.set_xlabel(f"{metric} (higher is better)", color=MUTED)
    chart_title(ax, title, "test", n_test)
    style_axes(ax)


def draw_cv(ax, summary, final):
    cv = summary["cv"]
    names_ = sorted(cv, key=lambda n: cv[n]["F1"][0])
    means = [cv[n]["F1"][0] for n in names_]
    stds = [cv[n]["F1"][1] for n in names_]
    ax.barh(names_, means, xerr=stds, capsize=4, height=0.6,
            color=[SORTED_GREEN if n == final else GREY_BAR for n in names_],
            error_kw={"ecolor": INK, "lw": 1})
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 0.004, i, f"{m:.3f}", va="center", fontsize=10, color=INK)
    ax.set_xlim(max(0, min(means) - 0.08), 1.0)
    ax.set_xlabel(f"mean F1 over {summary.get('cv_folds', 5)} folds of {summary.get('n_train', 0):,} "
                  f"training emails (bar) and spread between folds (whisker)", color=MUTED)
    chart_title(ax, "Step 1: which model scored best while training?", "train")
    style_axes(ax)


def draw_head_to_head(ax, correct, final, names_, n_test):
    others = [n for n in names_ if n != final]
    lim = 1
    for i, n in enumerate(others):
        a, b, p = mcnemar(correct[final], correct[n])
        lim = max(lim, a, b)
        ax.barh(i, a, color=SORTED_GREEN, height=0.5)
        ax.barh(i, -b, color=GREY_BAR, height=0.5)
        ax.text(a, i, f"  {a}", va="center", fontweight="bold", color=INK)
        ax.text(-b, i, f"{b}  ", va="center", ha="right", fontweight="bold", color=INK)
        ax.text(0, i - 0.38, f"{fmt_p(p)}: {'real difference' if p < 0.05 else 'could be chance'}",
                ha="center", va="bottom", fontsize=9.5, color=TEAL_DARK if p < 0.05 else CORAL_DARK)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(range(len(others)))
    ax.set_yticklabels([f"{final} vs {n}" for n in others])
    ax.set_xlim(-lim * 1.35, lim * 1.35)
    ax.set_ylim(len(others) - 0.5, -0.9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{abs(v):.0f}"))
    ax.set_xlabel(f"emails only the other model got right   |   emails only {final} got right", color=MUTED)
    chart_title(ax, "Step 3: is the lead real? Head to head where two models disagree", "test", n_test)
    style_axes(ax)


def draw_gap(ax, summary, all_metrics, final, n_test):
    rows = [(n, summary["train_accuracy"][n], all_metrics.loc[n, "Accuracy"])
            for n in all_metrics.index if n in summary.get("train_accuracy", {})]
    rows.sort(key=lambda r: r[1] - r[2])
    for i, (n, tr, te) in enumerate(rows):
        c = SORTED_GREEN if n == final else "#8C96A8"
        ax.plot([te, tr], [i, i], color=c, lw=3, alpha=.45)
        ax.scatter(tr, i, s=80, color=c, zorder=3)
        ax.scatter(te, i, s=80, facecolor="white", edgecolor=c, lw=2, zorder=4)
        ax.text(max(tr, te) + 0.004, i, f"gap {(tr - te) * 100:+.1f} pp", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    lo = min(min(r[1], r[2]) for r in rows) - 0.02
    ax.set_xlim(lo, 1.05)
    ax.set_xticks([t for t in np.arange(np.floor(lo * 50) / 50, 1.0001, 0.02)])
    ax.set_xlabel("accuracy: filled dot = training, hollow dot = test\n(a shorter line means less overfitting)", color=MUTED)
    chart_title(ax, "Step 4: does it generalise?", "both", n_test)
    style_axes(ax)


def draw_cost(ax, runs, final, n_test):
    ms = pd.Series({n: r["ms_per_1000"] for n, r in runs.items()}).sort_values()
    fmt = lambda v: f"{v:,.1f} ms" if v < 10 else f"{v:,.0f} ms"
    cols = [SORTED_GREEN if n == final else GREY_BAR for n in ms.index]
    ax.barh(ms.index, ms.values, color=cols, height=0.6)
    for i, v in enumerate(ms.values):
        ax.text(v, i, f"  {fmt(v)}", va="center", fontsize=10, color=INK)
    ax.set_xlim(0, ms.max() * 1.25)
    ax.set_xlabel("time to score 1,000 emails on this server (lower is faster)", color=MUTED)
    chart_title(ax, "Step 5: the cost - scoring speed", "test", n_test)
    style_axes(ax)
    # The slowest model squashes the others, so zoom in on the fast ones in an inset
    fast = ms[ms < ms.max() * 0.35]
    if len(fast) >= 2:
        ins = ax.inset_axes([0.47, 0.06, 0.5, 0.5])
        ins.barh(fast.index, fast.values, height=0.6,
                 color=[SORTED_GREEN if n == final else GREY_BAR for n in fast.index])
        for i, v in enumerate(fast.values):
            ins.text(v, i, f" {fmt(v)}", va="center", fontsize=8.5, color=INK)
        ins.set_xlim(0, fast.max() * 1.45)
        ins.set_title("zoom: the faster models", fontsize=8.5, color=MUTED, loc="left")
        ins.tick_params(labelsize=8, colors=INK)
        ins.set_facecolor("#FAFBFC")
        for s in ins.spines.values():
            s.set_color(RULE)


def draw_cm(ax, y, pred, name, final):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ok = i == j
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if ok else INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["said legit", "said spam"], fontsize=8.5)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["was legit", "was spam"], fontsize=8.5)
    ax.set_title(name + ("  (final)" if name == final else ""), fontsize=10.5, fontweight="bold",
                 color=SORTED_GREEN if name == final else INK)


# ================================================================ the downloadable "Why" report
def build_reasons(all_metrics, errors, correct, summary, final, n_emails):
    runner = all_metrics.drop(index=final)["F1"].idxmax()
    w, l, p_run = mcnemar(correct[final], correct[runner])
    fewer = 1 - errors[final] / max(errors[runner], 1)
    best_cols = [m for m in all_metrics.columns if all_metrics[m].idxmax() == final]
    reasons = []
    if summary and final in summary.get("cv", {}):
        cv = summary["cv"]
        cv_rank = sorted(cv, key=lambda n: -cv[n]["F1"][0]).index(final) + 1
        reasons.append(f"Chosen during training: {'the highest' if cv_rank == 1 else f'rank {cv_rank} for'} "
                       f"average F1 over {summary.get('cv_folds', 5)}-fold cross-validation "
                       f"({cv[final]['F1'][0]:.3f} ± {cv[final]['F1'][1]:.3f}) on "
                       f"{summary.get('n_train', 0):,} training emails.")
    reasons.append(f"Confirmed on {n_emails:,} unseen test emails: best on {len(best_cols)} of 6 metrics "
                   f"({', '.join(best_cols)}), with F1 {all_metrics.loc[final, 'F1']:.4f} and "
                   f"MCC {all_metrics.loc[final, 'MCC']:.4f}.")
    reasons.append(f"Fewest mistakes: {errors[final]}, which is {fewer:.0%} fewer than the runner-up, "
                   f"{runner} ({errors[runner]}).")
    reasons.append(f"Not luck: on the emails where {final} and {runner} disagree, {final} is right {w} times "
                   f"and {runner} {l} times (McNemar {fmt_p(p_run)}"
                   + (", a statistically real difference)." if p_run < 0.05 else ", which could still be chance)."))
    if summary and final in summary.get("train_accuracy", {}):
        gap = (summary["train_accuracy"][final] - all_metrics.loc[final, "Accuracy"]) * 100
        reasons.append(f"Generalises well: {gap:+.1f} percentage points between training and test accuracy.")
    return runner, reasons


@st.cache_data(show_spinner="Building the PDF report...")
def build_report_pdf(csv_bytes, final):
    meta, _, summary = load_bundle()
    _, _, scored = sort_mail(csv_bytes)
    y, runs = scored["y"], scored["runs"]
    names_ = list(runs)
    all_m = pd.DataFrame({n: r["metrics"] for n, r in runs.items()}).T
    errs = {n: int((r["pred"] != y).sum()) for n, r in runs.items()}
    corr = {n: (r["pred"] == y) for n, r in runs.items()}
    runner, reasons = build_reasons(all_m, errs, corr, summary, final, len(y))

    buf = io.BytesIO()
    A4 = (11.69, 8.27)

    def footer(fig, page):
        fig.text(0.04, 0.03, f"{APP_NAME}  |  {AUTHOR}  |  BITS Pilani WILP ML Assignment 2", fontsize=8, color=MUTED)
        fig.text(0.96, 0.03, f"page {page}", fontsize=8, color=MUTED, ha="right")

    with PdfPages(buf) as pdf:
        # page 1: the decision in words
        fig = plt.figure(figsize=A4)
        fig.patch.set_facecolor("white")
        fig.add_artist(plt.Rectangle((0, 0.965), 1, 0.035, transform=fig.transFigure,
                                     color=INK, lw=0))
        logo = HPacker(children=[TextArea("Spam", textprops=dict(color=CORAL, fontsize=20, fontweight="bold")),
                                 TextArea("urai", textprops=dict(color=INK, fontsize=20, fontweight="bold"))],
                       pad=0, sep=0)
        fig.add_artist(AnchoredOffsetbox(loc="upper left", child=logo, frameon=False,
                                         bbox_to_anchor=(0.055, 0.955), bbox_transform=fig.transFigure))
        if MASCOT_GIF.exists():
            try:
                from PIL import Image
                gif_img = Image.open(MASCOT_GIF)
                gif_img.seek(gif_img.n_frames - 1)
                ax_m = fig.add_axes([0.72, 0.74, 0.24, 0.2]); ax_m.axis("off")
                ax_m.imshow(np.asarray(gif_img.convert("RGBA")))
            except Exception:
                pass
        fig.text(0.06, 0.8, f"Why {final} is the final model", fontsize=26, fontweight="bold", color=INK)
        fig.text(0.06, 0.755, APP_TAGLINE, fontsize=11, color=MUTED)
        fig.text(0.06, 0.725, f"Report generated {date.today():%d %B %Y} from {len(y):,} test emails.",
                 fontsize=11, color=MUTED)
        yy = 0.655
        fig.text(0.06, yy, "The evidence", fontsize=15, fontweight="bold", color=INK)
        yy -= 0.05
        for k, r in enumerate(reasons, 1):
            lines = textwrap.wrap(r, 115)
            fig.text(0.07, yy, f"{k}.", fontsize=11.5, fontweight="bold", color=TEAL_DARK)
            for ln in lines:
                fig.text(0.095, yy, ln, fontsize=11.5, color=INK)
                yy -= 0.035
            yy -= 0.015
        yy -= 0.02
        fig.text(0.06, yy, "The trade-off", fontsize=15, fontweight="bold", color=INK)
        yy -= 0.05
        kb = {n: (MODEL_DIR / meta["model_files"][n]).stat().st_size / 1024 for n in names_}
        trade = (f"{final} is not the cheapest model: its saved model is {kb[final]:,.0f} KB and it scores "
                 f"1,000 emails in about {runs[final]['ms_per_1000']:,.0f} ms on the app server. "
                 f"{runner} is the runner-up. For a spam filter, losing fewer genuine emails to the spam "
                 f"folder is worth the extra size and time.")
        for ln in textwrap.wrap(trade, 120):
            fig.text(0.07, yy, ln, fontsize=11.5, color=INK)
            yy -= 0.035
        footer(fig, 1)
        pdf.savefig(fig); plt.close(fig)

        # page 2: cross-validation + test metrics
        fig = plt.figure(figsize=A4); fig.patch.set_facecolor("white")
        gs = fig.add_gridspec(2, 2, left=0.17, right=0.95, top=0.9, bottom=0.12, hspace=0.55, wspace=0.55)
        if summary and summary.get("cv"):
            draw_cv(fig.add_subplot(gs[0, :]), summary, final)
        else:
            ax = fig.add_subplot(gs[0, :]); ax.axis("off")
            ax.text(0, .5, "Cross-validation results are in Section 7 of the training notebook.", fontsize=12)
        draw_metric(fig.add_subplot(gs[1, 0]), all_m, "F1", final, "Step 2: F1", len(y))
        draw_metric(fig.add_subplot(gs[1, 1]), all_m, "MCC", final, "Step 2: MCC", len(y))
        footer(fig, 2)
        pdf.savefig(fig); plt.close(fig)

        # page 3: confusion matrices + mistakes
        fig = plt.figure(figsize=A4); fig.patch.set_facecolor("white")
        fig.text(0.06, 0.92, "Step 2 (continued): where each model put the test emails",
                 fontsize=14, fontweight="bold", color=INK)
        gs = fig.add_gridspec(2, 5, left=0.08, right=0.97, top=0.84, bottom=0.1, hspace=0.6, wspace=0.7)
        for k, n in enumerate(names_):
            ax = fig.add_subplot(gs[0, k] if k < 5 else gs[1, 0])
            draw_cm(ax, y, runs[n]["pred"], n, final)
        draw_mistakes(fig.add_subplot(gs[1, 1:4]), errs, final, len(y))
        footer(fig, 3)
        pdf.savefig(fig); plt.close(fig)

        # page 4: significance + generalisation
        fig = plt.figure(figsize=A4); fig.patch.set_facecolor("white")
        gs = fig.add_gridspec(2, 1, left=0.22, right=0.95, top=0.9, bottom=0.12, hspace=0.55)
        draw_head_to_head(fig.add_subplot(gs[0]), corr, final, names_, len(y))
        if summary and summary.get("train_accuracy"):
            draw_gap(fig.add_subplot(gs[1]), summary, all_m, final, len(y))
        else:
            draw_cost(fig.add_subplot(gs[1]), runs, final, len(y))
        footer(fig, 4)
        pdf.savefig(fig); plt.close(fig)

        # page 5: full metrics table
        fig = plt.figure(figsize=A4); fig.patch.set_facecolor("white")
        fig.text(0.06, 0.9, "All metrics on the test emails", fontsize=14, fontweight="bold", color=INK)
        tbl = all_m.copy()
        tbl["Mistakes"] = pd.Series(errs)
        cells = [[f"{v:.4f}" for v in tbl.loc[n, list(METRIC_MEANING)]] + [str(tbl.loc[n, "Mistakes"])]
                 for n in tbl.index]
        ax = fig.add_axes([0.22, 0.45, 0.72, 0.38]); ax.axis("off")
        t = ax.table(cellText=cells, rowLabels=list(tbl.index), colLabels=list(METRIC_MEANING) + ["Mistakes"],
                     loc="center", cellLoc="center")
        t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1, 2)
        for (r, c), cell in t.get_celld().items():
            cell.set_edgecolor(RULE)
            if r == 0:
                cell.set_text_props(fontweight="bold")
            if r > 0 and tbl.index[r - 1] == final:
                cell.set_facecolor("#E3EFE9")
        yy = 0.36
        for m, txt in METRIC_MEANING.items():
            fig.text(0.07, yy, f"{m}: {txt}.", fontsize=10.5, color=MUTED)
            yy -= 0.032
        footer(fig, 5)
        pdf.savefig(fig); plt.close(fig)
    return buf.getvalue()


@st.cache_data
def mascot_b64():
    return base64.b64encode(MASCOT_GIF.read_bytes()).decode() if MASCOT_GIF.exists() else ""


def plain_settings(name, params, summary):
    """Turn tuned hyperparameters into plain English for non-technical readers."""
    try:
        p = ast.literal_eval(params) if isinstance(params, str) and params.strip().startswith("{") else {}
    except (ValueError, SyntaxError):
        p = {}
    out = []
    if name == "SVM":
        C, g = p.get("C"), p.get("gamma")
        if C is not None:
            out.append("Allowed to bend its boundary around tricky emails" if C >= 5 else
                       "Balances a smooth boundary against fitting every email" if C >= 1 else
                       "Keeps a smooth boundary and tolerates a few misfits")
        if g is not None:
            out.append("How far each email's influence reaches is set automatically from the data"
                       if g == "scale" else f"How far each email's influence reaches is fixed at {g}")
    elif name == "Logistic Regression":
        C = p.get("C")
        if C is not None:
            out.append("Kept deliberately simple, so it learns broad patterns rather than quirks" if C <= 0.1 else
                       "Moderately simple" if C <= 1 else "Allowed to fit fine details")
    elif name == "KNN":
        k, wts = p.get("n_neighbors"), p.get("weights")
        if k:
            out.append(f"Looks at the {k} most similar training emails")
        if wts:
            out.append("Closer emails get a bigger vote" if wts == "distance" else "Every neighbour gets an equal vote")
    elif name == "Decision Tree":
        pr = (summary or {}).get("pruning")
        if p.get("criterion"):
            out.append("Each question is chosen to split spam from real mail as cleanly as possible")
        if pr:
            out.append(f"Pruned from {pr['leaves_before']} to {pr['leaves_after']} end points, so it learns "
                       f"rules instead of memorising emails")
        elif p.get("max_depth") is None and "max_depth" in p:
            out.append("No limit on how many questions it can ask")
    elif name == "Naive Bayes":
        out.append("Nothing to tune: it learns the typical value of each feature for spam and for real mail")
    tech = ", ".join(f"{k} = {v}" for k, v in p.items()) if p else "none"
    return out or ["Default settings"], tech



# ================================================================ page
meta, models, summary = load_bundle()
FEATURES, TARGET = meta["features"], meta["target"]
MODEL_NAMES = list(meta["model_files"].keys())
FINAL = (summary or {}).get("final_model", "SVM")
best_params = meta.get("best_params", {})
label_of = lambda n: f"{n}  (final model)" if n == FINAL else n

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 1. Choose the test emails")
    source_choice = st.radio(
        "Which emails should the models sort?",
        ["Test set included with this app", "Upload a CSV file"],
        help="The included file is the same test_data.csv stored in the GitHub repo: "
             "the 20% of emails held back during training.")
    uploaded = None
    if source_choice == "Upload a CSV file":
        uploaded = st.file_uploader("CSV with the 57 feature columns and a 'spam' column", type=["csv"])
        if SAMPLE_CSV.exists():
            st.caption("Don't have the file? Download it first, then upload it here.")
            st.download_button("Download test_data.csv", SAMPLE_CSV.read_bytes(),
                               file_name="test_data.csv", mime="text/csv")

    st.markdown("### 2. Choose a model")
    model_name = st.selectbox("Model to inspect", MODEL_NAMES, index=MODEL_NAMES.index(FINAL)
                              if FINAL in MODEL_NAMES else 0, format_func=label_of)
    fsize = (MODEL_DIR / meta["model_files"][model_name]).stat().st_size / 1024
    plain, tech = plain_settings(model_name, best_params.get(model_name, ""), summary)
    st.markdown(
        f"<div class='profile'><b class='t'>{model_name}</b>"
        + ("<span class='final'>final model</span>" if model_name == FINAL else "")
        + f"<br>{MODEL_NOTES.get(model_name, '')}<br><br><b>How it was tuned</b><ul>"
        + "".join(f"<li>{p}</li>" for p in plain)
        + f"</ul><span class='tech'>Technical settings: {tech}<br>Saved model size: {fsize:,.0f} KB</span></div>",
        unsafe_allow_html=True)

# ---------------------------------------------------------------- data
if source_choice == "Upload a CSV file":
    csv_bytes = uploaded.getvalue() if uploaded is not None else None
    source = uploaded.name if uploaded is not None else None
else:
    csv_bytes = SAMPLE_CSV.read_bytes() if SAMPLE_CSV.exists() else None
    source = "the test set included with this app (test_data.csv)"
result = sort_mail(csv_bytes) if csv_bytes is not None else None

# ---------------------------------------------------------------- header: logo + mascot
gif = mascot_b64()
st.markdown(
    "<div class='hero-head'><div>"
    "<div class='logo' role='heading' aria-level='1'><span class='spam'>Spam</span><span class='urai'>urai</span></div>"
    f"<div class='tagline'>{APP_TAGLINE}</div><div class='subline'>{APP_SUBLINE}</div>"
    "<a class='jump' href='#compare' target='_self'>Jump to the model comparison ↓</a></div>"
    + (f"<img class='mascot' src='data:image/gif;base64,{gif}' "
       "alt='A cartoon samurai slices a spam email in half and files the real email in the inbox tray'>"
       if gif else "")
    + "</div>", unsafe_allow_html=True)

if csv_bytes is None:
    st.markdown("<p class='section-title'>Waiting for a file</p>"
                "<p class='section-note'>Upload a CSV in the sidebar, or switch back to "
                "<b>Test set included with this app</b>. The file needs the 57 Spambase feature "
                "columns and a <b>spam</b> column (1 = spam, 0 = legitimate).</p>", unsafe_allow_html=True)
    st.stop()

df, problems, scored = result
if problems:
    st.error("This file can't be scored yet:\n\n- " + "\n- ".join(problems))
    st.stop()

y, runs = scored["y"], scored["runs"]
names = list(runs)
all_metrics = pd.DataFrame({n: r["metrics"] for n, r in runs.items()}).T
errors = {n: int((r["pred"] != y).sum()) for n, r in runs.items()}
correct = {n: (r["pred"] == y) for n, r in runs.items()}
n_spam, n_legit = int(y.sum()), int((y == 0).sum())

# ---------------------------------------------------------------- verdict hero + evidence pack
runner, reasons = build_reasons(all_metrics, errors, correct, summary, FINAL, len(y))
fm = all_metrics.loc[FINAL]
fewer = 1 - errors[FINAL] / max(errors[runner], 1)
w_, l_, p_run = mcnemar(correct[FINAL], correct[runner])
short_reasons = []
if summary and FINAL in summary.get("cv", {}):
    short_reasons.append(f"Picked during training: best average F1 in cross-validation "
                         f"({summary['cv'][FINAL]['F1'][0]:.3f}).")
short_reasons.append(f"Head to head with {runner}, it wins the disagreements {w_} to {l_} "
                     + (f"({fmt_p(p_run)}, a statistically real lead)." if p_run < 0.05 else f"({fmt_p(p_run)})."))

hero_l, hero_r = st.columns([2.15, 1], gap="medium")
with hero_l:
    st.markdown(f"""
<div class="verdict">
  <div class="verdict-top">
    <div class="belt">FINAL<b>{SHORT.get(FINAL, FINAL)}</b>MODEL</div>
    <div><h3>The final model is {FINAL}</h3>
    <p>Results on {len(y):,} test emails the models never saw during training</p></div>
  </div>
  <div class="stats">
    <div class="stat good"><div class="n">{fm['Accuracy'] * 100:.1f}%</div><div class="l">of emails sorted correctly</div></div>
    <div class="stat"><div class="n">{errors[FINAL]}</div><div class="l">mistakes out of {len(y):,}</div></div>
    <div class="stat"><div class="n">{fm['F1']:.3f}</div><div class="l">F1 score (spam caught vs false alarms)</div></div>
    <div class="stat good"><div class="n">{fewer:.0%}</div><div class="l">fewer mistakes than the runner-up, {runner}</div></div>
  </div>
  <ul class="reasons">{''.join(f'<li>{r}</li>' for r in short_reasons)}</ul>
</div>""", unsafe_allow_html=True)
with hero_r:
    st.markdown(f"""
<div class="evidence">
  <span class="tag">Evidence pack · PDF</span>
  <h3>Why {FINAL} won</h3>
  <p>A 5-page report with every chart and number behind the decision:</p>
  <ul>
    <li>cross-validation during training</li>
    <li>results on the test emails</li>
    <li>is the lead real? (significance test)</li>
    <li>overfitting check and the trade-off</li>
  </ul>
</div>""", unsafe_allow_html=True)
    st.download_button(f"⬇  Download the {FINAL} report", build_report_pdf(csv_bytes, FINAL),
                       file_name=f"spamurai_why_{FINAL.lower().replace(' ', '_')}.pdf", mime="application/pdf")

# ---------------------------------------------------------------- how to use
st.markdown(f"""
<div class="howto">
  <div class="step"><div class="num">1</div><div><b>Choose the emails</b>
    <span>In the sidebar. The test set loads by itself, or upload your own CSV.</span></div></div>
  <div class="step"><div class="num">2</div><div><b>Choose a model</b>
    <span>In the sidebar. The results below switch to that model.</span></div></div>
  <div class="step"><div class="num">3</div><div><b>Open the tabs</b>
    <span>See why {FINAL} won, compare all five, review mistakes or inspect one email.</span></div></div>
</div>""", unsafe_allow_html=True)
with st.expander(f"About {APP_NAME}: the data, the models and what this app does"):
    st.markdown(
        f"**What it does.** {APP_NAME} tests five machine-learning spam filters on emails they never saw while "
        f"learning. Each model was trained in the project notebook on 80% of the "
        f"[UCI Spambase](https://archive.ics.uci.edu/dataset/94/spambase) dataset. The other 20% were held back "
        f"as test emails, and that is what you see here.\n\n"
        f"**The data.** Each email is described by 57 numbers: how often 48 words (such as *free*, *money* or "
        f"*meeting*) and 6 characters (such as *!* and *$*) appear, and how much of it is written in capital letters.\n\n"
        f"**The models.** Logistic Regression, Decision Tree, K-Nearest Neighbours, Naive Bayes and a Support "
        f"Vector Machine. Each was tuned with 5-fold cross-validation on the training emails only.\n\n"
        f"**No retraining.** The app only loads the five saved models from the `model/` folder and scores the emails.\n\n"
        f"**Colours.** Teal means correct (or the final model), coral means a mistake, and navy marks the model "
        f"you chose in the sidebar.")

# ---------------------------------------------------------------- selected model: matrix + metrics
run = runs[model_name]
pred, proba, met = run["pred"], run["proba"], run["metrics"]
tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

st.markdown(f"""
<div class="res-head">
  <div><h2>Results for {model_name}{' (final model)' if model_name == FINAL else ''}</h2>
  <p>Scoring {source}: {len(y):,} emails, {n_legit:,} legitimate and {n_spam:,} spam.
  Choose another model in the sidebar to compare.</p></div>
  <div class="hanko" aria-label="F1 score {met['F1']:.3f}"><b>{SHORT.get(model_name, model_name)}</b>
  <span>F1 {met['F1']:.3f}</span></div>
</div>""", unsafe_allow_html=True)

left, right = st.columns([1.05, 1], gap="large")
with left:
    st.markdown(f"<p class='section-title'>Confusion matrix</p>"
                f"<p class='section-note'>Where {model_name} put each email. Teal trays are correct, coral "
                f"trays are mistakes: {fp + fn} of {len(y):,}.</p>", unsafe_allow_html=True)
    st.markdown(f"""
<div class="trays">
  <div></div><div class="head">Model said legitimate</div><div class="head">Model said spam</div>
  <div class="side">Actually legitimate<br>({n_legit:,})</div>
  <div class="tray ok"><span class="n">{tn}</span><span class="t">kept in the inbox</span></div>
  <div class="tray bad"><span class="n">{fp}</span><span class="t">wrongly sent to spam (false alarm)</span></div>
  <div class="side">Actually spam<br>({n_spam:,})</div>
  <div class="tray bad"><span class="n">{fn}</span><span class="t">slipped into the inbox (missed)</span></div>
  <div class="tray ok"><span class="n">{tp}</span><span class="t">caught as spam</span></div>
</div>""", unsafe_allow_html=True)

with right:
    st.markdown("<p class='section-title'>Evaluation metrics</p>"
                "<p class='section-note'>Spam is the positive class. Rank compares this model "
                "with the other four on the same emails.</p>", unsafe_allow_html=True)
    rows = []
    for m, v in met.items():
        if np.isnan(v):
            rows.append(f"<div class='m'>{m}</div><div class='v'>n/a</div><div></div><div></div>"
                        f"<div class='d'>Needs both classes in the file</div>")
            continue
        rank = int(all_metrics[m].rank(ascending=False, method="min")[model_name])
        width = max(0.0, min(1.0, (v + 1) / 2 if m == "MCC" else v)) * 100
        rows.append(f"<div class='m'>{m}</div><div class='v'>{v:.4f}</div>"
                    f"<div class='bar'><span style='width:{width:.1f}%'></span></div>"
                    f"<div class='r'>rank {rank} of {len(names)}</div><div class='d'>{METRIC_MEANING[m]}</div>")
    st.markdown("<div class='score'>" + "".join(rows) + "</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- tabs
st.markdown("<div id='compare'></div><p class='explore'>Explore the results</p>"
            "<p class='explore-note'>Click a folder tab to open it.</p>", unsafe_allow_html=True)
tab_why, tab_all, tab_report, tab_err, tab_one = st.tabs([
    f"🏆 Why {FINAL}?", "📊 Compare all five", "📋 Classification report",
    "⚠️ Where it went wrong", "🔍 Inspect one email"])

# ---- Why the final model
with tab_why:
    st.markdown(f"**How the final model was chosen.** All five models were tuned with cross-validation on "
                f"the training emails only. The one with the best average F1 was chosen, and the choice was "
                f"then checked on the test emails, which no model saw during training. The five steps below "
                f"are the evidence; the evidence pack PDF at the top of the page contains the same charts.")

    st.markdown("<p class='phase'><span class='scope train'>Model selection</span>"
                "Step 1 uses only the training emails. This is where the final model was picked.</p>",
                unsafe_allow_html=True)
    if summary and summary.get("cv"):
        fig, ax = new_fig(10, 3.8); draw_cv(ax, summary, FINAL)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        cv_rows = {n: {k: f"{v[0]:.4f} ± {v[1]:.4f}" for k, v in summary["cv"][n].items()} for n in summary["cv"]}
        cv_df = pd.DataFrame(cv_rows).T.loc[:, ["Accuracy", "AUC", "F1", "MCC"]]
        cv_df.index = [label_of(n) for n in cv_df.index]
        st.dataframe(cv_df.style.apply(
            lambda r: ["background-color:#E3EFE9; font-weight:700" if FINAL in r.name else ""] * len(r), axis=1))
        pr = summary.get("pruning")
        if pr:
            st.caption(f"Mean ± spread over {summary.get('cv_folds', 5)} folds of {summary.get('n_train', 0):,} "
                       f"training emails. The Decision Tree was pruned from {pr['leaves_before']} to "
                       f"{pr['leaves_after']} leaves to stop it memorising the training emails.")
    else:
        st.caption("Step 1 (cross-validation) results are in Section 7 of the training notebook.")

    st.markdown(f"<p class='phase'><span class='scope test'>Final evaluation</span>"
                f"Steps 2 to 5 use the {len(y):,} held-out test emails, which no model saw while learning. "
                f"They check that the choice holds up.</p>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = new_fig(6, 3.7); draw_metric(ax, all_metrics, "MCC", FINAL, "Step 2: overall quality (MCC)", len(y))
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    with c2:
        fig, ax = new_fig(6, 3.7); draw_mistakes(ax, errors, FINAL, len(y))
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown(f"**Step 3. Is the lead real, or luck?** We compare {FINAL} with each other model only on the "
                f"emails where they disagree. If one wins those far more often than chance would allow "
                f"(p below 0.05), the difference is real.")
    fig, ax = new_fig(10, 0.75 * (len(names) - 1) + 1.7); draw_head_to_head(ax, correct, FINAL, names, len(y))
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    c3, c4 = st.columns(2)
    with c3:
        if summary and summary.get("train_accuracy"):
            fig, ax = new_fig(6, 3.8); draw_gap(ax, summary, all_metrics, FINAL, len(y))
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        else:
            st.caption("Step 4 (training vs test accuracy) is in Section 9 of the training notebook.")
    with c4:
        fig, ax = new_fig(6, 3.8); draw_cost(ax, runs, FINAL, len(y))
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown(f"**The verdict.** {FINAL} wins on accuracy and on the mistake that matters most for email, "
                f"losing genuine messages to the spam folder. It is larger and slower than the simplest models; "
                f"if speed or explainability mattered more than accuracy, {runner} would be the sensible alternative.")

# ---- Compare all five
with tab_all:
    st.markdown("Every saved model scored on the same emails. Choose a metric to rank them by; "
                "the final model is teal and the model chosen in the sidebar is navy.")
    metric = st.radio("Rank the models by", list(METRIC_MEANING), index=4, horizontal=True)
    order = all_metrics[metric].sort_values()
    fig, ax = new_fig(8.5, 3.7)
    colors = [SORTED_GREEN if n == FINAL else POSTAL_BLUE if n == model_name else GREY_BAR for n in order.index]
    ax.barh([label_of(n) for n in order.index], order.values, color=colors, height=0.6)
    for i, v in enumerate(order.values):
        ax.text(v + 0.003, i, f"{v:.4f}", va="center", fontsize=10, color=INK)
    ax.set_xlim(max(0.0, np.nanmin(order.values) - 0.08), 1.0)
    ax.set_xlabel(f"{metric}: {METRIC_MEANING[metric].lower()}", color=MUTED)
    chart_title(ax, f"All five models ranked by {metric}", "test", len(y))
    style_axes(ax)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    table = all_metrics.copy()
    table["Mistakes"] = pd.Series(errors)
    table.index = [label_of(n) for n in table.index]

    def mark_best(col):
        best = col.min() if col.name == "Mistakes" else col.max()
        return ["background-color:#DCE6F5; font-weight:700" if v == best else "" for v in col]
    st.dataframe(table.style.format({**{m: "{:.4f}" for m in METRIC_MEANING}, "Mistakes": "{:.0f}"})
                 .apply(mark_best, axis=0))
    st.caption("Highlighted cells are the best value in each column.")

# ---- Classification report
with tab_report:
    st.markdown(f"The classification report scores **{model_name}** on each class separately: "
                f"how well it handles legitimate email, and how well it handles spam.")
    rep = classification_report(y, pred, labels=[0, 1], target_names=["Legitimate", "Spam"],
                                output_dict=True, zero_division=0)
    table = pd.DataFrame(rep).T.loc[["Legitimate", "Spam", "macro avg", "weighted avg"],
                                    ["precision", "recall", "f1-score", "support"]]
    table.columns = ["Precision", "Recall", "F1", "Emails"]
    table.index = ["Legitimate", "Spam", "Average of both classes", "Average weighted by size"]
    table["Emails"] = table["Emails"].astype(int)
    st.dataframe(table.style.format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}"}))
    st.markdown(
        f"In plain words: it kept **{tn} of {n_legit}** legitimate emails ({tn / max(n_legit, 1):.1%}) "
        f"and caught **{tp} of {n_spam}** spam ({tp / max(n_spam, 1):.1%}). When it says *spam* it is right "
        f"{tp / max(tp + fp, 1):.1%} of the time; when it says *legitimate* it is right "
        f"{tn / max(tn + fn, 1):.1%} of the time.")
    st.caption("Precision: when the model names this class, how often it is right. "
               "Recall: of all emails that really are this class, how many it found. F1: the balance of the two.")

# ---- Where it went wrong
with tab_err:
    wrong = np.flatnonzero(pred != y)
    st.markdown(f"A model can go wrong in two ways. Here are **{model_name}**'s {len(wrong)} mistakes "
                f"on these {len(y):,} emails.")
    st.markdown(f"""
<div class="cards2">
  <div class="mcard fa"><div class="n">{fp}</div><div class="h">good emails blocked</div>
    Real emails sent to the spam folder, so the user may never see them. This is usually the worse
    mistake for a mail filter.</div>
  <div class="mcard ms"><div class="n">{fn}</div><div class="h">spam emails let through</div>
    Spam that reached the inbox. Annoying, but the user can still see and delete it.</div>
</div>""", unsafe_allow_html=True)

    conf = np.abs(proba[wrong] - 0.5) * 2               # 0 = coin flip, 1 = completely sure
    close = conf < 0.4                                   # spam score between 0.3 and 0.7
    is_fa = y[wrong] == 0
    counts = pd.DataFrame({
        "close call (spam score 0.3 to 0.7)": [int((close & is_fa).sum()), int((close & ~is_fa).sum())],
        "confident but wrong": [int((~close & is_fa).sum()), int((~close & ~is_fa).sum())],
    }, index=["good emails blocked", "spam let through"])
    st.markdown("<p class='section-title'>How sure was it when it was wrong?</p>"
                "<p class='section-note'>Close calls are emails the model almost got right. Confident "
                "mistakes are emails that look convincingly like the other class.</p>", unsafe_allow_html=True)
    fig, ax = new_fig(8.5, 2.7)
    lefts = np.zeros(2)
    for col, colr in zip(counts.columns, ["#E8B4B8", AIRMAIL_RED]):
        ax.barh(counts.index, counts[col].values, left=lefts, color=colr, height=0.55, label=col)
        for i, (v, lft) in enumerate(zip(counts[col].values, lefts)):
            if v:
                ax.text(lft + v / 2, i, str(v), ha="center", va="center", color="white", fontweight="bold")
        lefts += counts[col].values
    ax.invert_yaxis()
    ax.set_xlabel("number of mistakes", color=MUTED)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=9.5)
    style_axes(ax)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown("<p class='section-title'>The mistakes, one by one</p>"
                "<p class='section-note'>Copy a row number into the <b>Inspect one email</b> tab "
                "to see what that email contained.</p>", unsafe_allow_html=True)
    show = st.radio("Show", ["All mistakes", "Only good emails blocked", "Only spam let through"], horizontal=True)
    mistakes = pd.DataFrame({
        "Email row": wrong,
        "Really was": np.where(y[wrong] == 1, "spam", "legitimate"),
        "Model said": np.where(pred[wrong] == 1, "spam", "legitimate"),
        "Spam score": np.round(proba[wrong], 3),
        "How sure": np.where(close, "close call", "confident"),
    })
    if show == "Only good emails blocked":
        mistakes = mistakes[mistakes["Really was"] == "legitimate"]
    elif show == "Only spam let through":
        mistakes = mistakes[mistakes["Really was"] == "spam"]
    st.dataframe(mistakes.reset_index(drop=True))

    out = pd.DataFrame({"email_row": np.arange(len(y)),
                        "actual": np.where(y == 1, "spam", "legitimate"),
                        "predicted": np.where(pred == 1, "spam", "legitimate"),
                        "spam_score": np.round(proba, 4), "correct": pred == y})
    st.download_button(f"Download all {model_name} predictions (CSV)", out.to_csv(index=False),
                       file_name=f"{model_name.lower().replace(' ', '_')}_predictions.csv", mime="text/csv")

# ---- Inspect one email
with tab_one:
    st.markdown("Pick any email by its row number and see how each of the five models judged it, "
                "and what the email contained.")
    first_wrong = int(np.flatnonzero(pred != y)[0]) if (pred != y).any() else 0
    idx = int(st.number_input(f"Email row (0 to {len(y) - 1})", min_value=0, max_value=len(y) - 1,
                              value=first_wrong, step=1))
    truth = "spam" if y[idx] == 1 else "legitimate"
    n_right = sum(int(runs[n]["pred"][idx] == y[idx]) for n in names)
    st.markdown(f"Row **{idx}** is actually **{truth}**. **{n_right} of {len(names)}** models got it right.")

    scores = [runs[n]["proba"][idx] for n in names]
    verdicts = [runs[n]["pred"][idx] for n in names]
    fig, ax = new_fig(8.5, 3.0)
    ax.barh([label_of(n) for n in names], scores,
            color=[SORTED_GREEN if v == y[idx] else AIRMAIL_RED for v in verdicts], height=0.55)
    ax.axvline(0.5, color=INK, ls="--", lw=1)
    for i, (s_, v) in enumerate(zip(scores, verdicts)):
        ax.text(min(s_ + 0.02, 0.8), i, f"{s_:.2f}  says {'spam' if v == 1 else 'legitimate'}",
                va="center", fontsize=10, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("spam score from each model (dashed line = 0.5). Teal = correct, coral = wrong", color=MUTED)
    ax.invert_yaxis()
    style_axes(ax)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    email = df.loc[idx, FEATURES].astype(float)
    present = email[(email > 0) & ~email.index.str.startswith("capital_")].sort_values(ascending=False)
    caps = email[email.index.str.startswith("capital_")]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Tracked words and characters in this email** (as % of its words or characters)")
        shown = present.head(12).rename(lambda s_: s_.replace("word_freq_", "word: ").replace("char_freq_", "character: "))
        if len(shown):
            st.dataframe(shown.to_frame("share %").style.format("{:.2f}"))
        else:
            st.caption("None of the 54 tracked words or characters appear in this email.")
    with c2:
        st.markdown("**Capital letters**")
        st.dataframe(caps.rename({"capital_run_length_average": "average run of capitals",
                                  "capital_run_length_longest": "longest run of capitals",
                                  "capital_run_length_total": "total capital letters"})
                     .to_frame("value").style.format("{:,.1f}"))

st.markdown(f"<div class='footer'>Built by {AUTHOR} for BITS Pilani WILP, M.Tech AIML, "
            "Machine Learning Assignment 2. Models were trained offline in the project notebook; "
            "this app loads them from <code>model/</code> and never retrains on uploaded data.</div>",
            unsafe_allow_html=True)
