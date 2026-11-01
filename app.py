"""
Kernel Post - five trained spam filters sorting email they have never seen
ML Assignment 2, M.Tech AIML, BITS Pilani WILP
Built by Naren (2026AJ05871)

The app only LOADS the models saved by the training notebook (model/*.pkl)
and scores the test emails. Nothing is retrained here.
"""
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
from scipy.stats import chi2
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

# Change the name here and it updates the whole app and the PDF report
APP_NAME = "Kernel Post"
APP_TAGLINE = "Spam detection with five classifiers, led by an RBF-kernel SVM"
AUTHOR = "Naren (2026AJ05871)"

HERE = Path(__file__).parent
MODEL_DIR = HERE / "model"
SAMPLE_CSV = HERE / "test_data.csv"

# ---------------------------------------------------------------- palette
PAPER, INK, MUTED = "#F3F5F8", "#1C2230", "#5E6779"
POSTAL_BLUE, AIRMAIL_RED, SORTED_GREEN = "#2350A0", "#C2303A", "#2E7D5B"
RULE, GREY_BAR = "#D5DBE4", "#B4BDCB"

MODEL_NOTES = {
    "Logistic Regression": "Weighs the evidence from every feature, adds it up and turns the total into a spam score.",
    "Decision Tree": "Asks a chain of yes/no questions about the email. Pruned so it learns rules, not individual emails.",
    "KNN": "Finds the most similar emails it saw in training and lets them vote; closer ones count more.",
    "Naive Bayes": "Treats each feature as independent evidence and multiplies the odds together.",
    "SVM": "Draws a curved boundary between spam and legitimate mail, using many features at once.",
}
POSTMARK_NAME = {"Logistic Regression": "LOGISTIC REGRESSION", "Decision Tree": "DECISION TREE",
                 "KNN": "NEAREST NEIGHBOURS", "Naive Bayes": "NAIVE BAYES", "SVM": "SUPPORT VECTOR MACHINE"}
METRIC_MEANING = {
    "Accuracy":  "Share of all emails sorted correctly",
    "AUC":       "How well spam is ranked above legitimate mail (1 = perfect, 0.5 = guessing)",
    "Precision": "When it says spam, how often it is right",
    "Recall":    "Share of the real spam it catches",
    "F1":        "Balance between precision and recall",
    "MCC":       "Overall agreement with the truth, fair even with uneven classes (-1 to 1)",
}

st.set_page_config(page_title=f"{APP_NAME} | Spam filter evaluation", page_icon="✉️", layout="wide")

# ---------------------------------------------------------------- styles
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@75..100,500..800&family=Source+Sans+3:wght@400;600;700&display=swap');

.block-container {{ padding-top: 1.6rem; max-width: 1250px; }}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{
    font-family: 'Source Sans 3', sans-serif; font-size: 1.02rem; line-height: 1.5; }}
h1, h2, h3 {{ font-family: 'Archivo', sans-serif !important; color: {INK}; letter-spacing: -0.01em; }}

/* ---- masthead with airmail band and postmark */
.airmail {{ height: 12px; border-radius: 2px; margin-bottom: 1.1rem;
    background: repeating-linear-gradient(135deg, {AIRMAIL_RED} 0 16px, {PAPER} 16px 26px,
                                                 {POSTAL_BLUE} 26px 42px, {PAPER} 42px 52px); }}
.masthead {{ display: flex; align-items: center; justify-content: space-between; gap: 1.5rem; }}
.masthead h1 {{ font-family: 'Archivo', sans-serif; font-stretch: 78%; font-weight: 800;
    font-size: 3.1rem; line-height: 1; margin: 0 0 .5rem 0; color: {INK}; }}
.masthead p {{ font-family: 'Source Sans 3', sans-serif; color: {MUTED}; font-size: 1.1rem;
    max-width: 60ch; margin: 0; }}
.postmark {{ flex-shrink: 0; transform: rotate(-9deg);
    animation: stamp .6s cubic-bezier(.2, .9, .3, 1.25) .15s both; }}
@keyframes stamp {{
    0%   {{ transform: rotate(-9deg) scale(1.7); opacity: 0; }}
    55%  {{ transform: rotate(-9deg) scale(.93); opacity: 1; }}
    100% {{ transform: rotate(-9deg) scale(1);   opacity: 1; }} }}
.postmark .arc {{ font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 11.5px;
    letter-spacing: .08em; fill: {POSTAL_BLUE}; }}
.postmark .num {{ font-family: 'Archivo', sans-serif; font-weight: 800; font-size: 34px; fill: {POSTAL_BLUE}; }}
@media (prefers-reduced-motion: reduce) {{ .postmark {{ animation: none; }} }}

/* ---- about panel */
.about {{ display: grid; grid-template-columns: 1.1fr 1fr; gap: 1.8rem; background: #FFFFFF;
    border: 1px solid {RULE}; border-top: 4px solid {POSTAL_BLUE}; border-radius: 8px;
    padding: 1.1rem 1.4rem 1.2rem; margin: 1.3rem 0 .3rem 0; font-family: 'Source Sans 3', sans-serif; color: {INK}; }}
.about h3 {{ margin: 0 0 .4rem 0; font-size: 1.2rem; }}
.about p {{ margin: 0 0 .55rem 0; line-height: 1.5; }}
.about ol {{ margin: 0; padding-left: 1.25rem; }}
.about li {{ margin: .15rem 0 .4rem 0; line-height: 1.45; }}
.about .key {{ display: flex; flex-wrap: wrap; gap: .3rem 1rem; margin-top: .4rem; font-size: .93rem; color: {MUTED}; }}
.about .sw {{ display: inline-block; width: .8rem; height: .8rem; border-radius: 2px;
    vertical-align: -1px; margin-right: .3rem; }}

/* ---- final model banner */
.verdict {{ display: grid; grid-template-columns: auto 1fr; gap: 1.2rem; align-items: center;
    background: #FFFFFF; border: 1px solid {RULE}; border-left: 8px solid {SORTED_GREEN};
    border-radius: 8px; padding: 1rem 1.3rem; margin: 1.2rem 0 .5rem 0; }}
.verdict .seal {{ font-family: 'Archivo', sans-serif; font-weight: 800; font-stretch: 80%;
    color: {SORTED_GREEN}; font-size: 1.05rem; line-height: 1.15; text-align: center;
    border: 2.5px solid {SORTED_GREEN}; border-radius: 6px; padding: .45rem .7rem; transform: rotate(-4deg); }}
.verdict .seal b {{ display: block; font-size: 1.9rem; }}
.verdict h3 {{ margin: 0 0 .3rem 0; font-size: 1.3rem; }}
.verdict ul {{ margin: 0; padding-left: 1.1rem; font-family: 'Source Sans 3', sans-serif; color: {INK}; }}
.verdict li {{ margin: .12rem 0; }}

.section-title {{ font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 1.35rem;
    color: {INK}; margin: 1.4rem 0 .2rem 0; }}
.section-note {{ font-family: 'Source Sans 3', sans-serif; color: {MUTED}; margin: 0 0 .8rem 0; }}

/* ---- confusion matrix drawn as four sorting trays */
.trays {{ display: grid; grid-template-columns: 8.5rem 1fr 1fr; gap: 6px; font-family: 'Source Sans 3', sans-serif; }}
.trays .head {{ font-weight: 600; color: {MUTED}; font-size: .92rem; align-self: end; padding: 0 .4rem .2rem; }}
.trays .side {{ font-weight: 600; color: {MUTED}; font-size: .92rem; align-self: center; }}
.tray {{ border-radius: 6px; padding: .9rem 1rem; min-height: 6.2rem; }}
.tray .n {{ display: block; font-family: 'Archivo', sans-serif; font-weight: 800; font-size: 2.3rem; line-height: 1; }}
.tray .t {{ display: block; margin-top: .35rem; font-size: .95rem; }}
.tray.ok  {{ background: #E3EFE9; color: #1F5A41; border-left: 5px solid {SORTED_GREEN}; }}
.tray.bad {{ background: #F7E4E5; color: #8E1F27; border-left: 5px solid {AIRMAIL_RED}; }}

/* ---- scorecard */
.score {{ display: grid; grid-template-columns: 6.2rem 4.6rem 1fr 5.6rem; gap: .35rem .8rem;
    align-items: center; font-family: 'Source Sans 3', sans-serif; }}
.score .m {{ font-weight: 700; color: {INK}; }}
.score .v {{ font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 1.15rem; color: {INK}; }}
.score .bar {{ background: {RULE}; border-radius: 3px; height: 9px; position: relative; }}
.score .bar span {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 3px; background: {POSTAL_BLUE}; }}
.score .r {{ color: {MUTED}; font-size: .9rem; text-align: right; white-space: nowrap; }}
.score .d {{ grid-column: 1 / -1; color: {MUTED}; font-size: .88rem; margin: -.2rem 0 .45rem 0; }}

/* ---- tabs styled as folder tabs so they read as clickable */
.explore {{ font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 1.35rem; color: {INK};
    margin: 2rem 0 .1rem 0; }}
.explore-note {{ font-family: 'Source Sans 3', sans-serif; color: {MUTED}; margin: 0 0 .6rem 0; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; flex-wrap: wrap; border-bottom: 2px solid {POSTAL_BLUE}; }}
.stTabs [data-baseweb="tab"] {{ height: auto; background: #E4E9F0; border: 1.5px solid {RULE};
    border-bottom: none; border-radius: 9px 9px 0 0; padding: .65rem 1.15rem; margin-bottom: 0; }}
.stTabs [data-baseweb="tab"] p {{ font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 1.02rem; color: {INK}; }}
.stTabs [data-baseweb="tab"]:hover {{ background: #FFFFFF; border-color: {POSTAL_BLUE}; }}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{ background: {POSTAL_BLUE}; border-color: {POSTAL_BLUE}; }}
.stTabs [data-baseweb="tab"][aria-selected="true"] p {{ color: #FFFFFF; }}
.stTabs [data-baseweb="tab"]:focus-visible {{ outline: 3px solid {AIRMAIL_RED}; outline-offset: 2px; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}
.stTabs [data-baseweb="tab-panel"] {{ background: #FFFFFF; border: 1px solid {RULE}; border-top: none;
    border-radius: 0 0 8px 8px; padding: 1.2rem 1.4rem 1.4rem; }}

/* ---- mistake cards */
.cards2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: .4rem 0 1rem 0; }}
.mcard {{ border-radius: 8px; padding: 1rem 1.2rem; font-family: 'Source Sans 3', sans-serif; }}
.mcard .n {{ font-family: 'Archivo', sans-serif; font-weight: 800; font-size: 2.6rem; line-height: 1; }}
.mcard .h {{ font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 1.1rem; margin: .2rem 0; }}
.mcard.fa {{ background: #F7E4E5; color: #8E1F27; }}
.mcard.ms {{ background: #FBEFD9; color: #7A4A06; }}

.profile {{ background: #FFFFFF; border: 1px solid {RULE}; border-radius: 6px; padding: .8rem .9rem;
    font-family: 'Source Sans 3', sans-serif; color: {INK}; font-size: .95rem; }}
.profile b {{ font-family: 'Archivo', sans-serif; }}
.footer {{ margin-top: 2.5rem; padding-top: .8rem; border-top: 1px solid {RULE};
    font-family: 'Source Sans 3', sans-serif; color: {MUTED}; font-size: .9rem; }}
@media (max-width: 760px) {{ .about, .cards2, .verdict {{ grid-template-columns: 1fr; }}
    .masthead {{ flex-wrap: wrap; }} }}
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


@st.cache_data(show_spinner="Sorting the emails with all five filters...")
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


def chart_title(ax, text):
    ax.set_title(text, loc="left", fontsize=12, fontweight="bold", color=INK, pad=10)


def draw_mistakes(ax, errors, final):
    order = pd.Series(errors).sort_values(ascending=False)
    ax.barh(order.index, order.values,
            color=[SORTED_GREEN if n == final else GREY_BAR for n in order.index], height=0.6)
    for i, v in enumerate(order.values):
        ax.text(v, i, f"  {v}", va="center", fontsize=10, color=INK)
    ax.set_xlim(0, order.max() * 1.15)
    ax.set_xlabel("mistakes on the test emails (lower is better)", color=MUTED)
    chart_title(ax, "Total mistakes")
    style_axes(ax)


def draw_metric(ax, all_metrics, metric, final, title):
    order = all_metrics[metric].sort_values()
    ax.barh(order.index, order.values,
            color=[SORTED_GREEN if n == final else GREY_BAR for n in order.index], height=0.6)
    for i, v in enumerate(order.values):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=10, color=INK)
    ax.set_xlim(max(0, order.min() - 0.1), 1.02)
    ax.set_xlabel(f"{metric} on the test emails (higher is better)", color=MUTED)
    chart_title(ax, title)
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
    ax.set_xlabel(f"mean F1 over {summary.get('cv_folds', 5)} folds (bar) and spread (whisker)", color=MUTED)
    chart_title(ax, "Step 1: cross-validation on the training emails")
    style_axes(ax)


def draw_head_to_head(ax, correct, final, names_):
    others = [n for n in names_ if n != final]
    lim = 1
    for i, n in enumerate(others):
        a, b, p = mcnemar(correct[final], correct[n])
        lim = max(lim, a, b)
        ax.barh(i, a, color=SORTED_GREEN, height=0.5)
        ax.barh(i, -b, color=GREY_BAR, height=0.5)
        ax.text(a, i, f"  {a}", va="center", fontweight="bold", color=INK)
        ax.text(-b, i, f"{b}  ", va="center", ha="right", fontweight="bold", color=INK)
        ax.text(0, i - 0.38, f"p = {p:.3f}: {'real difference' if p < 0.05 else 'could be chance'}",
                ha="center", va="bottom", fontsize=9.5, color=SORTED_GREEN if p < 0.05 else AIRMAIL_RED)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(range(len(others)))
    ax.set_yticklabels([f"{final} vs {n}" for n in others])
    ax.set_xlim(-lim * 1.35, lim * 1.35)
    ax.set_ylim(len(others) - 0.5, -0.9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{abs(v):.0f}"))
    ax.set_xlabel(f"emails only the other filter got right   |   emails only {final} got right", color=MUTED)
    chart_title(ax, "Step 3: head to head on the emails where two filters disagree")
    style_axes(ax)


def draw_gap(ax, summary, all_metrics, final):
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
    ax.set_xlabel("accuracy (filled = training emails, hollow = test emails)", color=MUTED)
    chart_title(ax, "Step 4: does it generalise? (shorter line = less overfitting)")
    style_axes(ax)


def draw_cost(ax, runs, final):
    ms = pd.Series({n: r["ms_per_1000"] for n, r in runs.items()}).sort_values()
    ax.barh(ms.index, ms.values, color=[SORTED_GREEN if n == final else GREY_BAR for n in ms.index], height=0.6)
    for i, v in enumerate(ms.values):
        ax.text(v, i, f"  {v:,.1f} ms", va="center", fontsize=10, color=INK)
    ax.set_xlim(0, ms.max() * 1.3)
    ax.set_xlabel("time to score 1,000 emails on this server (lower is faster)", color=MUTED)
    chart_title(ax, "Step 5: the cost - scoring speed")
    style_axes(ax)


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
                   f"and {runner} {l} times (McNemar p = {p_run:.3f}"
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
                                     color=POSTAL_BLUE, lw=0))
        fig.text(0.06, 0.88, f"Why {final} is the final model", fontsize=26, fontweight="bold", color=INK)
        fig.text(0.06, 0.83, f"{APP_NAME}: {APP_TAGLINE}. Report generated {date.today():%d %B %Y} "
                             f"from {len(y):,} test emails.", fontsize=11, color=MUTED)
        yy = 0.74
        fig.text(0.06, yy, "The evidence", fontsize=15, fontweight="bold", color=INK)
        yy -= 0.05
        for k, r in enumerate(reasons, 1):
            lines = textwrap.wrap(r, 115)
            fig.text(0.07, yy, f"{k}.", fontsize=11.5, fontweight="bold", color=SORTED_GREEN)
            for ln in lines:
                fig.text(0.095, yy, ln, fontsize=11.5, color=INK)
                yy -= 0.035
            yy -= 0.015
        yy -= 0.02
        fig.text(0.06, yy, "The trade-off", fontsize=15, fontweight="bold", color=INK)
        yy -= 0.05
        kb = {n: (MODEL_DIR / meta["model_files"][n]).stat().st_size / 1024 for n in names_}
        trade = (f"{final} is not the cheapest filter: its saved model is {kb[final]:,.0f} KB and it scores "
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
        draw_metric(fig.add_subplot(gs[1, 0]), all_m, "F1", final, "Step 2: F1 on the test emails")
        draw_metric(fig.add_subplot(gs[1, 1]), all_m, "MCC", final, "Step 2: MCC on the test emails")
        footer(fig, 2)
        pdf.savefig(fig); plt.close(fig)

        # page 3: confusion matrices + mistakes
        fig = plt.figure(figsize=A4); fig.patch.set_facecolor("white")
        fig.text(0.06, 0.92, "Step 2 (continued): where each filter put the test emails",
                 fontsize=14, fontweight="bold", color=INK)
        gs = fig.add_gridspec(2, 5, left=0.08, right=0.97, top=0.84, bottom=0.1, hspace=0.6, wspace=0.7)
        for k, n in enumerate(names_):
            ax = fig.add_subplot(gs[0, k] if k < 5 else gs[1, 0])
            draw_cm(ax, y, runs[n]["pred"], n, final)
        draw_mistakes(fig.add_subplot(gs[1, 1:4]), errs, final)
        footer(fig, 3)
        pdf.savefig(fig); plt.close(fig)

        # page 4: significance + generalisation
        fig = plt.figure(figsize=A4); fig.patch.set_facecolor("white")
        gs = fig.add_gridspec(2, 1, left=0.22, right=0.95, top=0.9, bottom=0.12, hspace=0.55)
        draw_head_to_head(fig.add_subplot(gs[0]), corr, final, names_)
        if summary and summary.get("train_accuracy"):
            draw_gap(fig.add_subplot(gs[1]), summary, all_m, final)
        else:
            draw_cost(fig.add_subplot(gs[1]), runs, final)
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


def postmark_svg(name, f1):
    return f"""
<svg class="postmark" viewBox="0 0 170 170" width="160" height="160" role="img"
     aria-label="{name}: F1 {f1:.3f} on this file">
  <defs>
    <path id="pm-top" d="M 27,85 a 58,58 0 1,1 116,0"/>
    <path id="pm-bot" d="M 22,85 a 63,63 0 0,0 126,0"/>
  </defs>
  <circle cx="85" cy="85" r="78" fill="none" stroke="{POSTAL_BLUE}" stroke-width="3"/>
  <circle cx="85" cy="85" r="47" fill="none" stroke="{POSTAL_BLUE}" stroke-width="1.3"/>
  <text class="arc"><textPath href="#pm-top" startOffset="50%" text-anchor="middle">{POSTMARK_NAME.get(name, name.upper())}</textPath></text>
  <text class="arc"><textPath href="#pm-bot" startOffset="50%" text-anchor="middle">F1 ON THIS FILE</textPath></text>
  <text class="num" x="85" y="97" text-anchor="middle">{f1:.3f}</text>
</svg>"""


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
        "Which emails should the filters sort?",
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

    st.markdown("### 2. Pick a filter to inspect")
    model_name = st.selectbox("Trained model", MODEL_NAMES, index=MODEL_NAMES.index(FINAL)
                              if FINAL in MODEL_NAMES else 0, format_func=label_of)
    fsize = (MODEL_DIR / meta["model_files"][model_name]).stat().st_size / 1024
    st.markdown(
        f"<div class='profile'><b>{label_of(model_name)}</b><br>{MODEL_NOTES.get(model_name, '')}"
        f"<br><br>Settings chosen by cross-validation: {best_params.get(model_name, '-')}"
        f"<br>Saved model size: {fsize:,.0f} KB</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- data
if source_choice == "Upload a CSV file":
    csv_bytes = uploaded.getvalue() if uploaded is not None else None
    source = uploaded.name if uploaded is not None else None
else:
    csv_bytes = SAMPLE_CSV.read_bytes() if SAMPLE_CSV.exists() else None
    source = "the test set included with this app (test_data.csv)"
result = sort_mail(csv_bytes) if csv_bytes is not None else None

# ---------------------------------------------------------------- masthead
f1_here = result[2]["runs"][model_name]["metrics"]["F1"] if result and result[2] else None
st.markdown("<div class='airmail'></div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='masthead'><div><h1>{APP_NAME}</h1>"
    f"<p>{APP_TAGLINE}, tested on emails the filters never saw during training.</p></div>"
    + (postmark_svg(model_name, f1_here) if f1_here is not None else "") + "</div>",
    unsafe_allow_html=True)

# ---------------------------------------------------------------- about this app
st.markdown(f"""
<div class="about">
  <div>
    <h3>About this app</h3>
    <p>{APP_NAME} checks how well five machine-learning spam filters sort email they have never seen.
    Each filter was trained in the project notebook on 80% of the
    <a href="https://archive.ics.uci.edu/dataset/94/spambase" target="_blank">UCI Spambase</a> emails,
    which describe each email with 57 numbers: how often certain words and characters appear,
    and how much of it is in capital letters.</p>
    <p>The remaining 20% were held back as test emails. This app loads the five saved filters,
    lets them sort those test emails, and measures how often they get it right. Nothing is retrained here.</p>
    <div class="key">
      <span><span class="sw" style="background:{SORTED_GREEN}"></span>correct, or the final model</span>
      <span><span class="sw" style="background:{AIRMAIL_RED}"></span>a mistake</span>
      <span><span class="sw" style="background:{POSTAL_BLUE}"></span>the filter you selected</span>
    </div>
  </div>
  <div>
    <h3>How to use it</h3>
    <ol>
      <li><b>Choose the emails</b> in the sidebar. The included test set loads automatically; to try the
          upload, pick <i>Upload a CSV file</i>, download <i>test_data.csv</i> and upload it.</li>
      <li><b>Pick a filter</b> in the sidebar. The confusion matrix, the metrics and the stamp at the top
          switch to that filter.</li>
      <li><b>Open the folder tabs</b> further down to see why {FINAL} was chosen, compare all five filters,
          read the classification report, review the mistakes or inspect a single email.</li>
      <li><b>Download the report</b> under the green banner for the full evidence as a PDF.</li>
    </ol>
  </div>
</div>""", unsafe_allow_html=True)

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

# ---------------------------------------------------------------- final model banner + report
runner, reasons = build_reasons(all_metrics, errors, correct, summary, FINAL, len(y))
st.markdown(f"""
<div class="verdict">
  <div class="seal">FINAL<b>{FINAL}</b>MODEL</div>
  <div><h3>Our final choice is {FINAL}</h3>
  <ul>{''.join(f'<li>{r}</li>' for r in reasons[:4])}</ul></div>
</div>""", unsafe_allow_html=True)
b1, b2 = st.columns([1, 2.2])
with b1:
    st.download_button(f"📄 Download the 'Why {FINAL}' report (PDF)", build_report_pdf(csv_bytes, FINAL),
                       file_name=f"why_{FINAL.lower().replace(' ', '_')}_report.pdf", mime="application/pdf")
with b2:
    st.markdown(f"<p class='section-note'>Scoring {source}: {len(y):,} emails, {n_legit:,} legitimate "
                f"and {n_spam:,} spam. The same evidence, with charts, is in the <b>Why {FINAL}?</b> "
                f"tab below.</p>", unsafe_allow_html=True)

# ---------------------------------------------------------------- selected model: matrix + metrics
run = runs[model_name]
pred, proba, met = run["pred"], run["proba"], run["metrics"]
tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

left, right = st.columns([1.05, 1], gap="large")
with left:
    st.markdown(f"<p class='section-title'>Confusion matrix: where {model_name} put each email</p>"
                f"<p class='section-note'>Green trays are correct. Red trays are mistakes: "
                f"{fp + fn} of {len(y):,} emails. Change the filter in the sidebar to compare.</p>",
                unsafe_allow_html=True)
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
    st.markdown(f"<p class='section-title'>Evaluation metrics for {model_name}</p>"
                "<p class='section-note'>Spam is the positive class. Rank compares this filter "
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
st.markdown("<p class='explore'>Explore the results</p>"
            "<p class='explore-note'>Click a folder tab to open it.</p>", unsafe_allow_html=True)
tab_why, tab_all, tab_report, tab_err, tab_one = st.tabs([
    f"🏆 Why {FINAL}?", "📊 Compare all five", "📋 Classification report",
    "⚠️ Where it went wrong", "🔍 Inspect one email"])

# ---- Why the final model
with tab_why:
    st.markdown(f"**How the final model was chosen.** All five filters were tuned with cross-validation on "
                f"the training emails only. The one with the best average F1 was chosen, and the choice was "
                f"then checked on the test emails, which no filter saw during training. The five steps below "
                f"are the evidence; the PDF report under the green banner contains the same charts.")

    if summary and summary.get("cv"):
        fig, ax = new_fig(10, 3.4); draw_cv(ax, summary, FINAL)
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

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = new_fig(6, 3.3); draw_metric(ax, all_metrics, "MCC", FINAL, "Step 2: overall quality (MCC)")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    with c2:
        fig, ax = new_fig(6, 3.3); draw_mistakes(ax, errors, FINAL)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown(f"**Step 3. Is the lead real, or luck?** We compare {FINAL} with each other filter only on the "
                f"emails where they disagree. If one wins those far more often than chance would allow "
                f"(p below 0.05), the difference is real.")
    fig, ax = new_fig(10, 0.75 * (len(names) - 1) + 1.3); draw_head_to_head(ax, correct, FINAL, names)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    c3, c4 = st.columns(2)
    with c3:
        if summary and summary.get("train_accuracy"):
            fig, ax = new_fig(6, 3.4); draw_gap(ax, summary, all_metrics, FINAL)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        else:
            st.caption("Step 4 (training vs test accuracy) is in Section 9 of the training notebook.")
    with c4:
        fig, ax = new_fig(6, 3.4); draw_cost(ax, runs, FINAL)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown(f"**The verdict.** {FINAL} wins on accuracy and on the mistake that matters most for email, "
                f"losing genuine messages to the spam folder. It is larger and slower than the simplest filters; "
                f"if speed or explainability mattered more than accuracy, {runner} would be the sensible alternative.")

# ---- Compare all five
with tab_all:
    st.markdown("Every saved model scored on the same emails. Choose a metric to rank them by; "
                "the final model is green and the filter selected in the sidebar is blue.")
    metric = st.radio("Rank the filters by", list(METRIC_MEANING), index=4, horizontal=True)
    order = all_metrics[metric].sort_values()
    fig, ax = new_fig(8.5, 3.3)
    colors = [SORTED_GREEN if n == FINAL else POSTAL_BLUE if n == model_name else GREY_BAR for n in order.index]
    ax.barh([label_of(n) for n in order.index], order.values, color=colors, height=0.6)
    for i, v in enumerate(order.values):
        ax.text(v + 0.003, i, f"{v:.4f}", va="center", fontsize=10, color=INK)
    ax.set_xlim(max(0.0, np.nanmin(order.values) - 0.08), 1.0)
    ax.set_xlabel(f"{metric}: {METRIC_MEANING[metric].lower()}", color=MUTED)
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
    st.markdown(f"A filter can go wrong in two ways. Here are **{model_name}**'s {len(wrong)} mistakes "
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
    st.markdown("Pick any email by its row number and see how each of the five filters judged it, "
                "and what the email contained.")
    first_wrong = int(np.flatnonzero(pred != y)[0]) if (pred != y).any() else 0
    idx = int(st.number_input(f"Email row (0 to {len(y) - 1})", min_value=0, max_value=len(y) - 1,
                              value=first_wrong, step=1))
    truth = "spam" if y[idx] == 1 else "legitimate"
    n_right = sum(int(runs[n]["pred"][idx] == y[idx]) for n in names)
    st.markdown(f"Row **{idx}** is actually **{truth}**. **{n_right} of {len(names)}** filters got it right.")

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
    ax.set_xlabel("spam score from each filter (dashed line = 0.5). Green = correct, red = wrong", color=MUTED)
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
