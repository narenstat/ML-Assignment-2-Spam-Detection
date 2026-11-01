"""
Sorting Desk - how five trained spam filters sort unseen email
ML Assignment 2, M.Tech AIML, BITS Pilani WILP
Built by Naren (2026AJ05871)

The app only LOADS the models saved by the training notebook (model/*.pkl)
and scores the uploaded test file. Nothing is retrained here.
"""
import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

HERE = Path(__file__).parent
MODEL_DIR = HERE / "model"
SAMPLE_CSV = HERE / "test_data.csv"

# ---------------------------------------------------------------- palette
PAPER, INK, MUTED = "#F3F5F8", "#1C2230", "#5E6779"
POSTAL_BLUE, AIRMAIL_RED, SORTED_GREEN = "#2350A0", "#C2303A", "#2E7D5B"
RULE = "#D5DBE4"

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

st.set_page_config(page_title="Sorting Desk | Spam filter evaluation", page_icon="✉️", layout="wide")

# ---------------------------------------------------------------- styles
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@75..100,500..800&family=Source+Sans+3:wght@400;600;700&display=swap');

.block-container {{ padding-top: 1.6rem; max-width: 1250px; }}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{
    font-family: 'Source Sans 3', sans-serif; font-size: 1.02rem; line-height: 1.5; }}
h1, h2, h3 {{ font-family: 'Archivo', sans-serif !important; color: {INK}; letter-spacing: -0.01em; }}

.airmail {{ height: 12px; border-radius: 2px; margin-bottom: 1.1rem;
    background: repeating-linear-gradient(135deg, {AIRMAIL_RED} 0 16px, {PAPER} 16px 26px,
                                                 {POSTAL_BLUE} 26px 42px, {PAPER} 42px 52px); }}
.masthead {{ display: flex; align-items: center; justify-content: space-between; gap: 1.5rem; }}
.masthead h1 {{ font-family: 'Archivo', sans-serif; font-stretch: 78%; font-weight: 800;
    font-size: 3.1rem; line-height: 1; margin: 0 0 .5rem 0; color: {INK}; }}
.masthead p {{ font-family: 'Source Sans 3', sans-serif; color: {MUTED}; font-size: 1.1rem;
    max-width: 58ch; margin: 0; }}
.postmark {{ transform: rotate(-9deg); flex-shrink: 0; }}
.postmark .arc {{ font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 11.5px;
    letter-spacing: .08em; fill: {POSTAL_BLUE}; }}
.postmark .num {{ font-family: 'Archivo', sans-serif; font-weight: 800; font-size: 34px; fill: {POSTAL_BLUE}; }}

.section-title {{ font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 1.35rem;
    color: {INK}; margin: 1.4rem 0 .2rem 0; }}
.section-note {{ font-family: 'Source Sans 3', sans-serif; color: {MUTED}; margin: 0 0 .8rem 0; }}

/* confusion matrix drawn as four sorting trays */
.trays {{ display: grid; grid-template-columns: 8.5rem 1fr 1fr; gap: 6px; font-family: 'Source Sans 3', sans-serif; }}
.trays .head {{ font-weight: 600; color: {MUTED}; font-size: .92rem; align-self: end; padding: 0 .4rem .2rem; }}
.trays .side {{ font-weight: 600; color: {MUTED}; font-size: .92rem; align-self: center; }}
.tray {{ border-radius: 6px; padding: .9rem 1rem; min-height: 6.2rem; }}
.tray .n {{ display: block; font-family: 'Archivo', sans-serif; font-weight: 800; font-size: 2.3rem; line-height: 1; }}
.tray .t {{ display: block; margin-top: .35rem; font-size: .95rem; }}
.tray.ok  {{ background: #E3EFE9; color: #1F5A41; border-left: 5px solid {SORTED_GREEN}; }}
.tray.bad {{ background: #F7E4E5; color: #8E1F27; border-left: 5px solid {AIRMAIL_RED}; }}

/* scorecard */
.score {{ display: grid; grid-template-columns: 6.2rem 4.6rem 1fr 5.6rem; gap: .35rem .8rem;
    align-items: center; font-family: 'Source Sans 3', sans-serif; }}
.score .m {{ font-weight: 700; color: {INK}; }}
.score .v {{ font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 1.15rem; color: {INK}; }}
.score .bar {{ background: {RULE}; border-radius: 3px; height: 9px; position: relative; }}
.score .bar span {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 3px; background: {POSTAL_BLUE}; }}
.score .r {{ color: {MUTED}; font-size: .9rem; text-align: right; white-space: nowrap; }}
.score .d {{ grid-column: 1 / -1; color: {MUTED}; font-size: .88rem; margin: -.2rem 0 .45rem 0; }}

.profile {{ background: #FFFFFF; border: 1px solid {RULE}; border-radius: 6px; padding: .8rem .9rem;
    font-family: 'Source Sans 3', sans-serif; color: {INK}; font-size: .95rem; }}
.profile b {{ font-family: 'Archivo', sans-serif; }}
.footer {{ margin-top: 2.5rem; padding-top: .8rem; border-top: 1px solid {RULE};
    font-family: 'Source Sans 3', sans-serif; color: {MUTED}; font-size: .9rem; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_bundle():
    with open(MODEL_DIR / "metadata.json") as f:
        meta = json.load(f)
    models = {name: joblib.load(MODEL_DIR / fname) for name, fname in meta["model_files"].items()}
    return meta, models


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
    meta, models = load_bundle()
    df = pd.read_csv(io.BytesIO(csv_bytes))
    problems = check_file(df, meta["features"], meta["target"])
    if problems:
        return df, problems, None
    X = df[meta["features"]].astype(float)
    y = df[meta["target"]].astype(int).to_numpy()
    runs = {}
    for name, model in models.items():
        pred = model.predict(X)
        proba = model.predict_proba(X)[:, 1]
        runs[name] = {"pred": pred, "proba": proba, "metrics": score_all_metrics(y, pred, proba)}
    return df, [], {"y": y, "runs": runs}


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
  <text class="arc"><textPath href="#pm-top" startOffset="50%" text-anchor="middle">{POSTMARK_NAME[name]}</textPath></text>
  <text class="arc"><textPath href="#pm-bot" startOffset="50%" text-anchor="middle">F1 ON THIS FILE</textPath></text>
  <text class="num" x="85" y="97" text-anchor="middle">{f1:.3f}</text>
</svg>"""


def style_axes(ax):
    ax.set_facecolor("white")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=INK, labelsize=10)
    ax.grid(axis="x", color=RULE, lw=0.8)
    ax.set_axisbelow(True)


meta, models = load_bundle()
FEATURES, TARGET = meta["features"], meta["target"]
MODEL_NAMES = list(meta["model_files"].keys())
best_params = meta.get("best_params", {})

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 1. Bring the test emails")
    uploaded = st.file_uploader("Upload test_data.csv (57 feature columns plus 'spam')", type=["csv"])
    use_sample = st.checkbox("Use the test file stored in this repo", value=False,
                             disabled=uploaded is not None)
    if SAMPLE_CSV.exists():
        st.download_button("Download test_data.csv", SAMPLE_CSV.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")

    st.markdown("### 2. Pick a filter")
    model_name = st.selectbox("Trained model", MODEL_NAMES, index=MODEL_NAMES.index("SVM")
                              if "SVM" in MODEL_NAMES else 0)
    fsize = (MODEL_DIR / meta["model_files"][model_name]).stat().st_size / 1024
    params = best_params.get(model_name, "-")
    st.markdown(
        f"<div class='profile'><b>{model_name}</b><br>{MODEL_NOTES.get(model_name, '')}"
        f"<br><br>Settings chosen by cross-validation: {params}<br>Saved model size: {fsize:,.0f} KB</div>",
        unsafe_allow_html=True)

# ---------------------------------------------------------------- data
csv_bytes, source = None, None
if uploaded is not None:
    csv_bytes, source = uploaded.getvalue(), uploaded.name
elif use_sample and SAMPLE_CSV.exists():
    csv_bytes, source = SAMPLE_CSV.read_bytes(), "test_data.csv (from the repo)"

result = None
if csv_bytes is not None:
    df, problems, result = sort_mail(csv_bytes)

# ---------------------------------------------------------------- masthead
f1_here = result["runs"][model_name]["metrics"]["F1"] if result else None
st.markdown("<div class='airmail'></div>", unsafe_allow_html=True)
st.markdown(
    "<div class='masthead'><div><h1>Sorting Desk</h1>"
    "<p>Five spam filters trained on the UCI Spambase emails, tested here on emails they never saw. "
    "Upload the test file, pick a filter, and see how it sorts the mail.</p></div>"
    + (postmark_svg(model_name, f1_here) if f1_here is not None else "") + "</div>",
    unsafe_allow_html=True)

if csv_bytes is None:
    st.markdown("<p class='section-title'>Nothing to sort yet</p>"
                "<p class='section-note'>Upload <b>test_data.csv</b> in the sidebar, or tick "
                "<b>Use the test file stored in this repo</b>. The file needs the 57 Spambase feature "
                "columns and a <b>spam</b> column (1 = spam, 0 = legitimate). The models are already "
                "trained; the app only scores the file.</p>", unsafe_allow_html=True)
    st.stop()

if problems:
    st.error("This file can't be scored yet:\n\n- " + "\n- ".join(problems))
    st.stop()

y = result["y"]
run = result["runs"][model_name]
pred, proba, met = run["pred"], run["proba"], run["metrics"]
all_metrics = pd.DataFrame({n: r["metrics"] for n, r in result["runs"].items()}).T
n_spam, n_legit = int(y.sum()), int((y == 0).sum())

st.markdown(f"<p class='section-note'>Scoring <b>{source}</b>: {len(y):,} emails, "
            f"{n_legit:,} legitimate and {n_spam:,} spam.</p>", unsafe_allow_html=True)

# ---------------------------------------------------------------- confusion matrix + metrics
left, right = st.columns([1.05, 1], gap="large")

with left:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    st.markdown(f"<p class='section-title'>Confusion matrix: where {model_name} put each email</p>"
                f"<p class='section-note'>Green trays are correct. Red trays are mistakes: "
                f"{fp + fn} of {len(y):,} emails.</p>", unsafe_allow_html=True)
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
                "<p class='section-note'>Spam is the positive class. Rank compares this filter "
                "with the other four on the same file.</p>", unsafe_allow_html=True)
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
                    f"<div class='r'>rank {rank} of 5</div><div class='d'>{METRIC_MEANING[m]}</div>")
    st.markdown("<div class='score'>" + "".join(rows) + "</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- tabs
tab_report, tab_all, tab_one, tab_err = st.tabs(
    ["Classification report", "All five filters compared", "Inspect one email", "Mistakes"])

with tab_report:
    rep = classification_report(y, pred, labels=[0, 1], target_names=["Legitimate", "Spam"],
                                output_dict=True, zero_division=0)
    table = pd.DataFrame(rep).T.loc[["Legitimate", "Spam", "macro avg", "weighted avg"],
                                    ["precision", "recall", "f1-score", "support"]]
    table.columns = ["Precision", "Recall", "F1", "Emails"]
    table.index = ["Legitimate", "Spam", "Average of both classes", "Average weighted by size"]
    table["Emails"] = table["Emails"].astype(int)
    st.dataframe(table.style.format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}"}))
    st.markdown(
        f"In plain words: {model_name} kept **{tn} of {n_legit}** legitimate emails "
        f"({tn / max(n_legit, 1):.1%}) and caught **{tp} of {n_spam}** spam ({tp / max(n_spam, 1):.1%}). "
        f"When it says *spam* it is right {tp / max(tp + fp, 1):.1%} of the time; when it says "
        f"*legitimate* it is right {tn / max(tn + fn, 1):.1%} of the time.")

with tab_all:
    st.markdown("Every saved model scored on the same file. The selected filter is drawn in blue.")
    metric = st.radio("Rank the filters by", list(METRIC_MEANING), index=4, horizontal=True)
    order = all_metrics[metric].sort_values()
    fig, ax = plt.subplots(figsize=(8.5, 3.3))
    fig.patch.set_facecolor(PAPER)
    colors = [POSTAL_BLUE if n == model_name else "#AEB8C7" for n in order.index]
    ax.barh(order.index, order.values, color=colors, height=0.6)
    for i, v in enumerate(order.values):
        ax.text(v + 0.003, i, f"{v:.4f}", va="center", fontsize=10, color=INK)
    lo = max(0.0, np.nanmin(order.values) - 0.08)
    ax.set_xlim(lo, 1.0)
    ax.set_xlabel(f"{metric}  ({METRIC_MEANING[metric].lower()})", color=MUTED)
    style_axes(ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    def mark_best(col):
        best = col.max()
        return [f"background-color:#DCE6F5; font-weight:700" if v == best else "" for v in col]
    st.dataframe(all_metrics.style.format("{:.4f}").apply(mark_best, axis=0))
    best = all_metrics[metric].idxmax()
    errors = {n: int((r["pred"] != y).sum()) for n, r in result["runs"].items()}
    st.markdown(f"On this file **{best}** has the best {metric} ({all_metrics.loc[best, metric]:.4f}). "
                f"Total mistakes per filter: " +
                ", ".join(f"{n} {e}" for n, e in sorted(errors.items(), key=lambda kv: kv[1])) + ".")

with tab_one:
    st.markdown("Pick any email from the file and see how each filter judged it.")
    idx = st.number_input(f"Email row (0 to {len(y) - 1})", min_value=0, max_value=len(y) - 1,
                          value=int(np.flatnonzero(pred != y)[0]) if (pred != y).any() else 0, step=1)
    truth = "spam" if y[idx] == 1 else "legitimate"
    st.markdown(f"Row **{idx}** is actually **{truth}**.")

    names = list(result["runs"])
    scores = [result["runs"][n]["proba"][idx] for n in names]
    verdicts = [result["runs"][n]["pred"][idx] for n in names]
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    fig.patch.set_facecolor(PAPER)
    bar_col = [SORTED_GREEN if v == y[idx] else AIRMAIL_RED for v in verdicts]
    ax.barh(names, scores, color=bar_col, height=0.55)
    ax.axvline(0.5, color=INK, ls="--", lw=1)
    for i, (s, v) in enumerate(zip(scores, verdicts)):
        ax.text(min(s + 0.02, 0.82), i, f"{s:.2f}  says {'spam' if v == 1 else 'legitimate'}",
                va="center", fontsize=10, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("spam score from each filter (dashed line = 0.5)", color=MUTED)
    ax.invert_yaxis()
    style_axes(ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("Green bars are correct verdicts, red bars are wrong.")

    email = df.loc[idx, FEATURES].astype(float)
    present = email[(email > 0) & ~email.index.str.startswith("capital_")].sort_values(ascending=False)
    caps = email[email.index.str.startswith("capital_")]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Words and characters present** (percentage of the email)")
        shown = present.head(12).rename(lambda s: s.replace("word_freq_", "word: ").replace("char_freq_", "char: "))
        st.dataframe(shown.to_frame("value").style.format("{:.2f}") if len(shown) else
                     pd.DataFrame({"value": ["none of the tracked words"]}))
    with c2:
        st.markdown("**Capital letters**")
        st.dataframe(caps.rename({"capital_run_length_average": "average run of capitals",
                                  "capital_run_length_longest": "longest run of capitals",
                                  "capital_run_length_total": "total capitals"}).to_frame("value")
                     .style.format("{:,.1f}"))

with tab_err:
    wrong = np.flatnonzero(pred != y)
    kind = np.where(y[wrong] == 0, "false alarm (legitimate sent to spam)", "missed spam (reached inbox)")
    st.markdown(f"{model_name} got **{len(wrong)}** emails wrong. Scores near 0.5 were close calls; "
                f"scores near 0 or 1 were confident mistakes.")
    show = st.radio("Show", ["All mistakes", "Only false alarms", "Only missed spam"], horizontal=True)
    mistakes = pd.DataFrame({"row": wrong, "mistake": kind, "spam score": np.round(proba[wrong], 4)})
    if show == "Only false alarms":
        mistakes = mistakes[mistakes["mistake"].str.startswith("false")]
    elif show == "Only missed spam":
        mistakes = mistakes[mistakes["mistake"].str.startswith("missed")]
    st.dataframe(mistakes.sort_values("spam score").reset_index(drop=True))

    out = pd.DataFrame({"row": np.arange(len(y)),
                        "actual": np.where(y == 1, "spam", "legitimate"),
                        "predicted": np.where(pred == 1, "spam", "legitimate"),
                        "spam_score": np.round(proba, 4)})
    st.download_button(f"Download all {model_name} predictions (CSV)", out.to_csv(index=False),
                       file_name=f"sorting_desk_{model_name.lower().replace(' ', '_')}.csv", mime="text/csv")

st.markdown("<div class='footer'>Built by Naren (2026AJ05871) for BITS Pilani WILP, M.Tech AIML, "
            "Machine Learning Assignment 2. Models were trained offline in the project notebook; "
            "this app loads them from <code>model/</code> and never retrains on uploaded data.</div>",
            unsafe_allow_html=True)
