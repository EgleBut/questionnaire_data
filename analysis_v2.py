"""
GenAI use and computing students' ratings of factors that encourage active lecture participation.
Kaunas University of Technology, Faculty of Informatics, 2025-2026 (repeated cross-sectional survey).

Version 2 (revision after peer review). Reproduces all analyses reported in the manuscript and adds:
  - sensitivity of the equivalence conclusions to the choice of equivalence bounds   (Table A3)
  - sample-specific Spearman correlations with GenAI use frequency                    (Table A4)
  - ordinal regression with categorical coding of GenAI use frequency and
    predicted probabilities across frequency categories                               (Table A5, Figure A1)
  - original Lithuanian item wording                                                  (Table A6)
  - revised Figure 1 (mean ratings with 95% CIs, labels clear of the CI whiskers)

Data: analysis_data.csv (https://github.com/EgleBut/questionnaire_data).


Bootstrap confidence intervals for Cliff's delta are percentile intervals based on 5000 resamples;
respondents are resampled with replacement independently within each of the two compared groups.
The random seed is fixed (2026), so all results are reproducible.
"""
import argparse
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.miscmodels.ordinal_model import OrderedModel

GITHUB_CSV = "https://raw.githubusercontent.com/EgleBut/questionnaire_data/main/analysis_data.csv"
HERE = Path(__file__).resolve().parent

SEED = 2026
N_BOOT = 5000           # bootstrap resamples for Cliff's delta CIs
N_BOOT_PRED = 1000      # bootstrap resamples for predicted probabilities
RNG = np.random.default_rng(SEED)

SAMPLE_MAP = {"PS_2025": "PS 2025", "PS_2026": "PS 2026", "IF_other_2026": "Other IF 2026"}
SAMPLES = list(SAMPLE_MAP.values())

# key, CSV column, English wording (Table 1), short label, Lithuanian wording (original questionnaire)
ITEMS = [
    ("structured", "structured_information", "Clearly and structurally presented information",
     "Structured information", "Aiškiai ir struktūrizuotai pateikiama informacija"),
    ("quizzes", "interactive_quizzes", "Interactive questionnaires and short quizzes",
     "Interactive quizzes", "Interaktyvūs klausimynai ar mini testai paskaitos metu"),
    ("group", "group_work", "Group work",
     "Group work", "Darbas grupėse"),
    ("gamification", "gamification", "Gamification elements",
     "Gamification", "Sužaidybintai pateikta mokymosi medžiaga (pvz., žaidimų elementai, taškai, lygiai)"),
    ("informal", "informal_environment", "Informal learning environment",
     "Informal learning environment", "Neformali mokymosi aplinka"),
    ("social", "social_media", "Use of social media in the learning process",
     "Social media", "Socialinių tinklų įtraukimas į mokymosi procesą (tiktok, instagram ir pan.)"),
    ("ai", "ai_in_learning", "Use of AI tools in the learning process",
     "AI tools", "Dirbtinio intelekto įrankių panaudojimas mokymosi procese"),
    ("practical", "practical_examples", "Analysis of real-life situations and practical examples",
     "Practical examples", "Realių praktinių pavyzdžių ar situacijų analizė"),
    ("vrar", "vr_ar_virtual_labs", "Use of VR/AR tools and virtual laboratories",
     "VR/AR and virtual laboratories", "Galimybė naudoti modernias technologijas (VR, AR, virtualias laboratorijas)"),
    ("peer", "peer_support", "Peer assistance and consultation",
     "Peer support", "Kitų studentų pagalba/konsultacijos"),
]
KEYS = [i[0] for i in ITEMS]
COL = {i[0]: i[1] for i in ITEMS}
LABEL = {i[0]: i[2] for i in ITEMS}
SHORT = {i[0]: i[3] for i in ITEMS}
LT = {i[0]: i[4] for i in ITEMS}

STEM_LT = "Kas paskatintų jus aktyviau dalyvauti paskaitoje?"
STEM_EN = "What would encourage you to participate more actively in lectures?"
FREQ_ITEM_LT = "Kaip dažnai naudojatės DI įrankiais (pvz. ChatGPT, Gemini, Microsoft Copilot) studijoms?"
FREQ_ITEM_EN = "How often do you use AI tools (e.g., ChatGPT, Gemini, Microsoft Copilot) for your studies?"
FREQ_LT = ["Niekada", "Kartą per mėnesį ar rečiau", "Kelis kartus per mėnesį",
           "Kelis kartus per savaitę", "Kasdien arba beveik kasdien"]
FREQ_LABELS = ["Never", "Once a month or less", "Several times a month",
               "Several times a week", "Daily or almost daily"]

# Equivalence bounds used in the manuscript and the alternative bounds examined in the sensitivity analysis
DELTA_EQ, RHO_EQ = 0.33, 0.20
DELTA_BOUNDS = [0.147, 0.20, 0.33]   # negligible/small threshold, intermediate, small/medium threshold
RHO_BOUNDS = [0.10, 0.15, 0.20]

BLUE = "#256abf"


# ---------- data -------------------------------------------------------------

def load(source):
    raw = pd.read_csv(source, encoding="utf-8-sig")
    data = pd.DataFrame({k: raw[COL[k]].astype(int) for k in KEYS})
    data["freq"] = raw["genai_use_frequency_num"].astype(int)
    data["sample"] = pd.Categorical(raw["sample"].map(SAMPLE_MAP), categories=SAMPLES)
    assert data["sample"].notna().all() and data[KEYS].isin(range(1, 6)).all().all()
    return data


# ---------- effect sizes and helpers -------------------------------------------

def rank_biserial_paired(x, y):
    """Matched-pairs rank-biserial r (Kerby, 2014), zero differences dropped."""
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[d != 0]
    ranks = stats.rankdata(np.abs(d))
    rp, rn = ranks[d > 0].sum(), ranks[d < 0].sum()
    return (rp - rn) / (rp + rn)


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.sign(a[:, None] - b[None, :]).mean()


def cliffs_delta_ci(a, b, level=0.90):
    """Percentile bootstrap CI; respondents resampled independently within each group."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        boots[i] = cliffs_delta(RNG.choice(a, a.size), RNG.choice(b, b.size))
    return tuple(np.percentile(boots, [(1 - level) / 2 * 100, (1 + level) / 2 * 100]))


def spearman_ci(rho, n, level=0.90):
    """Fisher-z CI with the Bonett & Wright (2000) standard error for Spearman's rho."""
    z, se = np.arctanh(rho), np.sqrt((1 + rho ** 2 / 2) / (n - 3))
    q = stats.norm.ppf((1 + level) / 2)
    return np.tanh(z - q * se), np.tanh(z + q * se)


def holm(p):
    return multipletests(p, method="holm")[1]


def fmt_p(p):
    return "<.001" if p < .001 else f"{p:.3f}".lstrip("0") if p < 1 else "1.000"


def _r(v, nd=3):
    s = f"{v:.{nd}f}"
    return (s.replace("0.", ".", 1) if abs(v) < 1 else s).replace("-", "−")


def _ci(lo, hi, nd=3):
    return f"[{_r(lo, nd)}, {_r(hi, nd)}]"


def mean_sd_median_iqr(x):
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return f"{x.mean():.2f} ({x.std(ddof=1):.2f}); {med:.0f} [{q1:.0f}–{q3:.0f}]"


# ---------- RQ1 ------------------------------------------------------------------

def table2_descriptives(data):
    rows = []
    for i, k in enumerate(KEYS, 1):
        row = {"No.": i, "Item": LABEL[k]}
        for s in SAMPLES:
            row[f"{s} (n={(data['sample'] == s).sum()})"] = mean_sd_median_iqr(data.loc[data["sample"] == s, k])
        row[f"Overall (n={len(data)})"] = mean_sd_median_iqr(data[k])
        rows.append(row)
    return pd.DataFrame(rows)


def rq1_friedman(data):
    chi2, p = stats.friedmanchisquare(*[data[k] for k in KEYS])
    n, k = len(data), len(KEYS)
    return {"chi2": chi2, "df": k - 1, "p": p, "W": chi2 / (n * (k - 1)), "n": n}


def table3_ai_vs_others(data):
    rows = []
    for k in KEYS:
        if k == "ai":
            continue
        _, p = stats.wilcoxon(data["ai"], data[k], zero_method="wilcox")
        rows.append({"Comparison": f"AI tools vs. {SHORT[k]}", "AI M": data["ai"].mean(),
                     "Comparison M": data[k].mean(), "Mean difference": data["ai"].mean() - data[k].mean(),
                     "Rank-biserial r": rank_biserial_paired(data["ai"], data[k]), "p": p})
    t = pd.DataFrame(rows)
    t["Holm-adjusted p"] = holm(t["p"])
    return t.sort_values("Mean difference").reset_index(drop=True)


# ---------- RQ2 / RQ3 -------------------------------------------------------------

def group_comparison(data, g1, g2):
    """Differences g2 - g1: Mann-Whitney U, Cliff's delta with 90% bootstrap CI (TOST logic)."""
    a, b = data[data["sample"] == g1], data[data["sample"] == g2]
    rows = []
    for k in KEYS:
        _, p = stats.mannwhitneyu(b[k], a[k], alternative="two-sided")
        lo, hi = cliffs_delta_ci(b[k], a[k], 0.90)
        rows.append({"key": k, "Item": SHORT[k], "ΔM": b[k].mean() - a[k].mean(),
                     "Cliff's δ": cliffs_delta(b[k], a[k]), "90% CI low": lo, "90% CI high": hi, "p": p})
    t = pd.DataFrame(rows)
    t["Holm-adjusted p"] = holm(t["p"])
    t["Smallest equivalence bound"] = t[["90% CI low", "90% CI high"]].abs().max(axis=1)
    return t, stats.spearmanr(a[KEYS].mean(), b[KEYS].mean())[0]


def table_freq_distribution(data):
    rows = []
    for code, lab in enumerate(FREQ_LABELS):
        row = {"GenAI use frequency": lab}
        for s in SAMPLES + ["Overall"]:
            sub = data["freq"] if s == "Overall" else data.loc[data["sample"] == s, "freq"]
            n = (sub == code).sum()
            row[s] = f"{n} ({n / len(sub) * 100:.1f}%)"
        rows.append(row)
    for label in ["Weekly or daily use", "Median [Q1–Q3] (0–4)"]:
        row = {"GenAI use frequency": label}
        for s in SAMPLES + ["Overall"]:
            sub = data["freq"] if s == "Overall" else data.loc[data["sample"] == s, "freq"]
            if label.startswith("Weekly"):
                n = (sub >= 3).sum()
                row[s] = f"{n} ({n / len(sub) * 100:.1f}%)"
            else:
                q1, med, q3 = np.percentile(sub, [25, 50, 75])
                row[s] = f"{med:.0f} [{q1:.0f}–{q3:.0f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def freq_tests(data):
    out = []
    for g1, g2, lab in [("PS 2025", "PS 2026", "PS 2026 vs. PS 2025"),
                        ("PS 2026", "Other IF 2026", "Other IF 2026 vs. PS 2026")]:
        a, b = data.loc[data["sample"] == g1, "freq"], data.loc[data["sample"] == g2, "freq"]
        u, p = stats.mannwhitneyu(b, a, alternative="two-sided")
        lo, hi = cliffs_delta_ci(b, a, 0.95)
        out.append({"Comparison": lab, "U": u, "Cliff's δ": cliffs_delta(b, a),
                    "95% CI low": lo, "95% CI high": hi, "p": p})
    t = pd.DataFrame(out)
    t["Holm-adjusted p"] = holm(t["p"])
    h, p_kw = stats.kruskal(*[data.loc[data["sample"] == s, "freq"] for s in SAMPLES])
    return t, (h, p_kw)


# ---------- RQ4 ------------------------------------------------------------------

def correlations(data, level=0.90):
    n = len(data)
    rows = []
    for k in KEYS:
        rho, p = stats.spearmanr(data["freq"], data[k])
        lo, hi = spearman_ci(rho, n, level)
        rows.append({"key": k, "Item": SHORT[k], "ρ": rho, "90% CI low": lo, "90% CI high": hi, "p": p, "n": n})
    t = pd.DataFrame(rows)
    t["Holm-adjusted p"] = holm(t["p"])
    t["Smallest equivalence bound"] = t[["90% CI low", "90% CI high"]].abs().max(axis=1)
    return t


def sample_specific_correlations(data):
    return {s: correlations(data[data["sample"] == s]) for s in SAMPLES}


def relative_ai(data):
    """AI rating relative to the respondent's mean rating of the other nine items."""
    others = data[[k for k in KEYS if k != "ai"]].mean(axis=1)
    rel = data["ai"] - others
    r1, p1 = stats.spearmanr(data["freq"], rel)
    r2, p2 = stats.spearmanr(data["freq"], others)
    return {"rho_relative": r1, "p_relative": p1, "rho_others_mean": r2, "p_others_mean": p2,
            "relative_mean": rel.mean(), "relative_sd": rel.std(ddof=1)}


def _design(d, freq_cols):
    X = d[freq_cols].copy()
    X["s26"] = (d["sample"] == "PS 2026").astype(float)
    X["if26"] = (d["sample"] == "Other IF 2026").astype(float)
    return X


def _fit(y, X):
    return OrderedModel(pd.Categorical(y, ordered=True), X, distr="logit").fit(
        method="bfgs", disp=False, maxiter=5000)


def _or_row(name, est, se):
    return {"Term": name, "OR": np.exp(est), "95% CI low": np.exp(est - 1.96 * se),
            "95% CI high": np.exp(est + 1.96 * se), "p": 2 * stats.norm.sf(abs(est / se))}


def ordinal_models(data):
    """Manuscript models: GenAI frequency coded 0-4 (linear), adjusted for sample; interaction model."""
    d = data.copy()
    d["freq_s26"] = d["freq"] * (d["sample"] == "PS 2026")
    d["freq_if26"] = d["freq"] * (d["sample"] == "Other IF 2026")
    m1 = _fit(d["ai"], _design(d, ["freq"]))
    m2 = _fit(d["ai"], _design(d, ["freq", "freq_s26", "freq_if26"]))
    lr = 2 * (m2.llf - m1.llf)
    main = pd.DataFrame([_or_row(n, m1.params[c], m1.bse[c]) for n, c in
                         [("GenAI use frequency (0–4)", "freq"), ("PS 2026 (vs. PS 2025)", "s26"),
                          ("Other IF 2026 (vs. PS 2025)", "if26")]])
    cov = m2.cov_params()
    spec = []
    for name, extra in [("PS 2025", None), ("PS 2026", "freq_s26"), ("Other IF 2026", "freq_if26")]:
        est = m2.params["freq"] + (m2.params[extra] if extra else 0)
        var = cov.loc["freq", "freq"] + ((cov.loc[extra, extra] + 2 * cov.loc["freq", extra]) if extra else 0)
        spec.append(_or_row(name, est, np.sqrt(var)))
    return main, pd.DataFrame(spec), (lr, stats.chi2.sf(lr, 2))


# Categorical coding: "Never" (n = 12) is merged with "Once a month or less" because of sparse cells.
FCAT_LABELS = ["Once a month or less (incl. never)", "Several times a month",
               "Several times a week", "Daily or almost daily"]


def _fcat(d):
    d = d.copy()
    d["fcat"] = d["freq"].clip(lower=1) - 1          # 0..3
    for c in (1, 2, 3):
        d[f"fc{c}"] = (d["fcat"] == c).astype(float)
    d["flin"] = d["fcat"].astype(float)
    return d


def _predict_by_category(model, d, cols, setter):
    """Average predicted probability of each AI rating when everyone is set to frequency category c."""
    out = []
    for c in range(4):
        dd = setter(d.copy(), c)
        X = _design(dd, cols)
        out.append(np.asarray(model.predict(X.values)).mean(axis=0))
    return np.array(out)        # shape (4 categories, 5 ratings)


def _set_cat(dd, c):
    for j in (1, 2, 3):
        dd[f"fc{j}"] = float(c == j)
    return dd


def _set_lin(dd, c):
    dd["flin"] = float(c)
    return dd


def ordinal_categorical(data):
    d = _fcat(data)
    cat_cols, lin_cols = ["fc1", "fc2", "fc3"], ["flin"]
    mc = _fit(d["ai"], _design(d, cat_cols))
    ml = _fit(d["ai"], _design(d, lin_cols))
    lr = 2 * (mc.llf - ml.llf)
    rows = [{"Term": f"{FCAT_LABELS[0]} (reference)", "n": int((d["fcat"] == 0).sum()), "OR": np.nan}]
    for j in (1, 2, 3):
        r = _or_row(FCAT_LABELS[j], mc.params[f"fc{j}"], mc.bse[f"fc{j}"])
        r["n"] = int((d["fcat"] == j).sum())
        rows.append(r)
    lin = _or_row("Linear coding (per category, merged 0–3 scale)", ml.params["flin"], ml.bse["flin"])
    fit = {"LR_cat_vs_lin": lr, "df": 2, "p": stats.chi2.sf(lr, 2), "AIC_cat": mc.aic, "AIC_lin": ml.aic}

    p_cat = _predict_by_category(mc, d, cat_cols, _set_cat)
    p_lin = _predict_by_category(ml, d, lin_cols, _set_lin)
    observed = np.array([(d.loc[d["fcat"] == c, "ai"] >= 4).mean() for c in range(4)])

    # bootstrap CIs for P(rating 4 or 5) from the categorical model
    rng = np.random.default_rng(SEED + 1)
    boot = np.full((N_BOOT_PRED, 4), np.nan)
    for b in range(N_BOOT_PRED):
        db = d.iloc[rng.integers(0, len(d), len(d))].reset_index(drop=True)
        try:
            mb = _fit(db["ai"], _design(db, cat_cols))
            boot[b] = _predict_by_category(mb, db, cat_cols, _set_cat)[:, 3:].sum(axis=1)
        except Exception:
            pass
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)

    pred = pd.DataFrame({
        "GenAI use frequency": FCAT_LABELS,
        "n": [int((d["fcat"] == c).sum()) for c in range(4)],
        **{f"P(rating = {r})": p_cat[:, r - 1] for r in range(1, 6)},
        "P(rating 4–5), categorical": p_cat[:, 3:].sum(axis=1),
        "95% CI low": lo, "95% CI high": hi,
        "P(rating 4–5), linear": p_lin[:, 3:].sum(axis=1),
        "Observed proportion 4–5": observed,
    })
    return pd.DataFrame(rows + [dict(lin, n=len(d))]), fit, pred


# ---------- equivalence sensitivity ------------------------------------------------

def equivalence_sensitivity(t4a, t4b, t5):
    rows = []
    t5i = t5.set_index("key")
    for k in KEYS:
        a, b = t4a.set_index("key").loc[k], t4b.set_index("key").loc[k]
        rows.append({
            "Item": SHORT[k],
            "Temporal δ [90% CI]": f"{_r(a['Cliff\'s δ'])} {_ci(a['90% CI low'], a['90% CI high'])}",
            "Temporal smallest bound": a["Smallest equivalence bound"],
            "Cross-program δ [90% CI]": f"{_r(b['Cliff\'s δ'])} {_ci(b['90% CI low'], b['90% CI high'])}",
            "Cross-program smallest bound": b["Smallest equivalence bound"],
            "Spearman ρ [90% CI]": f"{_r(t5i.loc[k, 'ρ'])} {_ci(t5i.loc[k, '90% CI low'], t5i.loc[k, '90% CI high'])}",
            "ρ smallest bound": t5i.loc[k, "Smallest equivalence bound"],
        })
    t = pd.DataFrame(rows)
    summary = {
        **{f"Temporal equivalent at ±{b}": int((t4a["Smallest equivalence bound"] < b).sum()) for b in DELTA_BOUNDS},
        **{f"Cross-program equivalent at ±{b}": int((t4b["Smallest equivalence bound"] < b).sum()) for b in DELTA_BOUNDS},
        **{f"ρ (9 non-AI items) equivalent at ±{b}":
           int((t5.loc[t5["key"] != "ai", "Smallest equivalence bound"] < b).sum()) for b in RHO_BOUNDS},
    }
    return t, summary


# ---------- figures ------------------------------------------------------------------

def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#444444", "axes.linewidth": 0.8, "savefig.dpi": 300,
    })
    return plt


def figure1_means(data, path):
    """Mean ratings with 95% CIs; value labels placed to the right of the CI whiskers."""
    plt = _style()
    rows = []
    for k in KEYS:
        x = data[k].to_numpy(float)
        m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
        h = stats.t.ppf(0.975, len(x) - 1) * se
        rows.append((SHORT[k], m, m - h, m + h))
    t = pd.DataFrame(rows, columns=["item", "m", "lo", "hi"]).sort_values("m").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 5.6))
    for y, r in t.iterrows():
        ax.plot([r.lo, r.hi], [y, y], color=BLUE, linewidth=2, solid_capstyle="round", zorder=2)
        ax.plot(r.m, y, "o", color=BLUE, markersize=7, zorder=3)
        ax.text(r.hi + 0.06, y, f"{r.m:.2f}", va="center", ha="left", fontsize=10.5)
    ax.set_yticks(range(len(t)), t["item"])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(1, 5.2)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Mean rating (1–5) with 95% confidence interval")
    ax.grid(axis="x", color="#dddddd", linestyle=":", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure_freq_distribution(data, path):
    """100% stacked bars of GenAI use frequency per sample (ordinal blue ramp, validated)."""
    plt = _style()
    colors = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
    order = SAMPLES[::-1]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for y, s in enumerate(order):
        sub = data.loc[data["sample"] == s, "freq"]
        left = 0
        for c in range(5):
            w = (sub == c).mean() * 100
            ax.barh(y, w, left=left, height=0.58, color=colors[c], edgecolor="white", linewidth=2)
            if w >= 6:
                ax.text(left + w / 2, y, f"{w:.0f}%", ha="center", va="center", fontsize=10,
                        color="#1a1a1a" if c < 2 else "white")
            left += w
    ax.set_yticks(range(len(order)), [f"{s} (n={(data['sample'] == s).sum()})" for s in order])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of respondents (%)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    ax.legend(handles, FREQ_LABELS, ncol=5, loc="lower center", bbox_to_anchor=(0.45, 1.02),
              frameon=False, fontsize=9.5, handlelength=1.2, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure_correlations(t5, path):
    """Spearman rho with 90% CI and the +/-.20 equivalence region."""
    plt = _style()
    t = t5.sort_values("ρ").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.axvspan(-RHO_EQ, RHO_EQ, color="#ecebe7", zorder=0)
    ax.axvline(0, color=BLUE, linewidth=1, zorder=1)
    for y, r in t.iterrows():
        ax.plot([r["90% CI low"], r["90% CI high"]], [y, y], color=BLUE, linewidth=2, zorder=2,
                solid_capstyle="round")
        ax.plot(r["ρ"], y, "o", color=BLUE, markersize=8, markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        p = r["Holm-adjusted p"]
        star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
        ax.text(r["90% CI high"] + 0.012, y, f"{r['ρ']:.2f}".replace("-", "−") + star, va="center", fontsize=10.5)
    ax.set_yticks(range(len(t)), t["Item"])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.4, 0.45)
    ax.set_xlabel("Spearman correlation with GenAI use frequency (ρ, 90% CI)")
    ax.grid(axis="x", color="#dddddd", linestyle=":", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.text(0, len(t) - 0.35, f"Equivalence region (|ρ| < .{int(RHO_EQ * 100)})", ha="center",
            va="bottom", fontsize=9.5, color="#555555")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figureA1_predicted(pred, path):
    """Predicted probability of rating AI tools 4 or 5 by GenAI use frequency: categorical vs linear coding."""
    plt = _style()
    x = np.arange(len(pred))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(x, pred["P(rating 4–5), linear"], "-", color="#86b6ef", linewidth=2, zorder=1,
            label="Linear coding (model prediction)")
    dodge = 0.07   # categorical prediction slightly left, observed proportion slightly right
    ax.vlines(x - dodge, pred["95% CI low"], pred["95% CI high"], color=BLUE, linewidth=2, zorder=2)
    ax.plot(x - dodge, pred["P(rating 4–5), categorical"], "o", color=BLUE, markersize=7, zorder=3,
            label="Categorical coding (prediction, 95% CI)")
    ax.plot(x + dodge, pred["Observed proportion 4–5"], "D", markerfacecolor="white", markeredgecolor="#0d366b",
            markersize=6.5, zorder=4, label="Observed proportion")
    for xi, v in zip(x, pred["P(rating 4–5), categorical"]):
        ax.text(xi - dodge - 0.06, v, f"{v:.2f}", va="center", ha="right", fontsize=10)
    ax.set_xticks(x, [f"{l}\n(n = {n})" for l, n in zip(
        ["≤ Once a month", "Several times\na month", "Several times\na week", "Daily or\nalmost daily"], pred["n"])],
        fontsize=10)
    ax.set_xlim(-0.55, len(pred) - 0.6)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("Probability of rating AI tools 4 or 5")
    ax.set_xlabel("GenAI use frequency")
    ax.grid(axis="y", color="#dddddd", linestyle=":", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------- export -----------------------------------------------------------------

def software_versions():
    import matplotlib
    import scipy
    import statsmodels
    return {"Python": platform.python_version(), "NumPy": np.__version__, "pandas": pd.__version__,
            "SciPy": scipy.__version__, "statsmodels": statsmodels.__version__,
            "matplotlib": matplotlib.__version__}


def build_tables(R):
    t3 = R["t3"].copy()
    t3 = pd.DataFrame({
        "Comparison": t3["Comparison"], "AI M": t3["AI M"].map("{:.2f}".format),
        "Comparison M": t3["Comparison M"].map("{:.2f}".format),
        "Mean difference": t3["Mean difference"].map(lambda v: f"{v:.2f}".replace("-", "−")),
        "Rank-biserial r": t3["Rank-biserial r"].map(lambda v: f"{v:.3f}".replace("-", "−")),
        "Holm-adjusted p": t3["Holm-adjusted p"].map(fmt_p)})

    def dm(v):
        return f"{v + 0:.2f}".replace("-0.00", "0.00").replace("-", "−")

    a, b = R["t4a"], R["t4b"]
    t5_table = pd.DataFrame({
        "Item": a["Item"], "2026−2025 PS ΔM": a["ΔM"].map(dm),
        "Cliff’s δ [90% CI]": [f"{_r(d)} {_ci(l, h)}" for d, l, h in zip(a["Cliff's δ"], a["90% CI low"], a["90% CI high"])],
        "Holm-adjusted p": a["Holm-adjusted p"].map(fmt_p),
        "Other IF−PS 2026 ΔM": b["ΔM"].map(dm),
        "Cliff’s δ [90% CI] ": [f"{_r(d)} {_ci(l, h)}" for d, l, h in zip(b["Cliff's δ"], b["90% CI low"], b["90% CI high"])],
        "Holm-adjusted p ": b["Holm-adjusted p"].map(fmt_p)})

    ft = R["ft"]
    ft_table = pd.DataFrame({
        "Comparison": ft["Comparison"], "U": ft["U"].map("{:.1f}".format),
        "Cliff’s δ [95% CI]": [f"{_r(d)} {_ci(l, h)}" for d, l, h in zip(ft["Cliff's δ"], ft["95% CI low"], ft["95% CI high"])],
        "Holm-adjusted p": ft["Holm-adjusted p"].map(fmt_p)})

    t5 = R["t5"].sort_values("ρ", ascending=False)
    a2 = pd.DataFrame({
        "Item": t5["Item"], "Spearman ρ": t5["ρ"].map(_r),
        "90% CI": [_ci(l, h) for l, h in zip(t5["90% CI low"], t5["90% CI high"])],
        "Holm-adjusted p": t5["Holm-adjusted p"].map(fmt_p),
        "Equivalence (|ρ| < .20)": np.where(t5["Smallest equivalence bound"] < RHO_EQ, "Equivalent", "Not equivalent")})

    def fm(df):
        return pd.DataFrame({"Term": df["Term"], "OR": df["OR"].map("{:.2f}".format),
                             "95% CI": [f"{l:.2f}–{h:.2f}" for l, h in zip(df["95% CI low"], df["95% CI high"])],
                             "p": df["p"].map(fmt_p)})

    sens = R["sens"].copy()
    for c in ["Temporal smallest bound", "Cross-program smallest bound", "ρ smallest bound"]:
        sens[c] = sens[c].map(_r)

    a4 = pd.DataFrame({"Item": [SHORT[k] for k in KEYS]})
    for s, t in R["ss"].items():
        t = t.set_index("key")
        a4[f"{s} (n = {int(t['n'].iloc[0])})"] = [
            f"{_r(t.loc[k, 'ρ'], 2)} {_ci(t.loc[k, '90% CI low'], t.loc[k, '90% CI high'], 2)}"
            + ("*" if t.loc[k, "Holm-adjusted p"] < .05 else "") for k in KEYS]

    cat = R["cat"]
    a5 = pd.DataFrame({
        "GenAI use frequency": cat["Term"], "n": cat["n"],
        "OR": cat["OR"].map(lambda v: "1 (ref.)" if np.isnan(v) else f"{v:.2f}"),
        "95% CI": ["–" if np.isnan(l) else f"{l:.2f}–{h:.2f}" for l, h in zip(cat["95% CI low"], cat["95% CI high"])],
        "p": cat["p"].map(lambda v: "–" if np.isnan(v) else fmt_p(v))})
    pred = R["pred"]
    a5b = pd.DataFrame({
        "GenAI use frequency": pred["GenAI use frequency"], "n": pred["n"],
        **{f"P({r})": pred[f"P(rating = {r})"].map("{:.2f}".format) for r in range(1, 6)},
        "P(4–5) [95% CI]": [f"{p:.2f} [{l:.2f}, {h:.2f}]" for p, l, h in zip(
            pred["P(rating 4–5), categorical"], pred["95% CI low"], pred["95% CI high"])],
        "P(4–5), linear": pred["P(rating 4–5), linear"].map("{:.2f}".format),
        "Observed 4–5": pred["Observed proportion 4–5"].map("{:.2f}".format)})

    a6 = pd.DataFrame(
        [{"No.": i + 1, "Lithuanian (original)": LT[k], "English (translation)": LABEL[k]} for i, k in enumerate(KEYS)]
        + [{"No.": 11, "Lithuanian (original)": FREQ_ITEM_LT + " (" + "; ".join(FREQ_LT) + ")",
            "English (translation)": FREQ_ITEM_EN + " (" + "; ".join(FREQ_LABELS).lower() + ")"}])

    return {
        "Table2_descriptives": R["t2"], "Table3_AI_vs_others": t3, "Table4_GenAI_frequency": R["tf"],
        "GenAI_frequency_tests": ft_table, "Table5_TOST": t5_table, "TableA1_sample_specific_OR": fm(R["spec"]),
        "Ordinal_model_linear": fm(R["om"]), "TableA2_correlations": a2, "TableA3_equiv_sensitivity": sens,
        "TableA4_sample_correlations": a4, "TableA5_categorical_OR": a5, "TableA5b_predicted_prob": a5b,
        "TableA6_Lithuanian_items": a6,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", default=None, help="path or URL of analysis_data.csv")
    ap.add_argument("--out", default=str(HERE / "outputs_v2"), help="output directory")
    args = ap.parse_args(argv)
    source = args.data or (HERE / "analysis_data.csv" if (HERE / "analysis_data.csv").exists() else GITHUB_CSV)
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    data = load(source)
    print("Data:", source)
    print("N per sample:", data["sample"].value_counts().reindex(SAMPLES).to_dict())

    R = {"t2": table2_descriptives(data), "fr": rq1_friedman(data), "t3": table3_ai_vs_others(data)}
    # order of bootstrap calls kept identical to version 1 so that the reported CIs are reproduced exactly
    R["t4a"], R["rho_a"] = group_comparison(data, "PS 2025", "PS 2026")
    R["t4b"], R["rho_b"] = group_comparison(data, "PS 2026", "Other IF 2026")
    R["tf"] = table_freq_distribution(data)
    R["ft"], R["kw"] = freq_tests(data)
    R["t5"] = correlations(data)
    R["rel"] = relative_ai(data)
    R["om"], R["spec"], R["lr_int"] = ordinal_models(data)
    R["ss"] = sample_specific_correlations(data)
    R["sens"], R["sens_summary"] = equivalence_sensitivity(R["t4a"], R["t4b"], R["t5"])
    R["cat"], R["cat_fit"], R["pred"] = ordinal_categorical(data)

    figure1_means(data, out / "figures" / "Figure1_mean_ratings.png")
    figure_freq_distribution(data, out / "figures" / "Figure2_GenAI_use_frequency_by_sample.png")
    figure_correlations(R["t5"], out / "figures" / "Figure3_GenAI_use_correlations.png")
    figureA1_predicted(R["pred"], out / "figures" / "FigureA1_predicted_probabilities.png")

    tables = build_tables(R)
    with pd.ExcelWriter(out / "results_v2.xlsx") as xw:
        for name, t in tables.items():
            t.to_excel(xw, sheet_name=name[:31], index=False)

    versions = software_versions()
    lines = [f"{k} {v}" for k, v in versions.items()]
    lines += ["", f"Friedman: {R['fr']}", f"Kruskal-Wallis (GenAI frequency): H = {R['kw'][0]:.2f}, p = {R['kw'][1]:.3f}",
              f"Profile rho: temporal {R['rho_a']:.3f}, cross-program {R['rho_b']:.3f}",
              f"Interaction LR (linear model): chi2(2) = {R['lr_int'][0]:.2f}, p = {R['lr_int'][1]:.3f}",
              f"Relative AI preference: {R['rel']}", f"Equivalence sensitivity: {R['sens_summary']}",
              f"Categorical vs linear coding: {R['cat_fit']}"]
    (out / "summary_v2.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    for name in ["TableA3_equiv_sensitivity", "TableA4_sample_correlations", "TableA5_categorical_OR",
                 "TableA5b_predicted_prob"]:
        print(f"\n{name}\n", tables[name].to_string(index=False))
    return R, tables


if __name__ == "__main__":
    main()
