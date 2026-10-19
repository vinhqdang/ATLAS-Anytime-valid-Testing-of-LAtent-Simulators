"""Generate tab_real.tex from results/exp8_*.json."""
import json, os
R = os.path.join(os.path.dirname(__file__), "..", "results")
NAMES = {"mnist": "Moving MNIST: native $\\to$ double speed",
         "kth": "KTH: walking $\\to$ running",
         "domain": "KTH: outdoor $\\to$ indoor walking"}
ROWS = [("ATLAS(eps+0.0sd)", "ATLAS, $k=0$"), ("ATLAS(eps+0.25sd)", "ATLAS, $k=0.25$"),
        ("ATLAS(eps+0.5sd)", "ATLAS, $k=0.5$"), ("ATLAS(eps+1.0sd)", "ATLAS, $k=1$"),
        ("CUSUM", "CUSUM"), ("ADWIN", "ADWIN"), ("KSWIN", "KSWIN"),
        ("Conformal MG", "Conformal martingale")]

def f(t, ci=True):
    if t[0] != t[0]:
        return "---"
    return f"{t[0]:.2f} [{t[1]:.2f},{t[2]:.2f}]" if ci else f"{t[0]:.2f}"

def g(t):
    return "---" if t[0] != t[0] else f"{t[0]:.1f} [{t[1]:.1f},{t[2]:.1f}]"

out = ["\\begin{table}[t]\\centering",
       "\\caption{Replicated real-video study (30 random splits per shift). False alarms over $L$ in-distribution clips and over $3L$; detection rate and delay in clips after the shift; bootstrap 95\\% intervals in brackets. $k$ is the tolerance margin in standard deviations of the calibration statistic. Tunable detectors are tuned to 5\\% on separate repetitions.}",
       "\\label{tab:real}", "\\small", "\\setlength{\\tabcolsep}{3pt}",
       "\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
       " & FA @ $L$ & FA @ $3L$ & detection rate & delay (clips) \\\\"]
for key in ["mnist", "kth", "domain"]:
    p = os.path.join(R, f"exp8_{key}.json")
    if not os.path.exists(p):
        continue
    S = json.load(open(p))["summary"]
    out.append("\\midrule")
    out.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{NAMES[key]}}}}} \\\\")
    for k, lab in ROWS:
        r = S[k]
        out.append(f"{lab} & {f(r['fa_L'])} & {f(r['fa_3L'])} & {f(r['det'], False)} & {g(r['delay'])} \\\\")
out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(os.path.join(os.path.dirname(__file__), "tab_real.tex"), "w").write("\n".join(out) + "\n")


# ----------------------------------------------------------------- control study table
def control_table():
    p = os.path.join(R, "exp10_gated_control.json")
    if not os.path.exists(p):
        return
    S = json.load(open(p))["summary"]
    ctrls = [("fixed-4", "fixed horizon 4"), ("fixed-16", "fixed horizon 16 (ungated)"),
             ("ATLAS-gated k=0.25", "ATLAS gate, $k=0.25$"), ("ATLAS-gated k=0.5", "ATLAS gate, $k=0.5$"),
             ("ATLAS-gated k=1.0", "ATLAS gate, $k=1$"), ("ATLAS-gated k=2.0", "ATLAS gate, $k=2$"),
             ("CUSUM-gated", "CUSUM gate"), ("oracle", "oracle gate"), ("brake", "brake always")]
    faults = [("gain", "gain $\\times0.4$"), ("flip", "$x$-axis flipped"), ("wind", "wind")]
    cap = ("Gated planning with a neural world model. Mean stage cost (lower is better), 20 episodes per "
           "cell, bootstrap 95\\% intervals in brackets. ``No fault'' is the mean cost over the second half "
           "of a fault-free episode and ``false trigger'' the fraction of fault-free episodes in which a gate "
           "fired. Gate controllers switch to the safe fallback (zero action) when they alarm. Cost before the "
           "fault is $0.154$ for every long-horizon controller.")
    out = ["\\begin{table}[t]\\centering", "\\caption{" + cap + "}", "\\label{tab:control}",
           "\\small", "\\setlength{\\tabcolsep}{3pt}",
           "\\begin{tabular}{@{}lcc" + "c" * len(faults) + "c@{}}", "\\toprule",
           " & no fault & false trigger & " + " & ".join(f[1] for f in faults)
           + " & mean (3 faults) \\\\", "\\midrule"]
    for k, lab in ctrls:
        gate = k.startswith(("ATLAS", "CUSUM"))
        r0 = S[f"none|{k}"]
        ft = f"{r0['early'] + r0['detect']:.2f}" if gate else "---"
        cells, vals = [f"{r0['post']:.2f}", ft], []
        for f, _ in faults:
            r = S[f"{f}|{k}"]
            cells.append(f"{r['post']:.2f} [{r['post_ci'][0]:.1f},{r['post_ci'][1]:.1f}]")
            vals.append(r["post"])
        out.append(f"{lab} & " + " & ".join(cells) + f" & {sum(vals)/len(vals):.1f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(os.path.dirname(__file__), "tab_control.tex"), "w").write("\n".join(out) + "\n")


control_table()


# ----------------------------------------------------------------- model-class table
def models_table():
    files = [("mnist", "Moving MNIST speed", [("exp8_mnist", "PCA + linear (30 splits)"),
                                              ("exp8n_mnist", "conv. autoencoder (10)")]),
             ("kth", "KTH walking$\\to$running", [("exp8_kth", "PCA + linear (30)"),
                                                   ("exp8n_kth", "conv. autoencoder (10)"),
                                                   ("exp8j_kth", "frozen ResNet + dynamics (10)")]),
             ("domain", "KTH outdoor$\\to$indoor", [("exp8_domain", "PCA + linear (30)"),
                                                     ("exp8n_domain", "conv. autoencoder (10)"),
                                                     ("exp8j_domain", "frozen ResNet + dynamics (10)")])]
    cols = [("ATLAS(eps+0.5sd)", "ATLAS $k{=}0.5$"), ("ATLAS(eps+1.0sd)", "ATLAS $k{=}1$"),
            ("CUSUM", "CUSUM"), ("Conformal MG", "conformal MG")]
    cap = ("Three world-model classes on the same real shifts. Each cell: false-alarm rate over $3L$ in-distribution "
           "clips / detection rate / mean delay in clips (``--'' if never detected); repetitions in parentheses. "
           "ADWIN and KSWIN, tuned to 5\\% on separate repetitions, almost never detected a shift with the neural models (one exception: ADWIN in 2 of 10 repetitions of the conv.\\ autoencoder on the domain shift) and are omitted.")
    out = ["\\begin{table}[t]\\centering", "\\caption{" + cap + "}", "\\label{tab:models}", "\\small",
           "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}ll" + "c" * len(cols) + "@{}}", "\\toprule",
           "shift & world model & " + " & ".join(c[1] for c in cols) + " \\\\", "\\midrule"]
    for key, name, models in files:
        first = True
        for fn, lab in models:
            p = os.path.join(R, fn + ".json")
            if not os.path.exists(p):
                continue
            S = json.load(open(p))["summary"]
            cells = []
            for ck, _ in cols:
                r = S[ck]
                d = "--" if r["delay"][0] != r["delay"][0] else f"{r['delay'][0]:.1f}"
                cells.append(f"{r['fa_3L'][0]:.2f} / {r['det'][0]:.2f} / {d}")
            out.append((name if first else "") + f" & {lab} & " + " & ".join(cells) + " \\\\")
            first = False
        out.append("\\midrule")
    out[-1] = "\\bottomrule"
    out += ["\\end{tabular}", "\\end{table}"]
    open(os.path.join(os.path.dirname(__file__), "tab_models.tex"), "w").write("\n".join(out) + "\n")


models_table()


# fit every generated table to the text width
for name in ("tab_real.tex", "tab_control.tex", "tab_models.tex"):
    p = os.path.join(os.path.dirname(__file__), name)
    if os.path.exists(p):
        t = open(p).read()
        if "\\resizebox" not in t:
            t = t.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{\\begin{tabular}").replace(
                "\\end{tabular}", "\\end{tabular}}")
            open(p, "w").write(t)
