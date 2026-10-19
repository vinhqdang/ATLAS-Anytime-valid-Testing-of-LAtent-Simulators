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
