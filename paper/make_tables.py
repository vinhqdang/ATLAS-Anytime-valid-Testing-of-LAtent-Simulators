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
