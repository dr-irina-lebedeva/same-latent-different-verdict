"""Convert GitHub-markdown pipe tables (from the run reports) to LaTeX tabulars."""
import re
REP = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"), ("$", r"\$"),
       ("Δ", r"$\Delta$"), ("σ", r"$\sigma$"), ("→", r"$\to$"), ("->", r"$\to$"), ("≥", r"$\ge$"), ("≤", r"$\le$"),
       ("±", r"$\pm$"), ("·", r"$\cdot$"), ("—", "--"), ("–", "--"), ("×", r"$\times$"), ("ρ", r"$\rho$"), ("≈", r"$\approx$")]
def esc(t):
    t = t.strip().replace("**", "")
    out = t
    for a, b in REP[:6]: out = out.replace(a, b)
    for a, b in REP[6:]: out = out.replace(a, b)
    out = re.sub(r"`([^`]*)`", r"\\texttt{\1}", out)
    return out
NAMES = {"ftmse": "ft-MSE", "taesd": "TAESD", "cons": "Cons.", "orig": "orig", "probe\\_clip": "CLIP probe",
         "probe\\_dino": "DINO probe", "commfor": "Comm.-For.", "aero\\_orig": "AERO(orig AE)", "aero\\_ftmse": "AERO(ft-MSE AE)",
         "aero\\_taesd": "AERO(TAESD AE)", "aero\\_min": "AERO(best of 3)", "univfd": "UnivFD", "npr": "NPR",
         "aeroblade": "AEROBLADE", "real\\_enc": "real-enc", "taef1": "TAEF1"}
def nice(c):
    return NAMES.get(c, c)
def sections(md):
    """[(heading, rows)] : each table with the nearest preceding '## ' heading"""
    out = []; head = ""; lines = md.splitlines(); i = 0
    while i < len(lines):
        if lines[i].startswith("#"): head = lines[i].lstrip("#").strip().strip("`")
        if lines[i].startswith("|") and i + 1 < len(lines) and re.match(r"^\|[-|: ]+\|$", lines[i + 1].strip()):
            rows = [lines[i]]; i += 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i]); i += 1
            out.append((head, [[nice(esc(c)) for c in r.strip().strip("|").split("|")] for r in rows])); continue
        i += 1
    return out
def tables(md):
    """yield list-of-rows for each pipe table in md text"""
    lines = md.splitlines(); i = 0
    while i < len(lines):
        if lines[i].startswith("|") and i + 1 < len(lines) and re.match(r"^\|[-|: ]+\|$", lines[i + 1].strip()):
            rows = [lines[i]]; i += 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i]); i += 1
            yield [[nice(esc(c)) for c in r.strip().strip("|").split("|")] for r in rows]
        else:
            i += 1
def to_tex(rows, caption=None, size=r"\scriptsize", fit=True, label=None):
    n = len(rows[0])
    body = " \\\\\n".join(" & ".join(r + [""] * (n - len(r))) for r in rows[1:])
    tab = (f"\\begin{{tabular}}{{l{'c' * (n - 1)}}}\n\\toprule\n" + " & ".join(rows[0]) + " \\\\\n\\midrule\n"
           + body + " \\\\\n\\bottomrule\n\\end{tabular}")
    if fit: tab = "\\begin{adjustbox}{max width=\\linewidth}\n" + tab + "\n\\end{adjustbox}"
    s = "\\begin{table}[h!]\n\\centering" + size + "\n" + tab + "\n"
    if caption: s += f"\\caption{{{caption}}}\n"
    if label: s += f"\\label{{{label}}}\n"
    return s + "\\end{table}\n"
