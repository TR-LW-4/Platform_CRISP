"""Comparison table for Wei from the overnight runs (results/ and jobs.jsonl), Markdown and LaTeX.

OPT is read from docs/validation/jovanovic_2019.md ("Checks on the O_v difference": O_f,30 exact vs Table 5).
Cross-checks per instance: Jovanović against step 5 (step5_b379031.jsonl, check Of30, seed 0) and López-Plata
against V35 (f2_bench.jsonl), on 3x5, 3x7, 4x6. After writing, every cell is recomputed from the result files and
compared with the written Markdown and LaTeX.

Inputs (all local, not in git): results/CRP-Time/ (gitignored), and in docs/validation/data/ (excluded locally)
overnight_2026-10-05/jobs.jsonl and driver.log, jovanovic_2019_step5/step5_b379031.jsonl and
lopez_plata_2019/f2_bench.jsonl. The machine of the run is recorded below, so the output does not depend on the
machine that regenerates it.

  python docs/benchmark/make_table.py      (from any directory)
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN_DIR = REPO / "docs" / "validation" / "data" / "overnight_2026-10-05"
OUT_DIR = REPO / "docs" / "benchmark"
# Machine of the overnight run (5-6 Oct 2026; the Python of the conda env rl that ran the server)
RUN_MACHINE = ("AMD Ryzen 7 8700F 8-Core Processor", "3.11.16", "Windows 11 (10.0.26300)")
CLASSES = ["3x5", "3x7", "4x6", "5x7", "6x10"]
ALGOS = [  # display name, result folder, LaTeX name
    ("Azari et al. (2017) CSUM", "Azari_et_al._(2017)_CSUM", r"Azari et al.\ (2017) CSUM"),
    ("Jovanović et al. (2019) ACO", "Jovanović_et_al._(2019)_ACO_(CRP-Time)", r"Jovanovi\'{c} et al.\ (2019) ACO"),
    ("López-Plata et al. (2019) A∗-Based Heuristic", "López-Plata_et_al._(2019)_A∗-Based_Heuristic",
     r"L\'{o}pez-Plata et al.\ (2019) A$^\ast$-Based Heuristic"),
]
TOL = 1e-9


def load(folder, cls):
    runs = {}
    for f in sorted((REPO / "results" / "CRP-Time" / folder).glob(f"caserta_{cls}_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        layout = d["prob_config"]["layout_file_path"]
        m = re.search(r"data(\d+)-(\d+)-(\d+)\.dat$", layout)
        assert f"{m.group(1)}x{m.group(2)}" == cls, (f, layout)
        runs[int(m.group(3))] = d["metrics"]
    return runs


def ms(values, nd):
    return f"{st.mean(values):.{nd}f} ± {st.stdev(values):.{nd}f}"


def runtime_cell(values):
    return ms(values, 3 if st.mean(values) < 10 else 1)


def opt_from_log():
    text = (REPO / "docs" / "validation" / "jovanovic_2019.md").read_text(encoding="utf-8")
    sec = text[text.index("**OPT column of Table 5 — equal to the exact rBRP optimum"):]
    opt = {}
    for h, w, exact, table5 in re.findall(r"\| (\d) × (\d+) \| ([\d.]+) \| ([\d.]+) \|", sec[:3000]):
        assert exact == table5, (h, w, exact, table5)
        opt[f"{h}x{w}"] = float(table5)
    return opt


def cells(opt):
    """All table cells, computed from the result files."""
    rows = []
    for cls in CLASSES:
        for name, folder, _ in ALGOS:
            r = load(folder, cls)
            v = list(r.values())
            f2 = [x["crane_time_f2"] for x in v]
            row = {"class": cls.replace("x", "×"), "algorithm": name, "n": str(len(v)), "f2": ms(f2, 1),
                   "f2vert": ms([x["crane_time_vertical"] for x in v], 1),
                   "relocations": ms([x["relocations"] for x in v], 2),
                   "runtime": runtime_cell([x["runtime_s"] for x in v]),
                   "limit": (f"{int(sum(x['time_limit_hit'] for x in v))}/{len(v)}" if "time_limit_hit" in v[0] else "–")}
            if cls in opt:
                gap = st.mean(f2) - opt[cls]
                row["gap"] = f"{gap:+.2f} ({100 * gap / opt[cls]:+.3f} %)"
            else:
                row["gap"] = "–"
            rows.append(row)
        rows.append({"class": cls.replace("x", "×"), "algorithm": "OPT (Jovanović et al. 2019, Table 5, O_f,30)" if cls in opt else
                     "OPT: none in Table 5", "n": "–", "f2": f"{opt[cls]:.1f}" if cls in opt else "–", "f2vert": "–",
                     "relocations": "–", "runtime": "–", "limit": "–", "gap": "–"})
    return rows


def cross_checks():
    out = []
    jov = {}
    for line in open(REPO / "docs/validation/data/jovanovic_2019_step5/step5_b379031.jsonl", encoding="utf-8"):
        d = json.loads(line)
        if d["check"] == "Of30" and d["seed"] == 0:
            jov[(d["size"].replace("-", "x"), d["inst"])] = d
    lp = {}
    for line in open(REPO / "docs/validation/data/lopez_plata_2019/f2_bench.jsonl", encoding="utf-8"):
        d = json.loads(line)
        if d.get("status") == "ok":
            lp[(d["class"].replace("-", "x"), int(d["instance"].rsplit("-", 1)[1]))] = d
    for label, folder, ref, keys in (
            ("Jovanović vs step 5 (Of30, seed 0, b379031)", ALGOS[1][1], jov,
             (("crane_time_f2", "f2"), ("crane_time_vertical", "f2vert"), ("relocations", "relocations"))),
            ("López-Plata vs V35 (f2_bench.jsonl)", ALGOS[2][1], lp,
             (("crane_time_f2", "f2"), ("crane_time_vertical", "f2vert"), ("relocations", "relocations")))):
        for cls in ("3x5", "3x7", "4x6"):
            runs = load(folder, cls)
            diffs = [(i, k) for i, m in runs.items() for k, rk in keys
                     if (cls, i) not in ref or abs(m[k] - ref[(cls, i)][rk]) > 1e-6]
            n_ref = sum(1 for (c, _) in ref if c == cls)
            out.append(f"{label}, {cls}: {len(runs)} runs, {n_ref} reference runs, differences: "
                       + ("none" if not diffs else f"{len(diffs)} {diffs[:6]}"))
    return out


def jobs_check():
    rows = [json.loads(line) for line in open(RUN_DIR / "jobs.jsonl", encoding="utf-8")]
    bad = [r for r in rows if r["status"] != "completed" or r["result_files"] != 40 or not r["all_have_runtime_s"]]
    return len(rows), bad


def machine():
    cpu, py, os_ = RUN_MACHINE
    log = (RUN_DIR / "driver.log").read_text(encoding="utf-8").splitlines()
    commit = re.search(r"commit (\w+)", log[0]).group(1)
    return cpu, py, os_, commit, log[0][:19], log[-1][:19]


HEAD = ["Class", "Algorithm", "n", "f2 (s)", "f2vert (s)", "Relocations", "runtime_s (s)", "Limit hit", "f2 − OPT (s, %)"]
KEYS = ["class", "algorithm", "n", "f2", "f2vert", "relocations", "runtime", "limit", "gap"]


def tex_escape(row):
    name = {a[0]: a[2] for a in ALGOS}.get(row["algorithm"])
    if name is None:
        name = (r"OPT (Jovanovi\'{c} et al.\ 2019, Table 5, $O_{f,30}$)" if row["algorithm"].startswith("OPT (")
                else r"OPT: none in Table 5")
    vals = [row["class"].replace("×", r"$\times$"), name] + [row[k] for k in KEYS[2:]]
    return [v.replace("±", r"$\pm$").replace("%", r"\%").replace("–", "--").replace("−", "$-$") for v in vals]


def write(rows, notes, opt):
    cpu, py, os_, commit, t_start, t_end = machine()
    intro = [
        "# CRP-Time comparison under f2: Azari, Jovanović, López-Plata",
        "",
        "Generated by `docs/benchmark/make_table.py` from the result files in "
        "`results/CRP-Time/` and `jobs.jsonl` of the overnight run; every cell is recomputed from the result files "
        "after writing.",
        "",
        "- **Objective:** f2 (crane time with constant pick-up/place-down effort, Voß & Schwarze 2019) is optimised; "
        "f2vert is reported from the same runs (the platform evaluates the final plan under both). Relocations are "
        "measured, not optimised. Restricted BRP, H = T + 2, CRISP time defaults (ts 1.2 s, tpp 30 s).",
        "- **Instances:** Caserta, all 40 instances per class. Seed 0.",
        f"- **Runs:** 15 web jobs through `/api/jobs` (the Workbench path), strictly one after another, one "
        f"instance at a time in one worker process; code `{commit}`, {t_start} – {t_end}. Machine: {cpu}, "
        f"Python {py}, {os_}. runtime_s is the wall clock of `train()` per instance.",
        "- **Time budgets:** Azari: GBH 3 s + CSUM 5 s (wall-clock limits of the paper's method, defaults). "
        "Jovanović: 5000 ACO iterations (defaults). López-Plata: paper parameters (ρ 300, λ 500, θ 40, κ 2, µ 5, "
        "τ 3, node budget 10⁷) with `time_limit_sec` 600, checked before step 2 and after every completed partial "
        "solution (L32); greedy and step 1 are not interrupted.",
        "- **OPT:** the O_f,30 column of Table 5 in Jovanović et al. (2019), equal to the average exact rBRP "
        "optimum where checked (`docs/validation/jovanovic_2019.md`, \"Checks on the O_v difference\"); O_f,30 = f2. "
        "Table 5 has no row for 5×7 and 6×10. Gap: unrounded class mean − OPT (OPT is printed to 0.1 s).",
        "- **Format:** mean ± standard deviation (n − 1) over the 40 instances; runtime_s with 3 decimals below 10 s, "
        "1 decimal above. Limit hit: López-Plata instances that reached the 600 s limit.",
        "",
    ]
    md = ["| " + " | ".join(HEAD) + " |", "|" + "---|" * len(HEAD)]
    md += ["| " + " | ".join(r[k] for k in KEYS) + " |" for r in rows]
    checks = ["", "**Cross-checks (per instance, f2, f2vert and relocations, tolerance 1e-6):**", ""]
    checks += [f"- {n}" for n in notes]
    (OUT_DIR / "crp_time_f2_comparison.md").write_text("\n".join(intro + md + checks) + "\n", encoding="utf-8")
    tex = [
        "% CRP-Time comparison under f2, generated by make_table.py from the overnight run; see crp_time_f2_comparison.md for the setup.",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrlllllr}",
        r"\toprule",
        " & ".join(["Class", "Algorithm", "$n$", "$f_2$ (s)", r"$f_2^{\mathrm{vert}}$ (s)", "Relocations",
                     "Runtime (s)", "Limit hit", r"$f_2 - \mathrm{OPT}$ (s, \%)"]) + r" \\",
        r"\midrule",
    ]
    for i, r in enumerate(rows):
        tex.append(" & ".join(tex_escape(r)) + r" \\")
        if r["algorithm"].startswith("OPT") and i != len(rows) - 1:
            tex.append(r"\midrule")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\caption{CRP-Time, Caserta instances, 40 per class, seed 0: $f_2$ optimised, $f_2^{\mathrm{vert}}$ reported from the same runs; mean $\pm$ standard deviation. OPT: $O_{f,30}$ column of Table 5 in Jovanovi\'{c} et al.\ (2019). L\'{o}pez-Plata with a 600 s time limit per instance.}",
            r"\label{tab:crp-time-f2-comparison}", r"\end{table}"]
    (OUT_DIR / "crp_time_f2_comparison.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def verify(opt):
    expected = cells(opt)
    md = (OUT_DIR / "crp_time_f2_comparison.md").read_text(encoding="utf-8").splitlines()
    md_rows = [[c.strip() for c in line.strip("|").split("|")] for line in md if line.startswith("| ") and "---" not in line][1:]
    tex = (OUT_DIR / "crp_time_f2_comparison.tex").read_text(encoding="utf-8").splitlines()
    tex_rows = [line[:-3].split(" & ") for line in tex if line.endswith(r" \\") and "$n$" not in line]
    assert len(md_rows) == len(expected) == len(tex_rows), (len(md_rows), len(expected), len(tex_rows))
    for e, m, t in zip(expected, md_rows, tex_rows):
        assert m == [e[k] for k in KEYS], (m, e)
        assert t == tex_escape(e), (t, e)
    return len(expected)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_jobs, bad = jobs_check()
    print(f"jobs.jsonl: {n_jobs} jobs, not completed / not 40 files / missing runtime_s: {bad or 'none'}")
    opt = opt_from_log()
    print("OPT from jovanovic_2019.md:", {k: v for k, v in opt.items() if k in CLASSES})
    opt = {k: v for k, v in opt.items() if k in CLASSES}
    for _, folder, _ in ALGOS:
        for cls in CLASSES:
            ids = sorted(load(folder, cls))
            assert ids == list(range(1, 41)), (folder, cls, ids)
    notes = cross_checks()
    print("\n".join(notes))
    rows = cells(opt)
    write(rows, notes, opt)
    print(f"verified {verify(opt)} rows in Markdown and LaTeX against the result files")


if __name__ == "__main__":
    sys.exit(main())
