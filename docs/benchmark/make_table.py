"""Comparison table for Wei from the web runs (result files and jobs.jsonl), Markdown and LaTeX.

Azari and López-Plata come from the overnight run of 5-6 Oct 2026 (code 0fc9a87). Jovanović comes from the rerun of
10 Oct 2026 with the valid lower bound LB_f (0cb0f50, jovanovic_2019.md §3.3); its overnight results with the
literal eq. 44 are archived and shown in a separate table for comparison.

OPT is read from docs/validation/jovanovic_2019.md ("Checks on the O_v difference": O_f,30 exact vs Table 5).
Cross-checks per instance: Jovanović against the in-process reference run of 0cb0f50 (refactor_golden), all classes;
the archived literal Jovanović results against step 5 (step5_b379031.jsonl, check Of30, seed 0) and López-Plata
against V35 (f2_bench.jsonl), on 3x5, 3x7, 4x6. After writing, every cell is recomputed from the result files and
compared with the written Markdown and LaTeX.

Inputs (all local, not in git): results/CRP-Time/ (gitignored), and in docs/validation/data/ (excluded locally)
overnight_2026-10-05/ (jobs.jsonl, driver.log, results_jovanovic_literal_eq44/), rerun_jovanovic_2026-10-10/
(jobs.jsonl, driver.log), refactor_golden/head_jovanovic_f2_0cb0f50.jsonl, jovanovic_2019_step5/step5_b379031.jsonl
and lopez_plata_2019/f2_bench.jsonl. The machine of the runs is recorded below, so the output does not depend on the
machine that regenerates it.

  python docs/benchmark/make_table.py      (from any directory)
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "docs" / "validation" / "data"
RUN_DIR = DATA / "overnight_2026-10-05"
RERUN_DIR = DATA / "rerun_jovanovic_2026-10-10"
RESULTS = REPO / "results" / "CRP-Time"
ARCHIVE = RUN_DIR / "results_jovanovic_literal_eq44"
JOV_REFERENCE = DATA / "refactor_golden" / "head_jovanovic_f2_0cb0f50.jsonl"
OUT_DIR = REPO / "docs" / "benchmark"
# Machine of both runs (the Python of the conda env rl that ran the server)
RUN_MACHINE = ("AMD Ryzen 7 8700F 8-Core Processor", "3.11.16", "Windows 11 (10.0.26300)")
CLASSES = ["3x5", "3x7", "4x6", "5x7", "6x10"]
ALGOS = [  # display name, result folder, LaTeX name
    ("Azari et al. (2017) CSUM", "Azari_et_al._(2017)_CSUM", r"Azari et al.\ (2017) CSUM"),
    ("Jovanović et al. (2019) ACO", "Jovanović_et_al._(2019)_ACO_(CRP-Time)", r"Jovanovi\'{c} et al.\ (2019) ACO"),
    ("López-Plata et al. (2019) A∗-Based Heuristic", "López-Plata_et_al._(2019)_A∗-Based_Heuristic",
     r"L\'{o}pez-Plata et al.\ (2019) A$^\ast$-Based Heuristic"),
]
JOV_FOLDER = ALGOS[1][1]
LITERAL_HEADING = "Jovanović with the literal eq. 44 (overnight run, for comparison)"
TOL = 1e-9


def load(folder, cls, base=RESULTS):
    runs = {}
    for f in sorted((base / folder).glob(f"caserta_{cls}_*.json")):
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


def row_of(name, runs, cls, opt):
    v = list(runs.values())
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
    return row


def cells(opt):
    """All cells of the main table, computed from the result files."""
    rows = []
    for cls in CLASSES:
        for name, folder, _ in ALGOS:
            rows.append(row_of(name, load(folder, cls), cls, opt))
        rows.append({"class": cls.replace("x", "×"), "algorithm": "OPT (Jovanović et al. 2019, Table 5, O_f,30)" if cls in opt else
                     "OPT: none in Table 5", "n": "–", "f2": f"{opt[cls]:.1f}" if cls in opt else "–", "f2vert": "–",
                     "relocations": "–", "runtime": "–", "limit": "–", "gap": "–"})
    return rows


def literal_cells(opt):
    """Cells of the comparison table: Jovanović with the literal eq. 44, from the archived overnight result files."""
    return [row_of(ALGOS[1][0], load(JOV_FOLDER, cls, ARCHIVE), cls, opt) for cls in CLASSES]


def cross_checks():
    out = []
    jov_now = {}
    for line in open(JOV_REFERENCE, encoding="utf-8"):
        d = json.loads(line)
        if d["time_model"] == "f2":
            jov_now[(d["class"], d["idx"])] = d["records"][-1]["metrics"]
    jov_lit = {}
    for line in open(DATA / "jovanovic_2019_step5" / "step5_b379031.jsonl", encoding="utf-8"):
        d = json.loads(line)
        if d["check"] == "Of30" and d["seed"] == 0:
            jov_lit[(d["size"].replace("-", "x"), d["inst"])] = d
    lp = {}
    for line in open(DATA / "lopez_plata_2019" / "f2_bench.jsonl", encoding="utf-8"):
        d = json.loads(line)
        if d.get("status") == "ok":
            lp[(d["class"].replace("-", "x"), int(d["instance"].rsplit("-", 1)[1]))] = d
    metric_keys = (("crane_time_f2", "crane_time_f2"), ("crane_time_vertical", "crane_time_vertical"),
                   ("relocations", "relocations"))
    bench_keys = (("crane_time_f2", "f2"), ("crane_time_vertical", "f2vert"), ("relocations", "relocations"))
    for label, folder, base, ref, keys, classes in (
            ("Jovanović vs the in-process reference run (0cb0f50, golden.py)", JOV_FOLDER, RESULTS, jov_now,
             metric_keys, CLASSES),
            ("Jovanović with the literal eq. 44 (archived) vs step 5 (Of30, seed 0, b379031)", JOV_FOLDER, ARCHIVE,
             jov_lit, bench_keys, ("3x5", "3x7", "4x6")),
            ("López-Plata vs V35 (f2_bench.jsonl)", ALGOS[2][1], RESULTS, lp, bench_keys, ("3x5", "3x7", "4x6"))):
        for cls in classes:
            runs = load(folder, cls, base)
            diffs = [(i, k) for i, m in runs.items() for k, rk in keys
                     if (cls, i) not in ref or abs(m[k] - ref[(cls, i)][rk]) > 1e-6]
            n_ref = sum(1 for (c, _) in ref if c == cls)
            out.append(f"{label}, {cls}: {len(runs)} runs, {n_ref} reference runs, differences: "
                       + ("none" if not diffs else f"{len(diffs)} {diffs[:6]}"))
    return out


def jobs_check(run_dir):
    rows = [json.loads(line) for line in open(run_dir / "jobs.jsonl", encoding="utf-8")]
    bad = [r for r in rows if r["status"] != "completed" or r["result_files"] != 40 or not r["all_have_runtime_s"]]
    return len(rows), bad


def run_info(run_dir):
    log = (run_dir / "driver.log").read_text(encoding="utf-8").splitlines()
    commit = re.search(r"commit (\w+)", log[0]).group(1)
    return commit, log[0][:19], log[-1][:19]


HEAD = ["Class", "Algorithm", "n", "f2 (s)", "f2vert (s)", "Relocations", "runtime_s (s)", "Limit hit", "f2 − OPT (s, %)"]
KEYS = ["class", "algorithm", "n", "f2", "f2vert", "relocations", "runtime", "limit", "gap"]
LIT_HEAD = ["Class", "n", "f2 (s)", "f2vert (s)", "Relocations", "runtime_s (s)", "f2 − OPT (s, %)"]
LIT_KEYS = ["class", "n", "f2", "f2vert", "relocations", "runtime", "gap"]


def tex_cell(v):
    return v.replace("±", r"$\pm$").replace("%", r"\%").replace("–", "--").replace("−", "$-$")


def tex_escape(row):
    name = {a[0]: a[2] for a in ALGOS}.get(row["algorithm"])
    if name is None:
        name = (r"OPT (Jovanovi\'{c} et al.\ 2019, Table 5, $O_{f,30}$)" if row["algorithm"].startswith("OPT (")
                else r"OPT: none in Table 5")
    return [tex_cell(v) for v in [row["class"].replace("×", r"$\times$"), name] + [row[k] for k in KEYS[2:]]]


def tex_literal(row):
    return [tex_cell(v) for v in [row["class"].replace("×", r"$\times$")] + [row[k] for k in LIT_KEYS[1:]]]


def write(rows, lit_rows, notes):
    cpu, py, os_ = RUN_MACHINE
    commit, t_start, t_end = run_info(RUN_DIR)
    re_commit, re_start, re_end = run_info(RERUN_DIR)
    intro = [
        "# CRP-Time comparison under f2: Azari, Jovanović, López-Plata",
        "",
        "Generated by `docs/benchmark/make_table.py` from the result files in "
        "`results/CRP-Time/` and `jobs.jsonl` of the web runs; every cell is recomputed from the result files "
        "after writing.",
        "",
        "- **Objective:** f2 (crane time with constant pick-up/place-down effort, Voß & Schwarze 2019) is optimised; "
        "f2vert is reported from the same runs (the platform evaluates the final plan under both). Relocations are "
        "measured, not optimised. Restricted BRP, H = T + 2, CRISP time defaults (ts 1.2 s, tpp 30 s).",
        "- **Instances:** Caserta, all 40 instances per class. Seed 0.",
        f"- **Runs:** web jobs through `/api/jobs` (the Workbench path), strictly one after another, one "
        f"instance at a time in one worker process. Azari and López-Plata: overnight run, code `{commit}`, "
        f"{t_start} – {t_end}. Jovanović: rerun with the same driver, code `{re_commit}`, {re_start} – {re_end}. "
        f"Machine: {cpu}, Python {py}, {os_}. runtime_s is the wall clock of `train()` per instance.",
        "- **Jovanović's lower bound:** since `0cb0f50` Jovanović uses the valid lower bound LB_f, eq. 44 without "
        "the ts per non-well-located container (Wei's decision; deviation from the paper, "
        "`docs/validation/jovanovic_2019.md` §3.3). Its overnight results with the literal eq. 44 are in the "
        "separate table below, for comparison only.",
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
    lit = ["", f"## {LITERAL_HEADING}", "",
           f"Code `{commit}`, same overnight run as Azari and López-Plata; result files archived in "
           "`docs/validation/data/overnight_2026-10-05/results_jovanovic_literal_eq44/` (local).", "",
           "| " + " | ".join(LIT_HEAD) + " |", "|" + "---|" * len(LIT_HEAD)]
    lit += ["| " + " | ".join(r[k] for k in LIT_KEYS) + " |" for r in lit_rows]
    checks = ["", "**Cross-checks (per instance, f2, f2vert and relocations, tolerance 1e-6):**", ""]
    checks += [f"- {n}" for n in notes]
    (OUT_DIR / "crp_time_f2_comparison.md").write_text("\n".join(intro + md + lit + checks) + "\n", encoding="utf-8")
    tex = [
        "% CRP-Time comparison under f2, generated by make_table.py from the web runs; see crp_time_f2_comparison.md for the setup.",
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
            r"\caption{CRP-Time, Caserta instances, 40 per class, seed 0: $f_2$ optimised, $f_2^{\mathrm{vert}}$ reported from the same runs; mean $\pm$ standard deviation. OPT: $O_{f,30}$ column of Table 5 in Jovanovi\'{c} et al.\ (2019). Jovanovi\'{c} with the valid lower bound $LB_f$ (eq.~44 without the $t_s$ per non-well-located container; deviation from the paper). L\'{o}pez-Plata with a 600 s time limit per instance.}",
            r"\label{tab:crp-time-f2-comparison}", r"\end{table}",
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lrlllll}",
            r"\toprule",
            " & ".join(["Class", "$n$", "$f_2$ (s)", r"$f_2^{\mathrm{vert}}$ (s)", "Relocations", "Runtime (s)",
                         r"$f_2 - \mathrm{OPT}$ (s, \%)"]) + r" \\",
            r"\midrule"]
    tex += [" & ".join(tex_literal(r)) + r" \\" for r in lit_rows]
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\caption{For comparison: Jovanovi\'{c} et al.\ (2019) ACO with the literal $LB_f$ of eq.~44, overnight run before the change to the valid lower bound; otherwise as Table~\ref{tab:crp-time-f2-comparison}.}",
            r"\label{tab:crp-time-f2-jovanovic-literal}", r"\end{table}"]
    (OUT_DIR / "crp_time_f2_comparison.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def verify(opt):
    expected, expected_lit = cells(opt), literal_cells(opt)

    def md_rows(lines):
        return [[c.strip() for c in line.strip("|").split("|")] for line in lines
                if line.startswith("| ") and "---" not in line][1:]

    md = (OUT_DIR / "crp_time_f2_comparison.md").read_text(encoding="utf-8").splitlines()
    split = md.index(f"## {LITERAL_HEADING}")
    main_md, lit_md = md_rows(md[:split]), md_rows(md[split:])
    tex = (OUT_DIR / "crp_time_f2_comparison.tex").read_text(encoding="utf-8").splitlines()
    split = tex.index(r"\label{tab:crp-time-f2-comparison}")

    def tex_rows(lines):
        return [line[:-3].split(" & ") for line in lines if line.endswith(r" \\") and "$n$" not in line]

    main_tex, lit_tex = tex_rows(tex[:split]), tex_rows(tex[split:])
    assert len(main_md) == len(expected) == len(main_tex), (len(main_md), len(expected), len(main_tex))
    assert len(lit_md) == len(expected_lit) == len(lit_tex), (len(lit_md), len(expected_lit), len(lit_tex))
    for e, m, t in zip(expected, main_md, main_tex):
        assert m == [e[k] for k in KEYS], (m, e)
        assert t == tex_escape(e), (t, e)
    for e, m, t in zip(expected_lit, lit_md, lit_tex):
        assert m == [e[k] for k in LIT_KEYS], (m, e)
        assert t == tex_literal(e), (t, e)
    return len(expected), len(expected_lit)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for run_dir in (RUN_DIR, RERUN_DIR):
        n_jobs, bad = jobs_check(run_dir)
        print(f"{run_dir.name}/jobs.jsonl: {n_jobs} jobs, not completed / not 40 files / missing runtime_s: "
              f"{bad or 'none'}")
    opt = opt_from_log()
    print("OPT from jovanovic_2019.md:", {k: v for k, v in opt.items() if k in CLASSES})
    opt = {k: v for k, v in opt.items() if k in CLASSES}
    for folder, base in [(a[1], RESULTS) for a in ALGOS] + [(JOV_FOLDER, ARCHIVE)]:
        for cls in CLASSES:
            ids = sorted(load(folder, cls, base))
            assert ids == list(range(1, 41)), (folder, base, cls, ids)
    notes = cross_checks()
    print("\n".join(notes))
    write(cells(opt), literal_cells(opt), notes)
    n_main, n_lit = verify(opt)
    print(f"verified {n_main} + {n_lit} rows in Markdown and LaTeX against the result files")


if __name__ == "__main__":
    sys.exit(main())
