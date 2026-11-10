"""Tables of the final CRP-Time campaign (campaign.py), Markdown and LaTeX: Azari, Jovanović and López-Plata under f2,
f2vert and the practice preset on the Caserta classes 3x5, 3x7, 4x6, 5x7 and 6x10.

Per algorithm, time model and class: mean ± standard deviation (n − 1) over the 40 instances of every crane-time
metric the result files store (f2, f2vert, crane time with acceleration; crane_time, time and objective_value are
checked to equal the optimised one; crane_time_rmgc, the legacy RMGC model, is not reported), relocations and
runtime_s, and the number of instances that hit López-Plata's time limit or node budget.

Only campaign results are used: every file read by make_table.load() must be the campaign's result file of its
instance and pass campaign.py's done() (written after the archive time in campaign.json, with runtime_s, train_wall_s,
crane_time_accel and the saved plan, under the block's objective digest), and its algo_config must equal the values
the campaign ran with. Commit, Python version, archive time and machine come from campaign.json. make_table.py and
its outputs document the October runs and are not touched.

  python docs/benchmark/make_campaign_tables.py docs/validation/data/final_campaign_20261029
"""
import json
import re
import sys
from pathlib import Path

import campaign as cp
import make_table as mt
from core.benchmarks.caserta import parse_caserta_filename

OUT_DIR = mt.OUT_DIR
STEM = "crp_time_campaign_tables"
FOLDERS = dict(zip(cp.FIRST_CAMPAIGN, mt.ALGOS))  # campaign key -> (display name, result folder, LaTeX name)
PRESET_LABELS = {"f2": "f2", "f2vert": "f2vert", "practice": "f2vert (practice parameters)"}
TEX_LABELS = {"f2": "$f_2$", "f2vert": r"$f_2^{\mathrm{vert}}$",
              "practice": r"$f_2^{\mathrm{vert}}$ (practice parameters)"}
F2, F2V, ACC = "crane_time_f2", "crane_time_vertical", "crane_time_accel"
TIME_KEYS = ("stack_s_per_stack", "pickup_place_s", "empty_vertical_s_per_tier", "loaded_vertical_s_per_tier",
             "outside_height")
AZARI_NOTE = "Azari's GBH (3 s) and CSUM (5 s) time limits are not recorded per instance."
# The GBH fallback and the source of the limits (docs/validation/data/azari_repeat_20261106/README.md).
AZARI_FALLBACK = (
    "On 6×10 #20, #21, #23, #29 and #39 CSUM finds no complete solution within its 5 s and returns the GBH plan, "
    "in the campaign and in all 10 repetitions of 6 Nov 2026 (`docs/validation/data/azari_repeat_20261106/`), "
    "under all three time models. On 6×10 #38 under f2vert this happened in one repetition (rep04); the main "
    "objective of #38 under f2vert spreads by 8.0 % over the 11 runs ((max − min) / min; three different plans). Of "
    "the 240 instance–time-model pairs of 5×7 and 6×10, 236 give the same plan in all 11 runs (campaign and 10 "
    "repetitions); on the other three (6×10 f2 #17 and #30, f2vert #19) the main objective spreads by at most "
    "0.24 %. The limits 3 s + 5 s are the paper's for the Lee & Lee (2010) instances (Azari et al. 2017, p. 316); "
    "applying them to the Caserta instances is our choice, not the paper's."
)


def tex_note(text):
    """A Markdown note as LaTeX text: &, #, _ and × escaped, `code` as \\texttt, then make_table.tex_cell (%, dashes)."""
    for a, b in (("&", r"\&"), ("#", r"\#"), ("_", r"\_"), ("×", r"$\times$")):
        text = text.replace(a, b)
    return mt.tex_cell(re.sub(r"`([^`]*)`", r"\\texttt{\1}", text))


def columns(preset):
    """(metric key or None, Markdown header, LaTeX header) of a time model's table: the optimised objective first,
    then the other crane-time value (none for the practice preset, which has no tpp), the crane time with
    acceleration, relocations and runtime_s."""
    if preset == "f2":
        crane = [(F2, "f2 (s), optimised", r"$f_2$ (s)*"), (F2V, "f2vert (s)", r"$f_2^{\mathrm{vert}}$ (s)")]
    elif preset == "f2vert":
        crane = [(F2V, "f2vert (s), optimised", r"$f_2^{\mathrm{vert}}$ (s)*"), (F2, "f2 (s)", r"$f_2$ (s)")]
    else:
        crane = [(F2V, "f2vert, practice (s), optimised", r"$f_2^{\mathrm{vert}}$ practice (s)*")]
    return crane + [(ACC, "Crane time with acceleration (s)", "Accel.\\ time (s)"),
                    ("relocations", "Relocations", "Relocations"), ("runtime_s", "runtime_s (s)", "Runtime (s)")]


def campaign_runs(campaign, info, algorithm, preset, cls):
    """{instance: metrics} of one block and class via make_table.load(), each checked against the campaign."""
    problem, algo_values, _ = cp.block_values(algorithm, preset)
    folder = FOLDERS[algorithm][1]
    runs = mt.load(folder, cls, time_model=cp.PRESETS[preset])
    layouts = cp.layouts(cls, None)
    assert len(runs) == len(layouts) == 40, (campaign, algorithm, preset, cls, len(runs), len(layouts))
    main = F2 if preset == "f2" else F2V
    for layout in layouts:
        path = cp.result_file(algorithm, problem, layout)
        assert path.parent.name == folder and cp.done(path, problem, info["start"]), path
        data = json.loads(path.read_text(encoding="utf-8"))
        m = runs[parse_caserta_filename(layout)[2]]
        assert data["metrics"] == m and data["algo_config"] == algo_values, path
        assert m["objective_value"] == m["crane_time"] == m["time"] == m[main], path
        assert (F2 in m) == (preset != "practice"), path
    return runs


def row(algorithm, preset, cls, runs):
    v = list(runs.values())
    cells = {"class": cls.replace("x", "×"), "algorithm": FOLDERS[algorithm][0], "n": str(len(v))}
    for key, _, _ in columns(preset):
        values = [x[key] for x in v]
        cells[key] = (mt.runtime_cell(values) if key == "runtime_s"
                      else mt.ms(values, 2 if key == "relocations" else 1))
    if algorithm == "lopez_plata":
        cells["limit"] = f"{int(sum(x['time_limit_hit'] for x in v))}/{len(v)}"
        cells["budget"] = f"{int(sum(x['budget_hit'] for x in v))}/{len(v)}"
    else:
        cells["limit"] = "n/a¹" if algorithm == "azari" else "–"
        cells["budget"] = "–"
    return cells


def tables(campaign, info):
    """{preset: rows}, the classes in order and within each class the algorithms in campaign order."""
    return {p: [row(a, p, c, campaign_runs(campaign, info, a, p, c)) for c in cp.CLASSES for a in cp.campaign_algorithms(info)]
            for p in cp.PRESETS}


def keys(preset):
    return ["class", "algorithm", "n"] + [k for k, _, _ in columns(preset)] + ["limit", "budget"]


def write(campaign, info, rows):
    values = {p: cp.block_values("azari", p)[0] for p in cp.PRESETS}
    models = "; ".join(f"{PRESET_LABELS[p]}: " + ", ".join(f"{k} {values[p][k]:g}" for k in TIME_KEYS)
                       for p in cp.PRESETS)
    algos = "; ".join(f"{FOLDERS[a][0]}: " + ", ".join(f"{k} {v:g}" for k, v in cp.block_values(a, "f2")[1].items())
                      for a in cp.campaign_algorithms(info))
    rel = campaign.relative_to(mt.REPO).as_posix() if campaign.is_relative_to(mt.REPO) else campaign.name
    md = [
        "# CRP-Time final campaign: Azari, Jovanović, López-Plata under f2, f2vert and the practice preset",
        "",
        f"Generated by `docs/benchmark/make_campaign_tables.py` from the campaign `{rel}` (result files in "
        "`results/CRP-Time/`); every file is checked to be the campaign's result of its instance.",
        "",
        f"- **Campaign:** commit `{info['commit'][:7]}`, Python {info['python']}, archive time {info['start']}; "
        f"machine {info['processor']}, {info['cpus']} logical CPUs, {info['platform']}.",
        "- **Runs:** `docs/benchmark/campaign.py`, the web worker of a Workbench batch job, one instance at a time; "
        "every saved plan replayed by `campaign.py check` (restricted BRP, H = T + 2, f2, f2vert and the crane time "
        "with acceleration recomputed).",
        "- **Instances:** Caserta, 40 per class.",
        f"- **Time models (s):** {models}.",
        f"- **Algorithm parameters:** {algos}.",
        "- **Metrics:** the optimised objective first, then the other crane-time value of the same plan (no f2 for "
        "the practice preset). The crane time with acceleration is evaluated only, never optimised. Relocations are "
        "counted, not optimised. runtime_s is the algorithm's own time, from reading the bay to its final plan.",
        "- **Not reported:** `crane_time_rmgc`, stored in every result file, is the legacy RMGC model and not part of "
        "this phase.",
        "- **Format:** mean ± standard deviation (n − 1) over the 40 instances; runtime_s with 3 decimals below "
        "10 s, 1 decimal above. Time limit hit: López-Plata instances that reached the 600 s limit (step 2); node "
        "budget hit: López-Plata instances whose step 1 reached the node budget; Jovanović has neither.",
    ]
    for p in cp.PRESETS:
        cols = columns(p)
        head = ["Class", "Algorithm", "n"] + [h for _, h, _ in cols] + ["Time limit hit", "Node budget hit"]
        md += ["", f"## {PRESET_LABELS[p]}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        md += ["| " + " | ".join(r[k] for k in keys(p)) + " |" for r in rows[p]]
    md += ["", f"¹ {AZARI_NOTE} {AZARI_FALLBACK}"]
    (OUT_DIR / f"{STEM}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    tex = [f"% CRP-Time final campaign, generated by make_campaign_tables.py from {rel} (commit "
           f"{info['commit'][:7]}); see {STEM}.md for the setup."]
    for p in cp.PRESETS:
        cols = columns(p)
        tex += ["", r"\begin{table}[htbp]", r"\centering", r"\small",
                r"\begin{tabular}{llr" + "l" * len(cols) + "rr}", r"\toprule",
                " & ".join(["Class", "Algorithm", "$n$"] + [h for _, _, h in cols] + ["Limit hit", "Budget hit"])
                + r" \\", r"\midrule"]
        for i, r in enumerate(rows[p]):
            if i and i % len(cp.campaign_algorithms(info)) == 0:
                tex.append(r"\midrule")
            name = {FOLDERS[a][0]: FOLDERS[a][2] for a in cp.campaign_algorithms(info)}[r["algorithm"]]
            tex.append(" & ".join([r["class"].replace("×", r"$\times$"), name]
                                  + [mt.tex_cell(r[k]) for k in keys(p)[2:]]) + r" \\")
        tex += [r"\bottomrule", r"\end{tabular}",
                r"\par\smallskip\footnotesize * optimised. $^{1}$ " + AZARI_NOTE + " " + tex_note(AZARI_FALLBACK),
                rf"\caption{{CRP-Time final campaign under {TEX_LABELS[p]}, Caserta instances, 40 per class: mean "
                r"$\pm$ standard deviation. Limit hit: instances that reached L\'{o}pez-Plata's 600 s time limit; "
                r"budget hit: instances whose step 1 reached its node budget.}",
                rf"\label{{tab:crp-time-campaign-{p}}}", r"\end{table}"]
    (OUT_DIR / f"{STEM}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__.rsplit("\n\n", 1)[1])
    campaign = Path(sys.argv[1]).resolve()
    info = cp.load_campaign(campaign)
    print(f"campaign {campaign.name}: commit {info['commit'][:7]}, Python {info['python']}, archive {info['start']}")
    rows = tables(campaign, info)
    write(campaign, info, rows)
    print(f"wrote {STEM}.md and {STEM}.tex: " + ", ".join(f"{p} {len(r)} rows" for p, r in rows.items()))


if __name__ == "__main__":
    sys.exit(main())
