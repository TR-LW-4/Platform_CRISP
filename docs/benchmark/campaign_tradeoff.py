"""Relocations versus crane time in the final CRP-Time campaign: validation against the exact f2 optima, the trade-off
between methods per time model, and the effect of the time model within one method.

Result files are loaded and checked exactly as in make_campaign_tables.py (campaign_runs); the exact f2 optima are
make_table.py's (jovanovic_2019_checks2/opt_all.jsonl, 3x5, 3x7, 4x6). Crane times within TOL seconds count as equal.

A) Per algorithm and class (f2 runs): instances equal to OPT, mean and max gap % (per-instance gap in % of OPT, gaps
   within TOL count as 0), instances below OPT.
B) Per time model and class: per instance, the fastest solution(s) = lowest value of that time model among the three
   algorithms (ties within TOL). Conflict: no fastest solution has the minimum relocation count of the three. For
   conflicts, the absolute increase r_fast - r_min and the relative increase (r_fast - r_min) / r_min, with r_fast the
   lowest relocation count among the fastest solutions; conflicts with r_min = 0 are counted separately and excluded
   from the relative increase only. Also: how often each algorithm is (one of) the fastest, instances with a time
   tie, and the López-Plata time-limit hits.
C) Per algorithm and class: relocations of the f2vert run (and of the practice run) against the f2 run of the same
   instance: instances with more / equal / fewer relocations, and the mean difference (other run − f2 run).

Prints the tables and writes them to <campaign>_tables/tradeoff.md.

  python docs/benchmark/campaign_tradeoff.py docs/validation/data/final_campaign_20261029
"""
import statistics as st
import sys
from pathlib import Path

import campaign as cp
import make_campaign_tables as mct
import make_table as mt

TOL = 1e-6
EXPECTED = {  # A) known values: instances equal to OPT and mean gap % per class 3x5, 3x7, 4x6
    "azari": ((22, 19, 11), (0.46, 0.60, 0.75)),
    "jovanovic": ((40, 40, 37), (0.00, 0.00, 0.02)),  # 4x6: 0.0248 (0.025 at 3 decimals in make_table.py)
    "lopez_plata": ((39, 38, 32), (0.04, 0.03, 0.14)),
}
MAIN = {"f2": mct.F2, "f2vert": mct.F2V, "practice": mct.F2V}


def name(algorithm):
    return mct.FOLDERS[algorithm][0].split(" (")[0]


def table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows]


def part_a(runs, opt):
    rows, diffs = [], []
    for a in cp.FIRST_CAMPAIGN:
        for k, cls in enumerate(mt.OPT_CLASSES):
            r = runs[a, "f2", cls]
            assert sorted(r) == sorted(opt[cls]), (a, cls)
            gaps = {i: r[i][mct.F2] - opt[cls][i] for i in sorted(r)}
            pct = {i: 0.0 if abs(g) <= TOL else 100 * g / opt[cls][i] for i, g in gaps.items()}
            equal = sum(abs(g) <= TOL for g in gaps.values())
            below = sum(g < -TOL for g in gaps.values())
            mean = st.mean(pct.values())
            worst = max(pct, key=pct.get)
            exp_eq, exp_gap = EXPECTED[a][0][k], EXPECTED[a][1][k]
            flag = []
            if equal != exp_eq:
                flag.append(f"equal {equal} ≠ {exp_eq}")
            if f"{mean:.2f}" != f"{exp_gap:.2f}":
                flag.append(f"mean gap {mean:.2f} ≠ {exp_gap:.2f}")
            if below:
                flag.append(f"{below} below OPT")
            diffs += [(name(a), cls, f) for f in flag]
            rows.append([cls.replace("x", "×"), name(a), f"{equal}/40", f"{mean:.2f}",
                         f"{pct[worst]:.2f} (data{cls.replace('x', '-')}-{worst})", str(below),
                         "; ".join(flag) or "as expected"])
    head = ["Class", "Algorithm", "= OPT", "Mean gap %", "Max gap % (instance)", "Below OPT", "Check"]
    return table(head, rows), diffs


def part_b(runs):
    out = []
    for p in cp.PRESETS:
        key = MAIN[p]
        rows = []
        for cls in cp.CLASSES:
            fastest_n = dict.fromkeys(cp.FIRST_CAMPAIGN, 0)
            ties = conflicts = 0
            incr, absolute, zero_min = [], [], []
            for i in sorted(runs["azari", p, cls]):
                t = {a: runs[a, p, cls][i][key] for a in cp.FIRST_CAMPAIGN}
                r = {a: runs[a, p, cls][i]["relocations"] for a in cp.FIRST_CAMPAIGN}
                best = min(t.values())
                fastest = [a for a in cp.FIRST_CAMPAIGN if t[a] - best <= TOL]
                ties += len(fastest) > 1
                for a in fastest:
                    fastest_n[a] += 1
                r_min, r_fast = min(r.values()), min(r[a] for a in fastest)
                if r_fast > r_min:
                    conflicts += 1
                    absolute.append(r_fast - r_min)
                    if r_min == 0:
                        zero_min.append(r_fast)
                    else:
                        incr.append(100 * (r_fast - r_min) / r_min)
            lp = runs["lopez_plata", p, cls]
            rows.append([cls.replace("x", "×"), f"{conflicts}/40",
                         f"{st.mean(absolute):.2f}" if absolute else "–", f"{max(absolute):g}" if absolute else "–",
                         f"{st.mean(incr):.1f}" if incr else "–", f"{max(incr):.1f}" if incr else "–",
                         f"{len(zero_min)} (+{', +'.join(f'{x:g}' for x in zero_min)})" if zero_min else "0"]
                        + [str(fastest_n[a]) for a in cp.FIRST_CAMPAIGN]
                        + [str(ties), f"{int(sum(m['time_limit_hit'] for m in lp.values()))}/40"])
        head = (["Class", "Conflicts", "Mean abs. increase", "Max abs. increase", "Mean rel. increase %",
                 "Max rel. increase %", "r_min = 0 (abs. increase)"]
                + [f"Fastest: {name(a)}" for a in cp.FIRST_CAMPAIGN] + ["Time ties", "López-Plata limit hit"])
        out += ["", f"### {mct.PRESET_LABELS[p]}", ""] + table(head, rows)
    return out


def part_c(runs):
    out = []
    for other in ("f2vert", "practice"):
        rows = []
        for a in cp.FIRST_CAMPAIGN:
            for cls in cp.CLASSES:
                base, run = runs[a, "f2", cls], runs[a, other, cls]
                d = [run[i]["relocations"] - base[i]["relocations"] for i in sorted(base)]
                rows.append([cls.replace("x", "×"), name(a), str(sum(x > 0 for x in d)), str(sum(x == 0 for x in d)),
                             str(sum(x < 0 for x in d)), f"{st.mean(d):+.2f}"])
        label = mct.PRESET_LABELS[other]
        head = ["Class", "Algorithm", f"More in {label}", "Equal", f"Fewer in {label}", f"Mean diff. ({label} − f2)"]
        out += ["", f"### {label} vs f2", ""] + table(head, rows)
    return out


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__.rsplit("\n\n", 1)[1])
    sys.stdout.reconfigure(encoding="utf-8")
    campaign = Path(sys.argv[1]).resolve()
    info = cp.load_campaign(campaign)
    head = (f"Campaign {campaign.name}: commit {info['commit'][:7]}, Python {info['python']}, archive {info['start']}, "
            f"{info['processor']}. Crane times equal within {TOL:g} s.")
    runs = {(a, p, c): mct.campaign_runs(campaign, info, a, p, c)
            for a in cp.FIRST_CAMPAIGN for p in cp.PRESETS for c in cp.CLASSES}
    a_lines, diffs = part_a(runs, mt.exact_opt())
    md = [f"# Final campaign: relocations versus crane time", "", head, "",
          "## A) Validation against the exact f2 optima (f2 runs)", "",
          "Gap % per instance = 100 (f2 − OPT) / OPT, 0 when |f2 − OPT| ≤ 1e-6 s; mean and max over the 40 "
          "instances.", ""] + a_lines
    if diffs:
        print("\n".join(md))
        raise SystemExit("A does not reproduce the known numbers: " + "; ".join(map(str, diffs)))
    md += ["", "## B) Trade-off between methods, per time model", "",
           "Fastest: the solution(s) with the lowest value of the optimised time model (ties within 1e-6 s). "
           "Conflict: no fastest solution has the minimum relocation count of the three. Over the conflict "
           "instances: absolute increase r_fast − r_min and relative increase (r_fast − r_min) / r_min, r_fast the "
           "lowest count among the fastest. Conflicts with r_min = 0 are listed separately (count, absolute "
           "increase) and excluded from the relative increase only. Fastest counts include ties."] + part_b(runs)
    md += ["", "## C) Effect of the time model within one method", "",
           "Relocations per instance of the f2vert run and of the practice run against the f2 run of the same "
           "algorithm."] + part_c(runs)
    print("\n".join(md))
    out = campaign.parent / f"{campaign.name}_tables" / "tradeoff.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main())
