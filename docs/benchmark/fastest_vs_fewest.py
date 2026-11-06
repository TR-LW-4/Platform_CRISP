"""Final CRP-Time campaign: per time model and algorithm, on how many instances the fastest solution also has the
fewest relocations of the three algorithms.

Result files are loaded and checked exactly as in make_campaign_tables.py (campaign_runs). Every saved plan is
replayed with campaign.py's own check (independent of core/objectives.py): it must have no problems, and its
relocations and optimised crane time must equal the saved metrics within TOL; otherwise the script stops.
Fastest: the lowest value of the optimised time model among the three algorithms, ties within TOL s count for
each algorithm involved. Prints the counts, writes nothing.

  python docs/benchmark/fastest_vs_fewest.py docs/validation/data/final_campaign_20261029
"""
import json
import sys
from pathlib import Path

import campaign as cp
import make_campaign_tables as mct
from core.benchmarks.caserta import parse_caserta_filename

TOL = 1e-6
MAIN = {"f2": mct.F2, "f2vert": mct.F2V, "practice": mct.F2V}


def recomputed(algorithm, preset, problem, layout, metrics):
    """(crane time, relocations) of the saved plan, after checking them against the saved metrics."""
    path = cp.result_file(algorithm, problem, layout)
    data = json.loads(path.read_text(encoding="utf-8"))
    stacks, H = cp.read_dat(layout)
    problems, values = cp.replay(stacks, H, data["moves"], data["prob_config"])
    key = MAIN[preset]
    assert not problems, (path, problems)
    assert values["relocations"] == metrics["relocations"], (path, values["relocations"], metrics["relocations"])
    assert abs(values[key] - metrics[key]) <= TOL, (path, values[key], metrics[key])
    return values[key], values["relocations"]


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__.rsplit("\n\n", 1)[1])
    sys.stdout.reconfigure(encoding="utf-8")
    campaign = Path(sys.argv[1]).resolve()
    info = cp.load_campaign(campaign)
    print(f"Campaign {campaign.name}: commit {info['commit'][:7]}, Python {info['python']}, archive {info['start']}. "
          f"Crane times equal within {TOL:g} s.")
    checked = 0
    for p in cp.PRESETS:
        fastest, fewest, both = (dict.fromkeys(cp.ALGORITHMS, 0) for _ in range(3))
        agree, ties = {}, 0
        for cls in cp.CLASSES:
            tr = {}
            for a in cp.ALGORITHMS:
                problem = cp.block_values(a, p)[0]
                runs = mct.campaign_runs(campaign, info, a, p, cls)
                for layout in cp.layouts(cls, None):
                    i = parse_caserta_filename(layout)[2]
                    tr[a, i] = recomputed(a, p, problem, layout, runs[i])
                    checked += 1
            agree[cls] = 0
            for i in sorted({i for _, i in tr}):
                t = {a: tr[a, i][0] for a in cp.ALGORITHMS}
                r = {a: tr[a, i][1] for a in cp.ALGORITHMS}
                fast = [a for a in cp.ALGORITHMS if t[a] - min(t.values()) <= TOL]
                ties += len(fast) > 1
                agree[cls] += any(r[a] == min(r.values()) for a in fast)
                for a in cp.ALGORITHMS:
                    fastest[a] += a in fast
                    fewest[a] += r[a] == min(r.values())
                    both[a] += a in fast and r[a] == min(r.values())
        n = 40 * len(cp.CLASSES)
        print(f"\n{mct.PRESET_LABELS[p]}: fastest solution also has the fewest relocations on {sum(agree.values())}/{n}"
              f" instances (time ties {ties}); per class " + ", ".join(f"{c} {v}/40" for c, v in agree.items()))
        for a in cp.ALGORITHMS:
            print(f"  {a:12s} fastest {fastest[a]:3d}, fewest relocations {fewest[a]:3d}, "
                  f"fastest and fewest {both[a]:3d}")
    print(f"\n{checked} result files checked")


if __name__ == "__main__":
    sys.exit(main())
