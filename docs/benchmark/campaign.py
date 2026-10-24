"""Final CRP-Time benchmark campaign: Azari, Jovanović and López-Plata under f2, f2vert and the practice preset on
the Caserta classes 3x5, 3x7, 4x6, 5x7 and 6x10 (40 instances each).

Every instance runs through the web worker (web.backend.jobs._worker_main), the function a Workbench batch job runs,
with the problem and algorithm values built as the web API builds them (_config_values, resolve_values), so the run,
its timing and the saved result file are those of a Workbench run. One command runs one block (algorithm and preset),
class by class, one instance at a time; it can be stopped and started again and continues with the instances that
have no result of this campaign yet. Works on Windows and Linux; on another machine, check out the campaign commit
and copy benchmark/Caserta_dataset/ (gitignored) into the checkout first.

  python docs/benchmark/campaign.py archive <campaign_dir>
      copy results/ to <campaign_dir>/results_before/ with an md5 manifest, check the copy, and record the start
      time, commit and machine in <campaign_dir>/campaign.json (refuses an existing campaign)
  python docs/benchmark/campaign.py run <campaign_dir> <algorithm> <preset> [--classes 3x5,3x7] [--instances 1-10]
      run one block; refuses when the checkout differs from the recorded commit, the Python version differs from
      the recorded one, or another run is active
  python docs/benchmark/campaign.py status <campaign_dir> [<algorithm> <preset>] [--classes ...] [--instances ...]
  python docs/benchmark/campaign.py check <campaign_dir> [<algorithm> <preset>] [--classes ...] [--instances ...]
      replay every saved plan independently against its Caserta file (only the top container of the target's stack
      moves, retrievals in priority order, no stack above H = T + 2, an empty bay at the end, tiers as saved),
      recompute relocations, f2 (not for the practice preset), f2vert and the crane time with acceleration, and
      compare them with the saved metrics; then list the files under results/ that differ from the manifest

algorithm: azari | jovanovic | lopez_plata. preset: f2 | f2vert | practice.
An instance counts as done when its result file parses, has the objective suffix of the block, was written after
the campaign start, and holds runtime_s, train_wall_s, crane_time_accel and the saved plan (files of earlier runs
have none of the last three). An interrupted instance is run again from the start.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RESULTS = REPO / "results"
ALGORITHMS = {
    "azari": "Azari et al. (2017) CSUM",
    "jovanovic": "Jovanović et al. (2019) ACO (CRP-Time)",
    "lopez_plata": "López-Plata et al. (2019) A∗-Based Heuristic",
}
PRESETS = {"f2": "f2", "f2vert": "f2_vertical", "practice": "f2_vertical_practice"}   # CRP-Time time-model option
CLASSES = ["3x5", "3x7", "4x6", "5x7", "6x10"]
DONE_KEYS = ("runtime_s", "train_wall_s", "crane_time_accel")
HEARTBEAT_S = 30.0
STALE_S = 120.0


# ------------------------------------------------------------------ helpers

def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log(campaign: Path, msg: str) -> None:
    line = f"{now()} {msg}"
    print(line, flush=True)
    with (campaign / "campaign.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): md5(p) for p in sorted(root.rglob("*")) if p.is_file()}


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def checkout() -> dict:
    return {"commit": git("rev-parse", "HEAD"), "modified": git("status", "--porcelain", "--untracked-files=no")}


def cpu_name() -> str:
    try:   # Linux; platform.processor() is often empty there
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def keep_awake(on: bool) -> None:
    """Windows only: no sleep while a block runs. On Linux use e.g. `systemd-inhibit` around the command."""
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))


def parse_range(text):
    if not text:
        return None
    lo, _, hi = text.partition("-")
    return set(range(int(lo), int(hi or lo) + 1))


def block_values(algorithm: str, preset: str):
    """Problem and algorithm values exactly as the web API builds them for a Workbench job."""
    from core.registry import get_algorithm_class
    from problems.CRP_Time import CRP_Time
    from web.backend.api import _config_values
    problem = CRP_Time.resolve_values(_config_values(CRP_Time.config_schema(), {"time_model": PRESETS[preset]}))
    cls = get_algorithm_class(ALGORITHMS[algorithm])
    return problem, _config_values(cls.config_schema(), {}), cls


def layouts(cls_name: str, instances):
    from core.benchmarks.caserta import parse_caserta_filename
    from web.backend.benchmarks import resolve_paths
    h, w = map(int, cls_name.split("x"))
    paths = resolve_paths("caserta", [{"h": h, "ws": [w]}])
    return [p for p in paths if instances is None or parse_caserta_filename(p)[2] in instances]


def result_file(algorithm: str, problem: dict, layout: Path) -> Path:
    from core.results.store import result_path_for
    return result_path_for("CRP-Time", ALGORITHMS[algorithm], layout, prob_config=problem)


def done(path: Path, problem: dict, start: str) -> bool:
    from core.results.store import _objective_result_suffix
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    metrics = data.get("metrics") or {}
    return (data.get("timestamp", "") >= start and all(k in metrics for k in DONE_KEYS) and bool(data.get("moves"))
            and _objective_result_suffix(data.get("prob_config") or {}) == _objective_result_suffix(problem))


def load_campaign(campaign: Path) -> dict:
    path = campaign / "campaign.json"
    if not path.is_file():
        raise SystemExit(f"{path} not found: run `archive` first")
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ archive

def cmd_archive(args) -> None:
    campaign = Path(args.campaign).resolve()
    if (campaign / "campaign.json").exists():
        raise SystemExit(f"{campaign} already holds a campaign; choose another folder")
    campaign.mkdir(parents=True, exist_ok=True)
    before = campaign / "results_before"
    if before.exists():
        raise SystemExit(f"{before} exists; remove it or choose another folder")
    source = manifest(RESULTS)
    shutil.copytree(RESULTS, before)
    copy = manifest(before)
    if copy != source:
        raise SystemExit("the copy of results/ differs from the original; nothing recorded")
    (campaign / "results_before.md5").write_text("".join(f"{h}  {p}\n" for p, h in source.items()), encoding="utf-8")
    info = {"start": now(), **checkout(), "python": sys.version.split()[0], "platform": platform.platform(),
            "processor": cpu_name(), "cpus": os.cpu_count(),
            "host": platform.node(), "results_files": len(source)}
    (campaign / "campaign.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    log(campaign, f"archive: {len(source)} files of results/ copied to {before.name}, md5 checked; "
                  f"commit {info['commit'][:7]}{' (modified files!)' if info['modified'] else ''}")


# ------------------------------------------------------------------ run

def _worker(alive, args) -> None:
    """Child process: the web worker, stopped as soon as the driver is gone (its end of the pipe closes)."""
    def watch():
        try:
            alive.recv()
        except (EOFError, OSError):
            pass
        os._exit(3)
    threading.Thread(target=watch, daemon=True).start()
    from web.backend.jobs import _worker_main
    _worker_main(*args)


def pid_alive(pid: int) -> bool:
    if sys.platform == "win32":   # os.kill(pid, 0) would terminate the process on Windows
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return code.value == 259   # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Lock:
    """One run per campaign folder: a lock file with the driver's pid and host, whose mtime the driver refreshes.
    It is taken over when its driver is gone: on the same host when the pid no longer runs, otherwise when the
    lock has not been refreshed for STALE_S."""

    def __init__(self, campaign: Path):
        self.path = campaign / "run.lock"
        self._stop = threading.Event()

    def __enter__(self):
        if self.path.exists():
            held = json.loads(self.path.read_text(encoding="utf-8"))
            fresh = time.time() - self.path.stat().st_mtime < STALE_S
            if (held["host"] == platform.node() and pid_alive(held["pid"])) or (held["host"] != platform.node()
                                                                                and fresh):
                raise SystemExit(f"another run is active: {held}")
            print(f"taking over the lock of a run that is gone: {held}", flush=True)
        self.path.write_text(json.dumps({"pid": os.getpid(), "host": platform.node(), "since": now()}),
                             encoding="utf-8")

        def beat():
            while not self._stop.wait(HEARTBEAT_S):
                os.utime(self.path)
        threading.Thread(target=beat, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self.path.unlink(missing_ok=True)


def run_class(campaign, algorithm, preset, cls_name, problem, algo_values, cls, start, instances) -> bool:
    paths = layouts(cls_name, instances)
    skip = [str(p) for p in paths if done(result_file(algorithm, problem, p), problem, start)]
    todo = len(paths) - len(skip)
    tag = f"{algorithm} {preset} {cls_name}"
    if not todo:
        log(campaign, f"{tag}: all {len(paths)} done")
        return True
    log(campaign, f"{tag}: start, {todo} to run, {len(skip)} done before")
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    stop = ctx.Event()
    alive_r, alive_w = ctx.Pipe(duplex=False)
    args = ("CRP-Time", ALGORITHMS[algorithm], problem, algo_values, results, stop, "caserta",
            [str(p) for p in paths], cls.category, skip)
    proc = ctx.Process(target=_worker, args=(alive_r, args), name=f"campaign-{algorithm}-{preset}-{cls_name}")
    t0 = time.monotonic()
    proc.start()
    alive_r.close()
    failure = None
    try:
        while proc.is_alive() or not results.empty():
            try:
                message = results.get(timeout=1.0)
            except queue.Empty:
                continue
            if type(message).__name__ == "WorkerFailure":
                failure = message
    except KeyboardInterrupt:
        stop.set()
        log(campaign, f"{tag}: interrupted, waiting for the worker to stop")
        proc.join(30)
        if proc.is_alive():
            proc.terminate()
        raise
    finally:
        proc.join(5)
        alive_w.close()
    now_done = sum(done(result_file(algorithm, problem, p), problem, start) for p in paths)
    status = "completed" if now_done == len(paths) else "INCOMPLETE"
    log(campaign, f"{tag}: {status}, {now_done}/{len(paths)} done, exit code {proc.exitcode}, "
                  f"wall {time.monotonic() - t0:.1f} s" + (f", worker failure: {failure.message}" if failure else ""))
    return now_done == len(paths)


def cmd_run(args) -> None:
    campaign = Path(args.campaign).resolve()
    info = load_campaign(campaign)
    current = checkout()
    if current["commit"] != info["commit"] or current["modified"]:
        raise SystemExit(f"checkout {current['commit'][:7]} (modified: {bool(current['modified'])}) differs from the "
                         f"campaign commit {info['commit'][:7]}; check out that commit with no modified files")
    python = sys.version.split()[0]
    if python != info["python"]:
        # Float sums differ between Python versions (sum() is compensated since 3.12), and so do runtimes.
        raise SystemExit(f"Python {python} ({sys.executable}) differs from the campaign's Python {info['python']}; "
                         "run with the same interpreter")
    problem, algo_values, cls = block_values(args.algorithm, args.preset)
    instances = parse_range(args.instances)
    classes = args.classes.split(",") if args.classes else CLASSES
    with Lock(campaign):
        keep_awake(True)
        try:
            log(campaign, f"block {args.algorithm} {args.preset}: classes {','.join(classes)}"
                          f"{f', instances {args.instances}' if instances else ''}; commit {current['commit'][:7]}")
            ok = all([run_class(campaign, args.algorithm, args.preset, c, problem, algo_values, cls, info["start"],
                                instances) for c in classes])
            log(campaign, f"block {args.algorithm} {args.preset}: {'complete' if ok else 'NOT complete, run it again'}")
        finally:
            keep_awake(False)
    sys.exit(0 if ok else 1)


# ------------------------------------------------------------------ status

def blocks(args):
    algos = [args.algorithm] if args.algorithm else list(ALGORITHMS)
    presets = [args.preset] if args.preset else list(PRESETS)
    classes = args.classes.split(",") if args.classes else CLASSES
    return [(a, p, c) for a in algos for p in presets for c in classes]


def cmd_status(args) -> None:
    campaign = Path(args.campaign).resolve()
    start = load_campaign(campaign)["start"]
    instances = parse_range(args.instances)
    for algorithm, preset, cls_name in blocks(args):
        problem = block_values(algorithm, preset)[0]
        paths = layouts(cls_name, instances)
        n = sum(done(result_file(algorithm, problem, p), problem, start) for p in paths)
        print(f"{algorithm:12s} {preset:9s} {cls_name:5s} {n:3d}/{len(paths)}")


# ------------------------------------------------------------------ check (independent of core/objectives.py)

def read_dat(path: Path):
    lines = [ln.split() for ln in path.read_text().splitlines() if ln.strip()]
    n_stacks, n = int(lines[0][0]), int(lines[0][1])
    stacks = [[int(x) for x in ln[1:1 + int(ln[0])]] for ln in lines[1:1 + n_stacks]]
    assert sum(map(len, stacks)) == n and n % n_stacks == 0, path
    return stacks, n // n_stacks + 2


def accel_time(distance: float, speed: float, accel: float) -> float:
    if distance <= 0:
        return 0.0
    return distance / speed + speed / accel if distance >= speed * speed / accel else 2.0 * math.sqrt(distance / accel)


def replay(stacks, H, moves, pc):
    """Problems of the saved plan under the restricted rule, and relocations, f2, f2vert and the crane time with
    acceleration recomputed from it (stack s of the bay is position (s, 1); the truck is at stack 0)."""
    ts, tpp = float(pc["stack_s_per_stack"]), float(pc["pickup_place_s"])
    tr, hout = float(pc["empty_vertical_s_per_tier"]) + float(pc["loaded_vertical_s_per_tier"]), float(pc["outside_height"])
    st = [list(s) for s in stacks]
    remaining = sorted(c for s in st for c in s)
    problems, rel, f2, f2v, acc = [], 0, 0.0, 0.0, 0.0
    h_max = H + 1
    for k, m in enumerate(moves):
        s = m["from"][0] - 1
        if m["from"][1] != 1 or not 0 <= s < len(st) or not st[s]:
            problems.append(f"move {k}: bad source {m['from']}")
            break
        c, target, tier = st[s][-1], remaining[0], len(st[s])
        if m["container_id"] != c or m["from_tier"] != tier:
            problems.append(f"move {k}: saved container {m['container_id']} tier {m['from_tier']}, top is {c} at {tier}")
        if m["to"] is None:
            if c != target:
                problems.append(f"move {k}: retrieves {c}, next in order is {target}")
            st[s].pop()
            remaining.remove(c)
            dist, drop = s + 1, hout
        else:
            d = m["to"][0] - 1
            if target not in st[s] or c == target:
                problems.append(f"move {k}: relocation from stack {s + 1}, not a blocker of {target}")
            if m["to"][1] != 1 or not 0 <= d < len(st) or d == s:
                problems.append(f"move {k}: bad destination {m['to']}")
                break
            st[d].append(st[s].pop())
            if len(st[d]) > H:
                problems.append(f"move {k}: stack {d + 1} above H = {H}")
            if m.get("to_tier") != len(st[d]):
                problems.append(f"move {k}: saved destination tier {m.get('to_tier')}, replay {len(st[d])}")
            rel += 1
            dist, drop = abs(s - d), len(st[d])
        f2 += 2 * ts * dist + tpp
        f2v += 2 * ts * dist + tr * (2 * h_max - tier - drop)
        up, down = (h_max - tier) * 2.591, (h_max - drop) * 2.591
        for v_tr, a_tr, a_sp in ((1.0, 0.33, 0.3), (1.2, 0.35, 0.5)):
            acc += (2 * accel_time(dist * 2.54, v_tr, a_tr) + accel_time(up, 1.5, a_sp) + accel_time(up, 0.75, a_sp)
                    + accel_time(down, 0.75, a_sp) + accel_time(down, 1.5, a_sp)) / 2
    if any(st):
        problems.append("bay not empty at the end")
    return problems, {"relocations": float(rel), "crane_time_f2": f2, "crane_time_vertical": f2v,
                      "crane_time_accel": acc}


def cmd_check(args) -> None:
    campaign = Path(args.campaign).resolve()
    info = load_campaign(campaign)
    instances = parse_range(args.instances)
    n = bad = missing = 0
    for algorithm, preset, cls_name in blocks(args):
        problem = block_values(algorithm, preset)[0]
        for layout in layouts(cls_name, instances):
            path = result_file(algorithm, problem, layout)
            if not done(path, problem, info["start"]):
                missing += 1
                continue
            n += 1
            data = json.loads(path.read_text(encoding="utf-8"))
            metrics, pc = data["metrics"], data["prob_config"]
            stacks, H = read_dat(layout)
            problems, recomputed = replay(stacks, H, data["moves"], pc)
            if preset == "practice":
                recomputed.pop("crane_time_f2")
                if "crane_time_f2" in metrics:
                    problems.append("f2 reported for the practice preset")
            for key, value in recomputed.items():
                if key not in metrics or abs(metrics[key] - value) > 1e-6:
                    problems.append(f"{key}: saved {metrics.get(key)}, recomputed {value}")
            main = "crane_time_f2" if preset == "f2" else "crane_time_vertical"
            if metrics.get("objective_value") != metrics.get(main):
                problems.append(f"objective_value {metrics.get('objective_value')} is not {main}")
            if not metrics["runtime_s"] <= metrics["train_wall_s"]:
                problems.append(f"runtime_s {metrics['runtime_s']} > train_wall_s {metrics['train_wall_s']}")
            if problems:
                bad += 1
                print(f"PROBLEM {algorithm} {preset} {layout.name}: {problems[:3]}")
    print(f"checked {n} results: with problems {bad}; not done {missing}")
    before = {}
    for line in (campaign / "results_before.md5").read_text(encoding="utf-8").splitlines():
        h, p = line.split("  ", 1)
        before[p] = h
    after = manifest(RESULTS)
    changed = sorted(p for p in after if before.get(p) != after[p])
    removed = sorted(p for p in before if p not in after)
    campaign_files = set()
    for a in ALGORITHMS:
        for pr in PRESETS:
            values = block_values(a, pr)[0]
            campaign_files |= {result_file(a, values, lp).relative_to(RESULTS).as_posix()
                               for c in CLASSES for lp in layouts(c, None)}
    other = [p for p in changed if p not in campaign_files]
    print(f"results/ against the manifest: {len(changed)} new or changed, {len(removed)} removed, "
          f"of which outside the campaign result files: {len(other) + len(removed)}")
    for p in (other + removed)[:20]:
        print("  OUTSIDE", p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("archive")
    a.add_argument("campaign")
    for name, positional in (("run", True), ("status", False), ("check", False)):
        p = sub.add_parser(name)
        p.add_argument("campaign")
        p.add_argument("algorithm", nargs=None if positional else "?", choices=list(ALGORITHMS))
        p.add_argument("preset", nargs=None if positional else "?", choices=list(PRESETS))
        p.add_argument("--classes")
        p.add_argument("--instances")
    args = ap.parse_args()
    {"archive": cmd_archive, "run": cmd_run, "status": cmd_status, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
