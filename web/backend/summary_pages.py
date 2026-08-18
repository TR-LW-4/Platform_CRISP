"""Plain HTML pages for benchmark mean/std — no React / npm rebuild required."""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from core.benchmark_summary import summarize_result_files, summarize_run_dicts
from core.result_store import load_run

_RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"


def results_root() -> Path:
    return _RESULTS_ROOT


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_summary_html(
    summary: Mapping[str, Any],
    *,
    title: str,
    subtitle: str = "",
) -> str:
    classes = list(summary.get("classes") or [])
    metric_keys: List[str] = []
    from core.benchmark_summary import SKIP_METRIC_KEYS

    preferred = ["relocations", "crane_time", "time", "moves", "shifters"]
    seen = set()
    for row in classes:
        for key in (row.get("metrics") or {}):
            if key not in SKIP_METRIC_KEYS:
                seen.add(key)
    for key in preferred:
        if key in seen:
            metric_keys.append(key)
    for key in sorted(seen):
        if key not in metric_keys:
            metric_keys.append(key)

    header_metrics = "".join(
        f"<th>{_esc(key)} mean</th><th>{_esc(key)} std</th>" for key in metric_keys
    )
    body_rows = []
    for row in classes:
        cells = [
            f"<td><code>{_esc(row.get('label'))}</code></td>",
            f"<td>{_esc(row.get('n'))}</td>",
        ]
        metrics = row.get("metrics") or {}
        for key in metric_keys:
            agg = metrics.get(key) or {}
            cells.append(f"<td>{_esc(_fmt(agg.get('mean')))}</td>")
            cells.append(f"<td>{_esc(_fmt(agg.get('std')))}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    if not body_rows:
        body_rows.append(
            "<tr><td colspan='99'>No grouped benchmark runs "
            "(need layout_file_path in saved results).</td></tr>"
        )

    err = summary.get("error")
    err_html = f"<p class='err'>{_esc(err)}</p>" if err else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{_esc(title)}</title>
  <style>
    body {{ font-family: Georgia, 'Times New Roman', serif; margin: 2rem; color: #1a1a1a;
           background: #f7f5f1; }}
    h1 {{ font-size: 1.6rem; margin-bottom: 0.25rem; }}
    .sub {{ color: #555; margin-bottom: 1.25rem; }}
    a {{ color: #0b3d91; }}
    table {{ border-collapse: collapse; background: #fff; width: 100%;
             box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
    th, td {{ border: 1px solid #ddd; padding: 0.55rem 0.75rem; text-align: left;
              font-variant-numeric: tabular-nums; }}
    th {{ background: #ece8e1; }}
    code {{ font-family: ui-monospace, monospace; }}
    .meta {{ margin: 0.75rem 0 1.25rem; color: #333; }}
    .err {{ color: #9f1239; }}
    nav {{ margin-bottom: 1.5rem; }}
    form {{ margin: 1rem 0 1.5rem; padding: 1rem; background: #fff;
            border: 1px solid #ddd; }}
    label {{ display: inline-block; margin-right: 1rem; margin-bottom: 0.5rem; }}
    input, select, button {{ padding: 0.35rem 0.5rem; margin-right: 0.5rem; }}
    button {{ cursor: pointer; }}
  </style>
</head>
<body>
  <nav><a href="/bench-summary">Benchmark summaries</a> ·
       <a href="/">Workbench</a> ·
       <a href="/docs">API docs</a></nav>
  <h1>{_esc(title)}</h1>
  <p class="sub">{_esc(subtitle)}</p>
  <p class="meta">total_runs={_esc(summary.get('total_runs'))}
    · ungrouped={_esc(summary.get('ungrouped'))}
    · classes={_esc(len(classes))}</p>
  {err_html}
  <table>
    <thead>
      <tr><th>Class</th><th>n</th>{header_metrics}</tr>
    </thead>
    <tbody>
      {''.join(body_rows)}
    </tbody>
  </table>
</body>
</html>
"""


def summarize_saved_runs(
    *,
    problem: Optional[str] = None,
    algorithm: Optional[str] = None,
    limit: int = 120,
) -> Dict[str, Any]:
    """Load newest matching result JSON files and aggregate by class."""
    root = results_root()
    if not root.is_dir():
        return {"classes": [], "ungrouped": 0, "total_runs": 0}

    paths: List[Path] = []
    if problem and algorithm:
        safe_prob = problem.replace(" ", "_").replace("/", "-")
        safe_algo = algorithm.replace(" ", "_").replace("/", "-")
        folder = root / safe_prob / safe_algo
        if folder.is_dir():
            paths = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime)
    else:
        paths = sorted(root.rglob("*.json"), key=lambda p: p.stat().st_mtime)

    # Drop prior batch_summary sidecar files if any
    paths = [p for p in paths if not p.name.startswith("batch_summary_")]
    chosen = paths[-max(1, limit) :] if paths else []
    return summarize_result_files([str(p) for p in chosen])


def save_batch_summary_sidecar(
    result_files: Sequence[str],
    summary: Mapping[str, Any],
    *,
    problem: str,
    algorithm: str,
) -> Optional[Path]:
    """Persist summary JSON next to algorithm results for later HTML viewing."""
    if not result_files:
        return None
    safe_prob = problem.replace(" ", "_").replace("/", "-")
    safe_algo = algorithm.replace(" ", "_").replace("/", "-")
    folder = results_root() / safe_prob / safe_algo
    folder.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = folder / f"batch_summary_{ts}.json"
    payload = {
        "problem": problem,
        "algorithm": algorithm,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "result_files": list(result_files),
        "summary": dict(summary),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def list_batch_summary_files(limit: int = 30) -> List[Dict[str, Any]]:
    root = results_root()
    if not root.is_dir():
        return []
    files = sorted(root.rglob("batch_summary_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out: List[Dict[str, Any]] = []
    for path in files[:limit]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append({
            "file": str(path),
            "problem": data.get("problem", "?"),
            "algorithm": data.get("algorithm", "?"),
            "timestamp": data.get("timestamp", ""),
            "n_files": len(data.get("result_files") or []),
        })
    return out


def load_batch_summary_file(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("summary") or summarize_run_dicts([])


def render_index_html(
    *,
    jobs: Sequence[Mapping[str, Any]],
    saved: Sequence[Mapping[str, Any]],
) -> str:
    job_rows = []
    for job in jobs:
        jid = job.get("id", "")
        job_rows.append(
            "<tr>"
            f"<td><a href='/bench-summary/job/{_esc(jid)}'>{_esc(jid[:12])}…</a></td>"
            f"<td>{_esc(job.get('status'))}</td>"
            f"<td>{_esc(job.get('problem_name'))}</td>"
            f"<td>{_esc(job.get('algorithm_name'))}</td>"
            f"<td>{_esc(job.get('batch_total'))}</td>"
            f"<td>{_esc(job.get('finished_at') or job.get('started_at'))}</td>"
            "</tr>"
        )
    if not job_rows:
        job_rows.append("<tr><td colspan='6'>No in-memory jobs (restart clears them).</td></tr>")

    saved_rows = []
    for item in saved:
        saved_rows.append(
            "<tr>"
            f"<td><a href='/bench-summary/file?path={_esc(item.get('file'))}'>"
            f"{_esc(item.get('timestamp'))}</a></td>"
            f"<td>{_esc(item.get('problem'))}</td>"
            f"<td>{_esc(item.get('algorithm'))}</td>"
            f"<td>{_esc(item.get('n_files'))}</td>"
            "</tr>"
        )
    if not saved_rows:
        saved_rows.append(
            "<tr><td colspan='4'>No saved batch_summary_*.json yet. "
            "Run a Caserta batch, or use the form below on existing results.</td></tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Benchmark summaries</title>
  <style>
    body {{ font-family: Georgia, 'Times New Roman', serif; margin: 2rem; color: #1a1a1a;
           background: #f7f5f1; }}
    h1, h2 {{ margin-bottom: 0.4rem; }}
    a {{ color: #0b3d91; }}
    table {{ border-collapse: collapse; background: #fff; width: 100%; margin-bottom: 1.5rem;
             box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
    th, td {{ border: 1px solid #ddd; padding: 0.5rem 0.7rem; text-align: left; }}
    th {{ background: #ece8e1; }}
    form {{ margin: 1rem 0 1.5rem; padding: 1rem; background: #fff; border: 1px solid #ddd; }}
    label {{ display: block; margin: 0.4rem 0; }}
    input, button {{ padding: 0.4rem 0.6rem; }}
    .hint {{ color: #555; max-width: 48rem; }}
  </style>
</head>
<body>
  <p><a href="/">← Workbench</a></p>
  <h1>Benchmark summaries</h1>
  <p class="hint">
    No npm required. After a Caserta / Zhu batch finishes, open this page for
    class-wise <strong>mean ± std</strong>. In-memory jobs disappear after server
    restart; use saved summaries or re-aggregate from <code>results/</code>.
  </p>

  <h2>From results on disk</h2>
  <form method="get" action="/bench-summary/from-results">
    <label>Problem <input name="problem" value="CRP-R" size="20"/></label>
    <label>Algorithm <input name="algorithm" value="Caserta (2012) HEUR" size="40"/></label>
    <label>Newest N files <input name="limit" type="number" value="120" min="1" max="5000"/></label>
    <button type="submit">Aggregate</button>
  </form>

  <h2>In-memory jobs</h2>
  <table>
    <thead><tr><th>Job</th><th>Status</th><th>Problem</th><th>Algorithm</th><th>Batch</th><th>Time</th></tr></thead>
    <tbody>{''.join(job_rows)}</tbody>
  </table>

  <h2>Saved batch summaries</h2>
  <table>
    <thead><tr><th>Timestamp</th><th>Problem</th><th>Algorithm</th><th>Files</th></tr></thead>
    <tbody>{''.join(saved_rows)}</tbody>
  </table>
</body>
</html>
"""
