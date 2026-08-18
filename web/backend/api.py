"""FastAPI application exposing CRISP registries and isolated jobs."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.registry import (
    get_algorithm_class,
    get_algorithm_info,
    get_problem_class,
    get_problem_info,
)
from core.compare_summary import (
    compare_algorithms,
    delete_algorithm_folder,
    delete_result_file,
    list_result_folders,
)
from core.result_store import list_saved_runs, load_run
from web.backend.benchmarks import (
    SOURCE_RANDOM,
    available_for_problem,
    resolve_paths,
    validate_source_for_problem,
)
from web.backend.jobs import JobManager
from web.backend.serialization import json_safe
from web.backend.summary_pages import (
    list_batch_summary_files,
    load_batch_summary_file,
    render_index_html,
    render_summary_html,
    results_root,
    summarize_saved_runs,
)


FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
manager = JobManager()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    manager.shutdown()


app = FastAPI(
    title="Platform CRISP API",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunInput(BaseModel):
    problem_name: str
    algorithm_name: str
    problem_config: Dict[str, Any] = Field(default_factory=dict)
    algorithm_config: Dict[str, Any] = Field(default_factory=dict)
    instance_source: str = SOURCE_RANDOM
    benchmark_queue: List[Dict[str, Any]] = Field(default_factory=list)
    zhu_dup_alpha: Optional[str] = None
    first_only: bool = False


class ResolveInput(BaseModel):
    source: str
    queue: List[Dict[str, Any]] = Field(default_factory=list)
    alpha: Optional[str] = None
    first_only: bool = False


def _algorithm_method_group(algorithm_cls: type) -> str:
    """Map package structure to a user-facing method family."""
    parts = algorithm_cls.__module__.split(".")
    if "solver_ip" in parts or (
        "exact" in parts and "solver" in parts
    ):
        return "Solver / IP"
    if "tree" in parts or (
        "exact" in parts and "search" in parts
    ):
        return "Tree Search"
    if "evolutionary" in parts:
        return "Evolutionary"
    if "heuristic" in parts:
        return "Heuristic"
    if "exact" in parts:
        return "Exact"
    return "Other"


def _catalog() -> Dict[str, Any]:
    problems = []
    for info in get_problem_info():
        problem_cls = get_problem_class(info["name"])
        problems.append({
            **json_safe(info),
            "config_schema": json_safe(
                problem_cls.config_schema() if problem_cls is not None else {}
            ),
        })

    algorithms = []
    for info in get_algorithm_info():
        if info["name"] == "BaseAlgorithm" and info["category"] == "Unknown":
            continue
        algorithm_cls = get_algorithm_class(info["name"])
        algorithms.append({
            **json_safe(info),
            "config_schema": json_safe(
                algorithm_cls.config_schema() if algorithm_cls is not None else {}
            ),
            "step_label": getattr(algorithm_cls, "step_label", "Step"),
            "method_group": (
                _algorithm_method_group(algorithm_cls)
                if algorithm_cls is not None
                else "Other"
            ),
        })
    return {"problems": problems, "algorithms": algorithms}


def _config_values(
    schema: Mapping[str, Mapping[str, Any]],
    supplied: Mapping[str, Any],
) -> Dict[str, Any]:
    unknown = sorted(set(supplied) - set(schema))
    if unknown:
        raise ValueError(f"Unknown configuration fields: {', '.join(unknown)}")

    values: Dict[str, Any] = {}
    for name, spec in schema.items():
        raw = supplied.get(name, spec.get("default"))
        kind = spec.get("type", "str")
        try:
            if kind == "int":
                value = int(raw)
            elif kind == "float":
                value = float(raw)
            elif kind == "bool":
                value = (
                    raw if isinstance(raw, bool)
                    else str(raw).lower() in {"1", "true", "yes", "on"}
                )
            else:
                value = "" if raw is None else str(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid value for {name}: {raw!r}") from exc

        if kind in {"int", "float"}:
            if "min" in spec and value < spec["min"]:
                raise ValueError(f"{name} must be at least {spec['min']}")
            if "max" in spec and value > spec["max"]:
                raise ValueError(f"{name} must be at most {spec['max']}")
        choices = spec.get("options", spec.get("choices"))
        if choices and value not in choices:
            raise ValueError(f"{name} must be one of {choices}")
        values[name] = value
    return values


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/catalog")
def catalog() -> Dict[str, Any]:
    return _catalog()


@app.get("/api/benchmarks")
def benchmarks(problem: str = Query(..., min_length=1)):
    return json_safe(available_for_problem(problem))


@app.post("/api/benchmarks/resolve")
def benchmarks_resolve(payload: ResolveInput):
    try:
        paths = resolve_paths(
            payload.source,
            payload.queue,
            alpha=payload.alpha,
            first_only=payload.first_only,
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc
    sample = [str(path) for path in paths[:8]]
    return {
        "count": len(paths),
        "sample_paths": sample,
        "first_only": payload.first_only,
    }


@app.get("/api/jobs")
def list_jobs():
    return manager.list()


@app.post("/api/jobs", status_code=201)
def start_job(payload: RunInput):
    problem_cls = get_problem_class(payload.problem_name)
    algorithm_cls = get_algorithm_class(payload.algorithm_name)
    if problem_cls is None:
        raise HTTPException(404, f"Unknown problem: {payload.problem_name}")
    if algorithm_cls is None:
        raise HTTPException(404, f"Unknown algorithm: {payload.algorithm_name}")
    try:
        validate_source_for_problem(payload.problem_name, payload.instance_source)
        problem_values = _config_values(
            problem_cls.config_schema(),
            payload.problem_config,
        )
        algorithm_values = _config_values(
            algorithm_cls.config_schema(),
            payload.algorithm_config,
        )
        layout_paths = []
        if payload.instance_source != SOURCE_RANDOM:
            layout_paths = [
                str(path)
                for path in resolve_paths(
                    payload.instance_source,
                    payload.benchmark_queue,
                    alpha=payload.zhu_dup_alpha,
                    first_only=payload.first_only,
                )
            ]
            if not layout_paths:
                raise ValueError(
                    "Instance list is empty — add height/size blocks before starting"
                )
        job = manager.start(
            payload.problem_name,
            payload.algorithm_name,
            problem_values,
            algorithm_values,
            instance_source=payload.instance_source,
            layout_paths=layout_paths,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return job.summary()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job.summary(include_records=True)


@app.get("/api/jobs/{job_id}/records")
def job_records(job_id: str, after: int = Query(0, ge=0)):
    payload = manager.records_after(job_id, after)
    if payload is None:
        raise HTTPException(404, "Job not found")
    return payload


@app.post("/api/jobs/{job_id}/stop")
def stop_job(job_id: str):
    job = manager.stop(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job.summary()


@app.get("/api/results")
def results(limit: int = Query(100, ge=1, le=1000)):
    return json_safe(list_saved_runs()[:limit])


@app.get("/api/results/detail")
def result_detail(path: str):
    candidate = Path(path).resolve()
    root = results_root().resolve()
    if root not in candidate.parents or candidate.suffix != ".json":
        raise HTTPException(400, "Result path is outside the results directory")
    if not candidate.exists():
        raise HTTPException(404, "Result not found")
    return json_safe(load_run(candidate))


@app.delete("/api/results")
def results_delete(path: str = Query(...)):
    """Delete one saved result JSON under results/."""
    try:
        delete_result_file(path, base_dir=results_root())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"ok": True, "path": path}


@app.delete("/api/results/folder")
def results_delete_folder(
    problem: str = Query(...),
    algorithm: str = Query(...),
):
    """Delete all run JSON files for one algorithm folder."""
    removed = delete_algorithm_folder(problem, algorithm, base_dir=results_root())
    return {"ok": True, "removed": removed, "problem": problem, "algorithm": algorithm}


class CompareInput(BaseModel):
    problem: str = "CRP-R"
    algorithms: List[str] = Field(default_factory=list)
    metric: str = "relocations"
    dedup: bool = True


@app.get("/api/compare/folders")
def compare_folders(problem: str = Query("CRP-R")):
    """List algorithm result folders available for comparison."""
    return json_safe(list_result_folders(problem, base_dir=results_root()))


@app.post("/api/compare")
def compare_post(payload: CompareInput):
    if len(payload.algorithms) < 1:
        raise HTTPException(400, "Select at least one algorithm folder")
    return json_safe(
        compare_algorithms(
            payload.problem,
            payload.algorithms,
            metric=payload.metric,
            limit=None,
            dedup=payload.dedup,
            base_dir=results_root(),
        )
    )


@app.delete("/api/jobs/{job_id}")
def jobs_delete(
    job_id: str,
    delete_files: bool = Query(
        False,
        description="Also delete result JSON files written by this job",
    ),
):
    summary = manager.remove(job_id, delete_files=delete_files)
    if summary is None:
        raise HTTPException(404, "Job not found")
    return json_safe(summary)


def _assert_under_results(path: Path) -> Path:
    candidate = path.resolve()
    root = results_root().resolve()
    if root not in candidate.parents and candidate != root:
        raise HTTPException(400, "Path is outside the results directory")
    return candidate


@app.get("/bench-summary", response_class=HTMLResponse, include_in_schema=False)
def bench_summary_index():
    """Plain HTML index — no React rebuild required."""
    jobs = manager.list()
    return HTMLResponse(
        render_index_html(jobs=jobs, saved=list_batch_summary_files())
    )


@app.get("/bench-summary/job/{job_id}", response_class=HTMLResponse, include_in_schema=False)
def bench_summary_job(job_id: str):
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found (in-memory only; try From results on disk)")
    summary = job.batch_summary
    if not summary and job.result_files:
        from core.benchmark_summary import summarize_result_files

        summary = summarize_result_files(job.result_files)
    if not summary:
        summary = {"classes": [], "ungrouped": 0, "total_runs": 0}
    return HTMLResponse(
        render_summary_html(
            summary,
            title=f"{job.algorithm_name} on {job.problem_name}",
            subtitle=f"job={job.id}  status={job.status}  source={job.instance_source}",
        )
    )


@app.get("/bench-summary/from-results", response_class=HTMLResponse, include_in_schema=False)
def bench_summary_from_results(
    problem: str = Query("CRP-R"),
    algorithm: str = Query("Caserta (2012) HEUR"),
    limit: int = Query(120, ge=1, le=5000),
):
    summary = summarize_saved_runs(problem=problem, algorithm=algorithm, limit=limit)
    return HTMLResponse(
        render_summary_html(
            summary,
            title=f"{algorithm} · {problem}",
            subtitle=f"Aggregated newest {limit} saved runs under results/",
        )
    )


@app.get("/bench-summary/file", response_class=HTMLResponse, include_in_schema=False)
def bench_summary_file(path: str = Query(...)):
    candidate = _assert_under_results(Path(path))
    if not candidate.is_file() or not candidate.name.startswith("batch_summary_"):
        raise HTTPException(404, "batch_summary file not found")
    summary = load_batch_summary_file(candidate)
    return HTMLResponse(
        render_summary_html(
            summary,
            title=candidate.name,
            subtitle=str(candidate),
        )
    )


@app.get("/api/bench-summary/from-results")
def api_bench_summary_from_results(
    problem: str = Query("CRP-R"),
    algorithm: str = Query("Caserta (2012) HEUR"),
    limit: int = Query(120, ge=1, le=5000),
):
    return json_safe(
        summarize_saved_runs(problem=problem, algorithm=algorithm, limit=limit)
    )


if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def frontend(full_path: str):
        candidate = (FRONTEND_DIST / full_path).resolve()
        if (
            full_path
            and FRONTEND_DIST.resolve() in candidate.parents
            and candidate.is_file()
        ):
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
else:
    @app.get("/", include_in_schema=False)
    def missing_frontend():
        return {
            "message": "Platform CRISP API is running.",
            "docs": "/docs",
            "bench_summary": "/bench-summary",
            "frontend": "Build web/frontend with npm run build.",
        }
