"""Long-running algorithm jobs isolated from the FastAPI process."""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from core.base_algorithm import ProgressRecord as AlgoProgress
from core.benchmark_keys import LAYOUT_FILE_EXTRA_KEY
from core.registry import (
    compatible_algorithms,
    get_algorithm_class,
    get_algorithm_info,
    get_problem_class,
)
from core.benchmark_summary import summarize_result_files
from core.result_store import (
    completed_layout_paths,
    moves_from_records,
    result_path_for,
    save_run,
)
from web.backend.summary_pages import save_batch_summary_sidecar
from web.backend.benchmarks import (
    SOURCE_RANDOM,
    problem_config_for_layout,
    source_tag,
    validate_source_for_problem,
)
from web.backend.config import algorithm_config, problem_config
from web.backend.serialization import json_safe, progress_to_json


FINAL_STATUSES = frozenset({"completed", "stopped", "failed"})


@dataclass(frozen=True)
class WorkerFinished:
    result_files: tuple = ()


@dataclass(frozen=True)
class WorkerFailure:
    message: str
    traceback_text: str


class _BatchProgressQueue:
    """Remap per-file progress onto a batch progress bar and tee for save_run."""

    def __init__(
        self,
        outer,
        *,
        batch_index: int,
        batch_total: int,
        layout_file: str,
        buffer: List[Any],
    ) -> None:
        self._outer = outer
        self._batch_index = batch_index
        self._batch_total = batch_total
        self._layout_file = layout_file
        self._buffer = buffer

    def put(self, item, block: bool = True, timeout=None) -> None:
        remapped = self._remap(item)
        self._buffer.append(item)
        self._outer.put(remapped, block=block, timeout=timeout)

    def put_nowait(self, item) -> None:
        remapped = self._remap(item)
        self._buffer.append(item)
        try:
            self._outer.put_nowait(remapped)
        except Exception:
            pass

    def _remap(self, item: Any) -> Any:
        if isinstance(item, AlgoProgress):
            local = float(item.progress)
            progress = (self._batch_index + max(0.0, min(1.0, local))) / max(
                1, self._batch_total
            )
            extra = dict(item.extra or {})
            extra.update({
                "batch_index": self._batch_index,
                "batch_total": self._batch_total,
                "layout_file": self._layout_file,
            })
            return AlgoProgress(
                step=item.step,
                metric=item.metric,
                metrics=item.metrics,
                best_metric=item.best_metric,
                progress=min(progress, 1.0),
                yard_snapshot=item.yard_snapshot,
                extra=extra,
            )
        if isinstance(item, dict):
            remapped = dict(item)
            local = float(remapped.get("progress", 0.0))
            remapped["progress"] = (
                self._batch_index + max(0.0, min(1.0, local))
            ) / max(1, self._batch_total)
            extra = dict(remapped.get("extra") or {})
            extra.update({
                "batch_index": self._batch_index,
                "batch_total": self._batch_total,
                "layout_file": self._layout_file,
            })
            remapped["extra"] = extra
            return remapped
        return item


def _save_layout_run(
    *,
    problem_name: str,
    algorithm_name: str,
    category: str,
    layout_path: Path,
    source: str,
    problem_values: Dict[str, Any],
    algorithm_values: Dict[str, Any],
    records: Sequence[Any],
) -> Optional[str]:
    if not records:
        return None
    final = records[-1]
    if isinstance(final, dict):
        metrics = final.get("metrics", {})
        history = [
            {
                "step": r.get("step", 0),
                "metric": r.get("metric", 0.0),
                "metrics": r.get("metrics", {}),
            }
            for r in records
            if isinstance(r, dict)
        ]
    else:
        metrics = getattr(final, "metrics", {})
        history = [
            {
                "step": getattr(r, "step", 0),
                "metric": getattr(r, "metric", 0.0),
                "metrics": getattr(r, "metrics", {}),
            }
            for r in records
        ]
    prob_save = {
        **problem_values,
        LAYOUT_FILE_EXTRA_KEY: str(layout_path),
        "source": source_tag(source),
    }
    path = save_run(
        problem=problem_name,
        algorithm=algorithm_name,
        category=category,
        seed=abs(hash(layout_path.name)) % 2_000_000_000,
        prob_config=prob_save,
        algo_config=algorithm_values,
        metrics=metrics,
        history=history,
        moves=moves_from_records(list(records)),
    )
    return str(path)


def _merge_unique(existing: Sequence[str], incoming: Sequence[str]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for item in list(existing) + list(incoming):
        if not item or item in seen:
            continue
        seen.add(item)
        merged.append(item)
    return merged


def _worker_main(
    problem_name: str,
    algorithm_name: str,
    problem_values: Dict[str, Any],
    algorithm_values: Dict[str, Any],
    result_queue,
    stop_event,
    instance_source: str = SOURCE_RANDOM,
    layout_paths: Optional[List[str]] = None,
    category: str = "Unknown",
    skip_paths: Optional[List[str]] = None,
) -> None:
    """Resolve classes inside the child and never import Web framework objects."""
    try:
        problem_cls = get_problem_class(problem_name)
        algorithm_cls = get_algorithm_class(algorithm_name)
        if problem_cls is None:
            raise LookupError(f"Unknown problem: {problem_name}")
        if algorithm_cls is None:
            raise LookupError(f"Unknown algorithm: {algorithm_name}")

        algorithm_cfg = algorithm_config(algorithm_values)
        paths = [Path(p) for p in (layout_paths or [])]
        skip = {str(Path(p)) for p in (skip_paths or [])}

        if not paths:
            problem_cfg = problem_config(problem_values)
            algorithm = algorithm_cls(config=algorithm_cfg)

            def factory():
                return problem_cls(config=problem_cfg)

            algorithm.train(factory, result_queue, stop_event)
            result_queue.put(WorkerFinished(), timeout=2)
            return

        result_files: List[str] = []
        batch_total = len(paths)
        for index, layout_path in enumerate(paths):
            if stop_event.is_set():
                break
            if str(layout_path) in skip:
                continue
            problem_cfg = problem_config_for_layout(
                instance_source,
                layout_path,
                problem_values,
            )
            algorithm = algorithm_cls(config=algorithm_cfg)
            buffer: List[Any] = []
            proxy = _BatchProgressQueue(
                result_queue,
                batch_index=index,
                batch_total=batch_total,
                layout_file=str(layout_path),
                buffer=buffer,
            )

            def factory(_cfg=problem_cfg):
                return problem_cls(config=_cfg)

            algorithm.train(factory, proxy, stop_event)
            # Interrupted instance is unfinished: drop the buffer and continue
            # later from this same layout file.
            if stop_event.is_set():
                break
            saved = _save_layout_run(
                problem_name=problem_name,
                algorithm_name=algorithm_name,
                category=category,
                layout_path=layout_path,
                source=instance_source,
                problem_values=problem_values,
                algorithm_values=algorithm_values,
                records=buffer,
            )
            if saved:
                result_files.append(saved)

        result_queue.put(WorkerFinished(result_files=tuple(result_files)), timeout=2)
    except BaseException as exc:
        try:
            result_queue.put(
                WorkerFailure(
                    message=f"{type(exc).__name__}: {exc}",
                    traceback_text=traceback.format_exc(),
                ),
                timeout=2,
            )
        except Exception:
            pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    problem_name: str
    algorithm_name: str
    category: str
    problem_config: Dict[str, Any]
    algorithm_config: Dict[str, Any]
    instance_source: str = SOURCE_RANDOM
    layout_paths: List[str] = field(default_factory=list)
    status: str = "starting"
    created_at: str = field(default_factory=_utc_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    records: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    result_file: Optional[str] = None
    result_files: List[str] = field(default_factory=list)
    batch_summary: Optional[Dict[str, Any]] = None
    stop_requested: bool = False
    process: Optional[mp.Process] = field(default=None, repr=False)
    result_queue: Any = field(default=None, repr=False)
    stop_event: Any = field(default=None, repr=False)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def batch_counts(self) -> Dict[str, Any]:
        """batch_index / completed_count / pending_count for API payloads."""
        batch_total = len(self.layout_paths)
        latest = self.records[-1] if self.records else None
        batch_index = None
        if latest and isinstance(latest.get("extra"), dict):
            batch_index = latest["extra"].get("batch_index")
        completed_count = None
        pending_count = None
        if batch_total:
            if self.status in FINAL_STATUSES:
                completed_count = len(completed_layout_paths(
                    self.problem_name,
                    self.algorithm_name,
                    self.layout_paths,
                    prob_config=self.problem_config,
                ))
            else:
                completed_count = len(self.result_files)
            pending_count = max(0, batch_total - completed_count)
        return {
            "batch_total": batch_total or None,
            "batch_index": batch_index,
            "completed_count": completed_count,
            "pending_count": pending_count,
        }

    def summary(self, include_records: bool = False) -> Dict[str, Any]:
        with self.lock:
            latest = self.records[-1] if self.records else None
            payload = {
                "id": self.id,
                "problem_name": self.problem_name,
                "algorithm_name": self.algorithm_name,
                "category": self.category,
                "problem_config": json_safe(self.problem_config),
                "algorithm_config": json_safe(self.algorithm_config),
                "instance_source": self.instance_source,
                **self.batch_counts(),
                "status": self.status,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "record_count": len(self.records),
                "latest": latest,
                "error": self.error,
                "result_file": self.result_file,
                "result_files": list(self.result_files),
                "batch_summary": json_safe(self.batch_summary),
            }
            if include_records:
                payload["records"] = list(self.records)
            return payload


class JobManager:
    """Thread-safe in-memory job registry with one monitor per process."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.RLock()
        self._algorithm_categories = {
            item["name"]: item.get("category", "Unknown")
            for item in get_algorithm_info()
        }

    def start(
        self,
        problem_name: str,
        algorithm_name: str,
        problem_values: Dict[str, Any],
        algorithm_values: Dict[str, Any],
        *,
        instance_source: str = SOURCE_RANDOM,
        layout_paths: Optional[Sequence[str]] = None,
    ) -> Job:
        if get_problem_class(problem_name) is None:
            raise ValueError(f"Unknown problem: {problem_name}")
        if get_algorithm_class(algorithm_name) is None:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        if algorithm_name not in compatible_algorithms(problem_name):
            raise ValueError(
                f"{algorithm_name} is not compatible with {problem_name}"
            )
        validate_source_for_problem(problem_name, instance_source)
        paths = [str(Path(p)) for p in (layout_paths or [])]
        if instance_source != SOURCE_RANDOM and not paths:
            raise ValueError("Benchmark run requires at least one layout file")
        if instance_source == SOURCE_RANDOM:
            paths = []

        category = self._algorithm_categories.get(algorithm_name, "Unknown")
        job = Job(
            id=uuid.uuid4().hex,
            problem_name=problem_name,
            algorithm_name=algorithm_name,
            category=category,
            problem_config=dict(problem_values),
            algorithm_config=dict(algorithm_values),
            instance_source=instance_source,
            layout_paths=paths,
        )
        with self._lock:
            self._jobs[job.id] = job
        self._spawn(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        return sorted(
            (job.summary() for job in jobs),
            key=lambda item: item["created_at"],
            reverse=True,
        )

    def records_after(self, job_id: str, after: int) -> Optional[Dict[str, Any]]:
        job = self.get(job_id)
        if job is None:
            return None
        with job.lock:
            records = [
                record for record in job.records
                if record["sequence"] > after
            ]
            return {
                "job_id": job.id,
                "status": job.status,
                "records": records,
                "next_sequence": (
                    job.records[-1]["sequence"] if job.records else after
                ),
                "error": job.error,
                "result_file": job.result_file,
                "result_files": list(job.result_files),
                "batch_summary": json_safe(job.batch_summary),
                **job.batch_counts(),
                "instance_source": job.instance_source,
            }

    def stop(self, job_id: str) -> Optional[Job]:
        job = self.get(job_id)
        if job is None:
            return None
        with job.lock:
            if job.status in FINAL_STATUSES:
                return job
            job.stop_requested = True
            job.status = "stopping"
            job.stop_event.set()
        return job

    def continue_job(self, job_id: str) -> Job:
        """Resume a stopped/failed batch from the first layout without a result file."""
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        with job.lock:
            if job.status not in FINAL_STATUSES:
                raise ValueError("Job is still running")
            if not job.layout_paths:
                raise ValueError("Only batch runs can be continued")
            if job.process is not None and job.process.is_alive():
                raise ValueError("Worker is still shutting down. Try again in a moment.")
            done = completed_layout_paths(
                job.problem_name,
                job.algorithm_name,
                job.layout_paths,
                prob_config=job.problem_config,
            )
            if len(done) >= len(job.layout_paths):
                raise ValueError("All instances already have saved results")
            self._sync_result_files_from_disk(job)
            job.stop_requested = False
            job.error = None
            job.finished_at = None
            job.status = "running"
            skip_paths = list(done)
        self._spawn(job, skip_paths=skip_paths)
        return job

    def _spawn(
        self,
        job: Job,
        *,
        skip_paths: Optional[Sequence[str]] = None,
    ) -> None:
        # FastAPI already owns worker threads; spawn avoids unsafe fork-after-thread.
        context = mp.get_context("spawn")
        job.result_queue = context.Queue(maxsize=512)
        job.stop_event = context.Event()
        job.process = context.Process(
            target=_worker_main,
            args=(
                job.problem_name,
                job.algorithm_name,
                job.problem_config,
                job.algorithm_config,
                job.result_queue,
                job.stop_event,
                job.instance_source,
                list(job.layout_paths),
                job.category,
                [str(p) for p in (skip_paths or [])],
            ),
            daemon=True,
            name=f"crisp-web-{job.id[:8]}",
        )
        job.process.start()
        job.status = "running"
        if job.started_at is None:
            job.started_at = _utc_now()
        threading.Thread(
            target=self._monitor,
            args=(job,),
            daemon=True,
            name=f"monitor-{job.id[:8]}",
        ).start()

    @staticmethod
    def _sync_result_files_from_disk(job: Job) -> None:
        existing: List[str] = []
        seen = set()
        for layout in job.layout_paths:
            path = result_path_for(
                job.problem_name,
                job.algorithm_name,
                layout,
                prob_config=job.problem_config,
            )
            if not path.exists():
                continue
            text = str(path)
            if text in seen:
                continue
            seen.add(text)
            existing.append(text)
        job.result_files = _merge_unique(existing, job.result_files)
        if job.result_files:
            job.result_file = job.result_files[-1]

    def remove(self, job_id: str, *, delete_files: bool = False) -> Optional[Dict[str, Any]]:
        """
        Drop an in-memory job. Stops a running worker first.
        Optionally deletes saved result JSON files written by the job.
        """
        job = self.get(job_id)
        if job is None:
            return None
        with job.lock:
            process = job.process
            if process is not None and process.is_alive():
                job.stop_requested = True
                job.stop_event.set()
                process.terminate()
                process.join(timeout=2)
            summary = job.summary()
            result_files = list(job.result_files)
        if delete_files:
            for path in result_files:
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    pass
        with self._lock:
            self._jobs.pop(job_id, None)
        summary["status"] = "removed"
        return summary

    def shutdown(self) -> None:
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            with job.lock:
                process = job.process
                if process is None or not process.is_alive():
                    continue
                job.stop_event.set()
                process.terminate()
                process.join(timeout=2)

    def _monitor(self, job: Job) -> None:
        failure: Optional[WorkerFailure] = None
        saw_finished = False
        stop_deadline: Optional[float] = None
        empty_after_exit = 0
        finished_files: List[str] = []

        while True:
            drained = 0
            while True:
                try:
                    message = job.result_queue.get_nowait()
                except queue.Empty:
                    break
                except (EOFError, OSError):
                    break
                drained += 1
                if isinstance(message, WorkerFailure):
                    failure = message
                elif isinstance(message, WorkerFinished):
                    saw_finished = True
                    finished_files = list(message.result_files)
                else:
                    with job.lock:
                        sequence = len(job.records) + 1
                        job.records.append(progress_to_json(message, sequence))

            process_alive = bool(job.process and job.process.is_alive())
            if job.stop_requested and process_alive:
                if stop_deadline is None:
                    stop_deadline = time.monotonic() + 5.0
                elif time.monotonic() >= stop_deadline:
                    job.process.terminate()
                    stop_deadline = None

            if process_alive:
                time.sleep(0.1)
                continue

            empty_after_exit = 0 if drained else empty_after_exit + 1
            if empty_after_exit < 3 and not saw_finished and failure is None:
                time.sleep(0.1)
                continue
            break

        if job.process is not None:
            job.process.join(timeout=0)
        exit_code = job.process.exitcode if job.process is not None else None

        with job.lock:
            if finished_files:
                job.result_files = _merge_unique(job.result_files, finished_files)
                job.result_file = job.result_files[-1]
            summary_files = list(job.result_files)
            if len(summary_files) > 1:
                try:
                    job.batch_summary = summarize_result_files(summary_files)
                    save_batch_summary_sidecar(
                        summary_files,
                        job.batch_summary,
                        problem=job.problem_name,
                        algorithm=job.algorithm_name,
                    )
                except Exception as exc:
                    job.batch_summary = {
                        "classes": [],
                        "ungrouped": 0,
                        "total_runs": len(summary_files),
                        "error": f"summary failed: {exc}",
                    }
            if job.stop_requested:
                job.status = "stopped"
            elif failure is not None:
                job.status = "failed"
                job.error = f"{failure.message}\n\n{failure.traceback_text}"
            elif exit_code not in (0, None) and not saw_finished:
                job.status = "failed"
                job.error = f"Worker exited unexpectedly with code {exit_code}."
            else:
                job.status = "completed"
                if job.instance_source == SOURCE_RANDOM:
                    self._save_completed(job)
            job.finished_at = _utc_now()

        try:
            job.result_queue.close()
        except Exception:
            pass

    @staticmethod
    def _save_completed(job: Job) -> None:
        if not job.records:
            return
        final = job.records[-1]
        history = [
            {
                "step": record["step"],
                "metric": record["metric"],
                "metrics": record["metrics"],
            }
            for record in job.records
        ]
        try:
            path = save_run(
                problem=job.problem_name,
                algorithm=job.algorithm_name,
                category=job.category,
                seed=int(job.problem_config.get("seed", 0)),
                prob_config=job.problem_config,
                algo_config=job.algorithm_config,
                metrics=final["metrics"],
                history=history,
                moves=moves_from_records(job.records),
            )
            job.result_file = str(path)
            job.result_files = [str(path)]
        except Exception as exc:
            job.error = f"Run completed, but result saving failed: {exc}"
