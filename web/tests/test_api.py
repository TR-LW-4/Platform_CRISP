from __future__ import annotations

import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from web.backend.api import app
from web.backend.benchmarks import available_for_problem, caserta_root, resolve_paths


class WebApiTests(unittest.TestCase):
    def test_catalog_and_isolated_job(self) -> None:
        with TestClient(app) as client:
            health = client.get("/api/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["status"], "ok")

            frontend = client.get("/")
            self.assertEqual(frontend.status_code, 200)
            self.assertIn("text/html", frontend.headers["content-type"])

            catalog_response = client.get("/api/catalog")
            self.assertEqual(catalog_response.status_code, 200)
            catalog = catalog_response.json()
            self.assertGreater(len(catalog["problems"]), 0)
            self.assertGreater(len(catalog["algorithms"]), 0)
            self.assertNotIn(
                "BaseAlgorithm",
                {item["name"] for item in catalog["algorithms"]},
            )
            crp_r_groups = {
                item["method_group"]
                for item in catalog["algorithms"]
                if (
                    not item["compatible_problems"]
                    or "CRP-R" in item["compatible_problems"]
                )
            }
            self.assertTrue(
                {"Heuristic", "Tree Search", "Solver / IP"}.issubset(crp_r_groups)
            )
            self.assertTrue(
                {item["method_group"] for item in catalog["algorithms"]}
                <= {
                    "Heuristic",
                    "Tree Search",
                    "Solver / IP",
                    "Evolutionary",
                    "Exact",
                    "Other",
                }
            )

            started = client.post(
                "/api/jobs",
                json={
                    "problem_name": "CRP-R",
                    "algorithm_name": "Kim–Hong (2006) ENAR",
                    "problem_config": {
                        "num_bays": 3,
                        "num_rows": 2,
                        "max_tiers": 3,
                        "num_containers": 8,
                        "num_groups": 8,
                        "seed": 0,
                    },
                    "algorithm_config": {},
                },
            )
            self.assertEqual(started.status_code, 201, started.text)
            job_id = started.json()["id"]

            deadline = time.monotonic() + 30
            job = started.json()
            while job["status"] not in {"completed", "failed", "stopped"}:
                self.assertLess(time.monotonic(), deadline, "Web job timed out")
                time.sleep(0.1)
                job = client.get(f"/api/jobs/{job_id}").json()

            self.assertEqual(job["status"], "completed", job.get("error"))
            self.assertGreater(job["record_count"], 0)
            self.assertTrue(job["records"])

            if job.get("result_file"):
                Path(job["result_file"]).unlink(missing_ok=True)

    def test_benchmark_index_and_first_only_job(self) -> None:
        with TestClient(app) as client:
            payload = client.get("/api/benchmarks", params={"problem": "CRP-R"})
            self.assertEqual(payload.status_code, 200)
            sources = {item["id"]: item for item in payload.json()["sources"]}
            self.assertIn("random", sources)
            self.assertIn("caserta", sources)
            self.assertIn("zhu", sources)
            self.assertTrue(sources["caserta"]["available"])

            heights = sources["caserta"]["heights"]
            self.assertTrue(heights)
            h = heights[0]
            ws = sources["caserta"]["ws_by_height"][str(h)]
            self.assertTrue(ws)
            queue = [{"h": h, "ws": [ws[0]]}]

            resolved = client.post(
                "/api/benchmarks/resolve",
                json={
                    "source": "caserta",
                    "queue": queue,
                    "first_only": True,
                },
            )
            self.assertEqual(resolved.status_code, 200, resolved.text)
            self.assertEqual(resolved.json()["count"], 1)

            paths = resolve_paths("caserta", queue, first_only=True)
            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].is_file())
            self.assertTrue(caserta_root().is_dir())

            started = client.post(
                "/api/jobs",
                json={
                    "problem_name": "CRP-R",
                    "algorithm_name": "Kim–Hong (2006) ENAR",
                    "problem_config": {},
                    "algorithm_config": {},
                    "instance_source": "caserta",
                    "benchmark_queue": queue,
                    "first_only": True,
                },
            )
            self.assertEqual(started.status_code, 201, started.text)
            job_id = started.json()["id"]
            self.assertEqual(started.json()["instance_source"], "caserta")
            self.assertEqual(started.json()["batch_total"], 1)

            deadline = time.monotonic() + 60
            job = started.json()
            while job["status"] not in {"completed", "failed", "stopped"}:
                self.assertLess(time.monotonic(), deadline, "Benchmark job timed out")
                time.sleep(0.2)
                job = client.get(f"/api/jobs/{job_id}").json()

            self.assertEqual(job["status"], "completed", job.get("error"))
            self.assertGreater(job["record_count"], 0)
            self.assertTrue(job.get("result_file") or job.get("result_files"))
            for path in job.get("result_files") or []:
                Path(path).unlink(missing_ok=True)
            if job.get("result_file"):
                Path(job["result_file"]).unlink(missing_ok=True)

            dup = available_for_problem("CRP-D")
            dup_ids = {item["id"] for item in dup["sources"]}
            self.assertIn("zhu_dup", dup_ids)
            self.assertNotIn("caserta", dup_ids)


if __name__ == "__main__":
    unittest.main()
