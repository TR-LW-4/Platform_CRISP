import unittest

from core.benchmarks.stow import (
    DEFAULT_STOW_DIR,
    collect_paths_from_stow_queue,
    index_classes_by_min_vessel_height,
    parse_stow_filename,
    problem_config_for_stow_pro,
)
from web.backend.benchmarks import (
    SOURCE_STOW,
    available_for_problem,
    resolve_paths,
    validate_source_for_problem,
)


class StowBenchmarkTests(unittest.TestCase):
    def test_filename_parser(self) -> None:
        self.assertEqual(
            parse_stow_filename("Bay-3-20-38-6_23.pro"),
            (3, 20, 38, 6, 23),
        )

    def test_official_dataset_index(self) -> None:
        indexed = index_classes_by_min_vessel_height(DEFAULT_STOW_DIR)
        self.assertEqual(sorted(indexed), [3, 5, 10, 15])
        self.assertEqual(sum(len(classes) for classes in indexed.values()), 39)
        self.assertIn((20, 38, 6), indexed[3])

    def test_queue_resolves_all_40_seed_instances(self) -> None:
        queue = [{
            "h": 3,
            "stow_classes": [{"vs": 20, "ys": 38, "yt": 6}],
        }]
        paths = collect_paths_from_stow_queue(DEFAULT_STOW_DIR, queue)
        self.assertEqual(len(paths), 40)
        self.assertEqual(parse_stow_filename(paths[0])[4], 0)
        self.assertEqual(parse_stow_filename(paths[-1])[4], 39)
        self.assertEqual(resolve_paths(SOURCE_STOW, queue), paths)

    def test_problem_config_comes_from_pro_file(self) -> None:
        path = DEFAULT_STOW_DIR / "Bay-3-3-3-6_0.pro"
        config = problem_config_for_stow_pro(path, {"rc_ratio": 0.2})
        self.assertEqual(config.num_containers, 10)
        self.assertEqual(config.num_bays, 3)
        self.assertEqual(config.num_rows, 1)
        self.assertEqual(config.max_tiers, 6)
        self.assertEqual(config.num_groups, 3)
        self.assertEqual(config.seed, 0)
        self.assertEqual(config.rc_ratio, 0.2)
        self.assertEqual(config.extra["layout_file_path"], str(path.resolve()))

    def test_web_catalog_exposes_stow_source(self) -> None:
        payload = available_for_problem("CRP-Stow")
        source_ids = [
            source["id"] for source in payload["sources"] if source["available"]
        ]
        self.assertIn("random", source_ids)
        self.assertIn(SOURCE_STOW, source_ids)
        validate_source_for_problem("CRP-Stow", SOURCE_STOW)
        with self.assertRaises(ValueError):
            validate_source_for_problem("CRP-R", SOURCE_STOW)


if __name__ == "__main__":
    unittest.main()
