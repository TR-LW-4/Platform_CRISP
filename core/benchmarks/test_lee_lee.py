"""
Checks for the Lee & Lee (2010) instance reader (core/benchmarks/lee_lee.py):
the format on small written files, and, if benchmark/lee_lee is present, the
71 published files against their names, Example 1 (Fig. 6, p. 1144), the
"upside-down" property of set U (Table 3, p. 1146) and Lee & Lee's lower bound
L = C + badly placed containers (p. 1140; Tables 1 and 3).

  python -m unittest core.benchmarks.test_lee_lee   (from the repo root)
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.benchmarks.lee_lee import DEFAULT_LEE_LEE_DIR, list_lee_lee_files, parse_lee_lee, parse_lee_lee_name

FIG6 = [[10, 4, 18], [9, 17, 8, 20], [5, 19, 2], [16, 14, 7, 13, 3, 6], [1, 11], [15, 12]]


def n_bad(stack):
    """Containers with a smaller group somewhere below them."""
    return sum(1 for i, g in enumerate(stack) if i and min(stack[:i]) < g)


class TestFormat(unittest.TestCase):
    def write(self, text):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        p = Path(d.name) / "R010203_0004_001.txt"
        p.write_text(text)
        return p

    def test_name(self):
        self.assertEqual(parse_lee_lee_name("R011606_0070_001.txt"), ("R", 1, 16, 6, 70, 1))
        self.assertEqual(parse_lee_lee_name("U101606_0720_002"), ("U", 10, 16, 6, 720, 2))
        self.assertIsNone(parse_lee_lee_name("data3-3-1.dat"))

    def test_parse(self):
        ins = parse_lee_lee(self.write("R010203_0004_001 2 2 3 4 4\n"
                                       "1 1 2 3 3 1 1\n1 2 0\n2 1 1 4 4\n2 2 1 2 2\n"))
        self.assertEqual((ins.n_bays, ins.n_stacks, ins.max_height, ins.n_containers, ins.n_groups), (2, 2, 3, 4, 4))
        self.assertEqual(ins.groups, (((3, 1), ()), ((4,), (2,))))
        self.assertEqual(ins.ids, ins.groups)

    def test_groups_and_ids_are_read_separately(self):
        ins = parse_lee_lee(self.write("R010203_0004_001 1 2 3 2 1\n1 1 1 7 5\n1 2 1 8 5\n"))
        self.assertEqual(ins.ids, (((7,), (8,)),))
        self.assertEqual(ins.groups, (((5,), (5,)),))

    def test_rejects_inconsistent_files(self):
        bad = ["R 1 2 3 2 2\n1 1 1 1 1\n",                    # one stack line for 1 x 2
               "R 1 2 3 2 2\n1 1 1 1 1\n1 3 1 2 2\n",         # stack index out of order
               "R 1 2 3 2 2\n1 1 2 1 1\n1 2 0\n",             # height 2, one pair
               "R 1 2 1 3 3\n1 1 2 1 1 2 2\n1 2 1 3 3\n",     # higher than H
               "R 1 2 3 3 3\n1 1 1 1 1\n1 2 1 2 2\n"]         # 2 containers, header 3
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_lee_lee(self.write(text))


@unittest.skipUnless(list_lee_lee_files(), f"Lee & Lee instances not found in {DEFAULT_LEE_LEE_DIR}")
class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.insts = {p.stem: parse_lee_lee(p) for p in list_lee_lee_files()}

    def test_inventory(self):
        names = [parse_lee_lee_name(n) for n in self.insts]
        self.assertEqual(len(self.insts), 71)
        self.assertEqual(sum(n.set == "R" for n in names), 51)       # 50 of Table 1 and Example 1
        self.assertEqual(sum(n.set == "U" for n in names), 20)

    def test_headers_match_names(self):
        for name, ins in self.insts.items():
            nm = parse_lee_lee_name(name)
            with self.subTest(name=name):
                self.assertEqual(ins.name, name)
                self.assertEqual((ins.n_bays, ins.n_stacks, ins.max_height, ins.n_containers),
                                 (nm.n_bays, nm.n_stacks, nm.max_height, nm.n_containers))
                ids = sorted(c for bay in ins.ids for s in bay for c in s)
                self.assertEqual(ids, list(range(1, ins.n_containers + 1)))

    def test_example_1(self):
        ins = self.insts["R020306_0020_001"]
        self.assertEqual([list(s) for bay in ins.groups for s in bay], FIG6)

    def test_upside_down(self):
        for name, ins in self.insts.items():
            if name.startswith("U"):
                for s in (s for bay in ins.groups for s in bay):
                    self.assertEqual(n_bad(s), max(len(s) - 1, 0), name)

    def test_lower_bound_of_lee_lee(self):
        L = {"R020306_0020_001": 27, "R011606_0070_001": 100, "R101606_0720_005": 1085,
             "R061608_0570_005": 914, "U011606_0070_001": 125, "U061608_0570_002": 1044}
        for name, want in L.items():
            ins = self.insts[name]
            self.assertEqual(ins.n_containers + sum(n_bad(s) for bay in ins.groups for s in bay), want, name)


if __name__ == "__main__":
    unittest.main()
