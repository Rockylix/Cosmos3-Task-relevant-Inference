"""Synthetic SQLite tests; no CUDA, Torch or Nsight dependency."""

import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "audit_asi_compile_trace", Path(__file__).resolve().parents[1] / "tools/audit_asi_compile_trace.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)
PID = 7 << 24
TID = PID + 12


class CompileTraceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "trace.sqlite"
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript("""
            CREATE TABLE StringIds (id INTEGER, value TEXT);
            CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, textId INTEGER, globalTid INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER,end INTEGER,globalTid INTEGER,nameId INTEGER,correlationId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER (start INTEGER,end INTEGER,globalTid INTEGER,nameId INTEGER,correlationId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER,end INTEGER,globalPid INTEGER,correlationId INTEGER);
        """)
        self.conn.executemany(
            "INSERT INTO StringIds VALUES (?,?)",
            [
                (1, "asi.chunk.generate"),
                (2, "cudaGraphLaunch_v10000"),
                (3, "cuGraphLaunch"),
                (4, "asi.step0.conditional.B00"),
            ],
        )
        self.range(100, 1000, None, 1)
        self.range(120, 500, None, 4)
        self.range(510, 900, "asi.step0.unconditional.B00", None)
        self.api("RUNTIME", 200, 230, 2, 1)
        self.api("DRIVER", 205, 220, 3, 99)
        self.api("DRIVER", 550, 560, 3, 100)
        self.conn.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?)",
            [
                (240, 400, PID, 777),
                (300, 450, PID, 778),
                (580, 800, PID, 779),
            ],
        )

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def range(self, start, end, text, text_id):
        self.conn.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?)", (start, end, text, text_id, TID))

    def api(self, kind, start, end, name, corr):
        self.conn.execute(f"INSERT INTO CUPTI_ACTIVITY_KIND_{kind} VALUES (?,?,?,?,?)", (start, end, TID, name, corr))

    def parse(self):
        self.conn.commit()
        return AUDIT.parse(self.path)

    def test_double_count_and_root_only_kernel_stats(self):
        result = self.parse()
        self.assertEqual(result["actual_graph_launch_count"], 2)
        self.assertEqual(result["deduplicated_runtime_driver_pairs"], 1)
        self.assertEqual(result["raw_graph_launch_api_counts"], {"runtime": 1, "driver": 2})
        self.assertEqual(result["graph_launches_by_layer"]["asi.step0.conditional.B00"], 1)
        self.assertEqual(result["graph_launches_by_layer"]["asi.step0.unconditional.B00"], 1)
        self.assertEqual(result["kernel_count"], 3)
        self.assertAlmostEqual(result["kernel_sum_ms"], 530 / 1e6)
        self.assertAlmostEqual(result["kernel_busy_union_ms"], 430 / 1e6)
        self.assertFalse(result["all_224_layers_observed_with_graph"])

    def test_warmup_and_other_process_excluded(self):
        self.api("RUNTIME", 20, 30, 2, 5)
        self.range(10, 90, "asi.step0.conditional.B00", None)
        self.conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?)", (40, 70, PID, 5))
        self.conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?)", (250, 280, PID * 2, 6))
        result = self.parse()
        self.assertEqual(result["actual_graph_launch_count"], 2)
        self.assertEqual(result["kernel_count"], 3)

    def test_unique_closed_root_required(self):
        self.range(10, 90, "asi.chunk.generate", None)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.parse()

    def test_missing_root_rejected(self):
        self.conn.execute("DELETE FROM NVTX_EVENTS WHERE textId=1")
        with self.assertRaisesRegex(ValueError, "got 0"):
            self.parse()

    def test_crossing_kernel_rejected(self):
        self.conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?)", (999, 1001, PID, 888))
        with self.assertRaisesRegex(ValueError, "cross root"):
            self.parse()

    def test_repeated_correlation_does_not_collapse_separate_launches(self):
        self.api("RUNTIME", 600, 620, 2, 1)
        self.assertEqual(self.parse()["actual_graph_launch_count"], 3)

    def test_driver_table_optional(self):
        self.conn.execute("DROP TABLE CUPTI_ACTIVITY_KIND_DRIVER")
        self.assertEqual(self.parse()["actual_graph_launch_count"], 1)

    def test_all_224_layer_evidence(self):
        self.conn.execute("DELETE FROM NVTX_EVENTS WHERE textId IS NULL OR textId != 1")
        self.conn.execute("DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME")
        self.conn.execute("DELETE FROM CUPTI_ACTIVITY_KIND_DRIVER")
        index = 0
        for step in range(4):
            for branch in ("conditional", "unconditional"):
                for block in range(28):
                    start = 120 + index * 3
                    self.range(start, start + 3, f"asi.step{step}.{branch}.B{block:02d}", None)
                    self.api("DRIVER", start + 1, start + 2, 3, index)
                    index += 1
        result = self.parse()
        self.assertEqual(result["actual_graph_launch_count"], 224)
        self.assertTrue(result["all_224_layers_observed_with_graph"])
        self.assertTrue(result["first_conditional_28_layers_observed_with_graph"])
        self.assertEqual(result["graph_launches_by_step_branch"]["step0.conditional"], 28)

    def test_ambiguous_driver_nesting_fails_closed(self):
        self.api("DRIVER", 221, 225, 3, 102)
        with self.assertRaisesRegex(ValueError, "multiple Driver"):
            self.parse()


if __name__ == "__main__":
    unittest.main()
