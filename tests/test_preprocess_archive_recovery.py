import os
import logging
import tempfile
import unittest
from unittest import mock

import h5py

from sotodlib.core.metadata.manifest import ManifestDb
from sotodlib.site_pipeline.jobdb import JobManager, JState
from sotodlib.site_pipeline.preprocess_tod import _recover_temp_files


class TestPreprocessArchiveRecovery(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = self.temp_dir.name
        self.index = os.path.join(self.root, "process_archive.sqlite")
        self.policy = os.path.join(self.root, "preprocess_archive.h5")
        self.dataset = "obs0_wafer_slot_ws0"
        self.temp_file = os.path.join(self.root, "temporary.h5")
        with h5py.File(self.temp_file, "w") as output:
            output.create_group(self.dataset)
        self.out_dict = {
            "temp_file": self.temp_file,
            "db_data": {
                "obs:obs_id": "obs0",
                "dets:wafer_slot": "ws0",
                "dataset": self.dataset,
            },
        }
        self.configs = {
            "archive": {
                "index": self.index,
                "policy": {"filename": self.policy},
                "batch_size": 10,
            }
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_recovery_commits_manifest_then_completes_job(self):
        jdb = JobManager(sqlite_file=":memory:")
        jdb.create_job(
            "init",
            {
                "obs:obs_id": "obs0",
                "dets:wafer_slot": "ws0",
                "error": None,
            },
        )
        recovered = [(self.out_dict, ("obs0", ["ws0"]))]

        with mock.patch(
            "sotodlib.site_pipeline.preprocess_tod.pp_util.get_temp_group_outputs",
            return_value=(recovered, (None, None, None)),
        ):
            _recover_temp_files(
                ["obs0"],
                self.configs,
                context=object(),
                temp_subdir="temp",
                group_by=["wafer_slot"],
                writer_lanes=1,
                writer_queue_depth=2,
                jdb=jdb,
                logger=logging.getLogger("archive-recovery-test"),
            )

        manifest = ManifestDb(self.index)
        entries = manifest.inspect({"obs:obs_id": "obs0"})
        self.assertEqual(len(entries), 1)
        self.assertIn("_lane00_000.h5", entries[0]["filename"])
        manifest.conn.close()
        self.assertFalse(os.path.exists(self.temp_file))

        jobs = jdb.get_jobs(jclass="init", jstate=JState.done)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].visit_count, 1)
        self.assertIsNone(jobs[0].tags["error"])


if __name__ == "__main__":
    unittest.main()
