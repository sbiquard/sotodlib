import os
import tempfile
import unittest

import h5py

from sotodlib.core.metadata.manifest import ManifestDb, ManifestScheme
from sotodlib.preprocess.preprocess_util import cleanup_archive


class TestPreprocessArchiveCleanup(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = self.temp_dir.name
        self.index = os.path.join(self.root, "process_archive.sqlite")
        self.policy = os.path.join(self.root, "preprocess_archive.h5")
        scheme = ManifestScheme()
        scheme.add_exact_match("obs:obs_id")
        scheme.add_data_field("dataset")
        self.db = ManifestDb(self.index, scheme=scheme)

    def tearDown(self):
        self.db.conn.close()
        self.temp_dir.cleanup()

    def make_archive(self, name, groups):
        path = os.path.join(self.root, name)
        with h5py.File(path, "w") as archive:
            for group in groups:
                archive.create_group(group)
        return path

    def test_cleanup_checks_latest_file_in_every_lane(self):
        old_lane_file = self.make_archive(
            "preprocess_archive_lane00_000.h5", ["old_orphan"]
        )
        lane0 = self.make_archive(
            "preprocess_archive_lane00_001.h5", ["keep0", ".__copying__.x"]
        )
        lane1 = self.make_archive(
            "preprocess_archive_lane01_000.h5", ["keep1", "orphan1"]
        )
        self.db.add_entry(
            {"obs:obs_id": "obs0", "dataset": "keep0"},
            os.path.basename(lane0),
        )
        self.db.add_entry(
            {"obs:obs_id": "obs1", "dataset": "keep1"},
            os.path.basename(lane1),
        )
        configs = {
            "archive": {
                "index": self.index,
                "policy": {"filename": self.policy},
            }
        }

        cleanup_archive(configs)

        with h5py.File(old_lane_file, "r") as archive:
            self.assertIn("old_orphan", archive)
        with h5py.File(lane0, "r") as archive:
            self.assertEqual(list(archive), ["keep0"])
        with h5py.File(lane1, "r") as archive:
            self.assertEqual(list(archive), ["keep1"])


if __name__ == "__main__":
    unittest.main()
