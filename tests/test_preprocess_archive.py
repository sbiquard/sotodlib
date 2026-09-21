import os
import tempfile
import unittest

import h5py

from sotodlib.core.metadata.manifest import DbBatchManager
from sotodlib.preprocess import preprocess_util as pp_util


class TestPreprocessArchivePublication(unittest.TestCase):
    def test_temp_files_survive_until_manifest_batch_commits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            configs = {
                "archive": {
                    "index": os.path.join(temp_dir, "manifest.sqlite"),
                    "policy": {
                        "type": "simple",
                        "filename": os.path.join(temp_dir, "archive.h5"),
                    },
                },
            }
            db = pp_util.get_preprocess_db(configs, ["detset"])
            manager = DbBatchManager(db, batch_size=2)

            sources = []
            with manager:
                for index in range(2):
                    source = os.path.join(temp_dir, f"temp-{index}.h5")
                    dataset = f"obs-{index}_dets_detset_ds"
                    with h5py.File(source, "w") as handle:
                        handle.create_dataset(dataset, data=[index])
                    sources.append(source)
                    pp_util.cleanup_mandb(
                        {
                            "temp_file": source,
                            "db_data": {
                                "obs:obs_id": f"obs-{index}",
                                "dets:detset": "ds",
                                "dataset": dataset,
                            },
                        },
                        (f"obs-{index}", ["ds"]),
                        (None, None, None),
                        configs,
                        db_manager=manager,
                    )
                    if index == 0:
                        self.assertTrue(os.path.exists(source))

            self.assertTrue(all(not os.path.exists(path) for path in sources))
            self.assertEqual(len(db.inspect({})), 2)
            with h5py.File(
                os.path.join(temp_dir, "archive_000.h5"), "r"
            ) as handle:
                self.assertEqual(len(handle.keys()), 2)
            db.conn.close()

    def test_existing_archive_group_repairs_missing_manifest_entry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            configs = {
                "archive": {
                    "index": os.path.join(temp_dir, "manifest.sqlite"),
                    "policy": {
                        "type": "simple",
                        "filename": os.path.join(temp_dir, "archive.h5"),
                    },
                },
            }
            source = os.path.join(temp_dir, "temp.h5")
            dataset = "obs_dets_detset_ds"
            with h5py.File(source, "w") as handle:
                handle.create_dataset(dataset, data=[1])
            destination = os.path.join(temp_dir, "archive_000.h5")
            with h5py.File(destination, "w") as handle:
                handle.create_dataset(dataset, data=[1])

            pp_util.cleanup_mandb(
                {
                    "temp_file": source,
                    "db_data": {
                        "obs:obs_id": "obs",
                        "dets:detset": "ds",
                        "dataset": dataset,
                    },
                },
                ("obs", ["ds"]),
                (None, None, None),
                configs,
            )

            self.assertFalse(os.path.exists(source))
            db = pp_util.get_preprocess_db(configs, ["detset"])
            self.assertEqual(len(db.inspect({})), 1)
            db.conn.close()


if __name__ == "__main__":
    unittest.main()
