import os
import tempfile
import unittest

import h5py
import numpy as np
import so3g
from scipy.sparse import csr_array

from sotodlib import core
from sotodlib.core.metadata.manifest import DbBatchManager
from sotodlib.preprocess import preprocess_util as pp_util


class TestPreprocessArchivePublication(unittest.TestCase):
    def test_recursive_copy_round_trips_axismanager_encodings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "temp.h5")
            dataset = "obs_dets_detset_ds"
            configs = {
                "archive": {
                    "index": os.path.join(temp_dir, "manifest.sqlite"),
                    "policy": {
                        "type": "simple",
                        "filename": os.path.join(temp_dir, "archive.h5"),
                    },
                },
            }

            dets = core.LabelAxis("dets", ["a", "b"])
            samps = core.OffsetAxis("samps", 8)
            aman = core.AxisManager(dets, samps)
            signal = np.arange(16, dtype=np.float32).reshape(2, 8)
            aman.wrap(
                "signal", signal, [(0, "dets"), (1, "samps")]
            )
            aman.wrap("scalar", 7)

            flags = core.FlagManager.for_tod(aman, "dets", "samps")
            flags.wrap(
                "cuts",
                so3g.proj.RangesMatrix.zeros(aman.shape),
                [(0, "dets"), (1, "samps")],
            )
            aman.wrap("flags", flags)

            sub = core.AxisManager(dets)
            sub.wrap("gain", np.array([1.5, 2.5]), [(0, "dets")])
            sub.wrap(
                "sparse",
                csr_array(np.eye(2)),
                [(0, "dets"), (1, "dets")],
            )
            aman.wrap("sub", sub)
            aman.save(source, dataset)

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

            destination = os.path.join(temp_dir, "archive_000.h5")
            loaded = core.AxisManager.load(destination, dataset)
            np.testing.assert_array_equal(loaded.signal, signal)
            np.testing.assert_array_equal(loaded.sub.gain, sub.gain)
            np.testing.assert_array_equal(
                loaded.sub.sparse.toarray(), sub.sparse.toarray()
            )
            self.assertEqual(loaded.scalar, aman.scalar)
            self.assertIsInstance(loaded.flags, core.FlagManager)
            self.assertIsInstance(
                loaded.flags.cuts, so3g.proj.RangesMatrix
            )

            with h5py.File(destination, "r") as handle:
                self.assertNotIn(dataset, handle[dataset])

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
