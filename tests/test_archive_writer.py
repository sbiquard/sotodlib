import os
import tempfile
import unittest

import h5py
import numpy as np

from sotodlib.preprocess.archive_writer import (
    ArchivePublisher,
    ArchiveRequest,
    ArchiveWriterPool,
    lane_for_dataset,
    publish_archive_request,
)
from sotodlib.core.metadata.manifest import (
    DbBatchManager,
    ManifestDb,
    ManifestScheme,
)


class TestArchiveWriter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = self.temp_dir.name
        self.archive_base = os.path.join(self.root, "preprocess_archive.h5")
        self.index = os.path.join(self.root, "process_archive.sqlite")

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_request(self, dataset, value, **kwargs):
        temp_file = os.path.join(self.root, f"temp-{dataset}.h5")
        with h5py.File(temp_file, "w") as output:
            group = output.create_group(dataset)
            group.create_dataset("value", data=np.asarray([value]))
        return ArchiveRequest(
            archive_name="init",
            temp_file=temp_file,
            db_data={"obs:obs_id": dataset, "dataset": dataset},
            policy_filename=self.archive_base,
            index_filename=self.index,
            **kwargs,
        )

    def test_lane_assignment_is_stable(self):
        self.assertEqual(lane_for_dataset("dataset", 7), 2)
        self.assertEqual(lane_for_dataset("dataset", 7), 2)
        with self.assertRaises(ValueError):
            lane_for_dataset("dataset", 0)

    def test_publish_and_recover_existing_group(self):
        request = self.make_request("obs0", 12)
        result = publish_archive_request(request, lane=2)
        self.assertTrue(result.success)
        self.assertIn("_lane02_000.h5", result.archive_file)
        with h5py.File(result.archive_file, "r") as archive:
            np.testing.assert_array_equal(archive["obs0/value"][:], [12])

        recovery = ArchiveRequest(
            **{
                **request.__dict__,
                "request_id": "recovery",
                "recover": True,
            }
        )
        recovered = publish_archive_request(recovery, lane=2)
        self.assertTrue(recovered.success)
        self.assertEqual(recovered.archive_file, result.archive_file)

    def test_writer_pool_uses_independent_lane_files(self):
        requests = [self.make_request(f"obs{i}", i) for i in range(12)]
        expected_lanes = {
            request.request_id: lane_for_dataset(request.db_data["dataset"], 3)
            for request in requests
        }
        with ArchiveWriterPool(lane_count=3, queue_depth=4) as writers:
            for request in requests:
                writers.submit(request)
            results = [writers.get_result() for _ in requests]

        self.assertTrue(all(result.success for result in results))
        self.assertEqual(
            {result.lane for result in results}, set(expected_lanes.values())
        )
        for result in results:
            self.assertEqual(result.lane, expected_lanes[result.request_id])
            self.assertIn(f"_lane{result.lane:02d}_", result.archive_file)
            with h5py.File(result.archive_file, "r") as archive:
                self.assertIn(result.db_data["dataset"], archive)

    def test_rollover_is_independent_per_lane(self):
        first = self.make_request("first", 1, max_file_size=1)
        second = self.make_request("second", 2, max_file_size=1)
        first_result = publish_archive_request(first, lane=0)
        second_result = publish_archive_request(second, lane=0)
        self.assertNotEqual(first_result.archive_file, second_result.archive_file)
        self.assertTrue(first_result.archive_file.endswith("_000.h5"))
        self.assertTrue(second_result.archive_file.endswith("_001.h5"))

    def test_publisher_removes_temp_only_after_manifest_commit(self):
        scheme = ManifestScheme()
        scheme.add_exact_match('obs:obs_id')
        scheme.add_data_field('dataset')
        db = ManifestDb(self.index, scheme=scheme)
        request = self.make_request("obs0", 4)
        committed = []
        failed = []

        with DbBatchManager(db, batch_size=10) as db_manager:
            publisher = ArchivePublisher(
                configs={
                    "init": {
                        "archive": {
                            "index": self.index,
                            "policy": {"filename": self.archive_base},
                        }
                    }
                },
                db_managers={"init": db_manager},
                lane_count=1,
            )
            publisher.submit(
                "init",
                {"temp_file": request.temp_file, "db_data": request.db_data},
                token="job0",
            )
            publisher.finish(
                on_commit=lambda token, result: committed.append(token),
                on_error=lambda token, result: failed.append(token),
            )
            self.assertTrue(os.path.exists(request.temp_file))
            self.assertEqual(committed, [])

        self.assertEqual(committed, ["job0"])
        self.assertEqual(failed, [])
        self.assertFalse(os.path.exists(request.temp_file))
        self.assertEqual(db.inspect({"obs:obs_id": "obs0"})[0]["dataset"], "obs0")
        db.conn.close()


if __name__ == "__main__":
    unittest.main()
