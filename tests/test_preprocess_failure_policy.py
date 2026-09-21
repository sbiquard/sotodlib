import os
import tempfile
import unittest

from sotodlib.preprocess import preprocess_util as pp_util
from sotodlib.site_pipeline import jobdb
from sotodlib.site_pipeline import preprocess_tod

from ._helpers import mpi_multi


@unittest.skipIf(mpi_multi(), "Running with multiple MPI processes")
class TestPreprocessFailurePolicy(unittest.TestCase):
    def test_layer_classification(self):
        proc_failure = pp_util.PreprocessFailure.from_errors((
            pp_util.PreprocessErrors.ProcPipelineStepError,
            "no detectors remain",
            None,
        ))
        self.assertEqual(proc_failure.layer, "proc")
        self.assertEqual(
            proc_failure.severity, pp_util.FailureSeverity.expected
        )
        self.assertIsNone(proc_failure.for_layer("init"))

        init_failure = pp_util.PreprocessFailure.from_errors((
            pp_util.PreprocessErrors.InitPipeLineRunError,
            "bad configuration",
            None,
        ))
        self.assertEqual(init_failure.layer, "init")
        self.assertEqual(
            init_failure.for_layer("proc").category,
            pp_util.PreprocessErrors.BlockedByInitError,
        )

    def test_circuit_breakers(self):
        failure = pp_util.PreprocessFailure.from_errors((
            pp_util.PreprocessErrors.ExecutorFutureError,
            "ValueError: repeated bug",
            None,
        ))
        tracker = pp_util.PreprocessFailureTracker(max_repeated_error=2)
        tracker.record("obs1", ["g1"], failure)
        with self.assertRaises(pp_util.PreprocessCircuitBreaker):
            tracker.record("obs2", ["g2"], failure)

        infrastructure = pp_util.PreprocessFailure.from_errors(
            (pp_util.PreprocessErrors.ExecutorFutureError, "disk", None),
            exception=OSError("disk"),
        )
        with self.assertRaises(pp_util.PreprocessCircuitBreaker):
            pp_util.PreprocessFailureTracker().record(
                "obs", ["group"], infrastructure
            )

    def test_end_of_run_policy(self):
        expected = pp_util.PreprocessFailure.from_errors((
            pp_util.PreprocessErrors.InitPipelineStepError,
            "selection removed all detectors",
            None,
        ))
        tracker = pp_util.PreprocessFailureTracker(max_repeated_error=0)
        tracker.record("obs", ["group"], expected)
        tracker.raise_if_needed(total_groups=2)
        with self.assertRaises(pp_util.PreprocessFailureSummary):
            tracker.raise_if_needed(total_groups=2, max_failed_fraction=0.4)
        with self.assertRaises(pp_util.PreprocessFailureSummary):
            tracker.raise_if_needed(total_groups=2, raise_error=True)

    def test_raise_error_cli_is_a_flag(self):
        parser = preprocess_tod.get_parser()
        self.assertTrue(
            parser.parse_args(["config.yaml", "--raise-error"]).raise_error
        )
        self.assertFalse(parser.parse_args(["config.yaml"]).raise_error)

    def test_layer_aware_job_updates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = jobdb.JobManager(
                sqlite_file=os.path.join(temp_dir, "jobs.sqlite")
            )
            init_job = manager.create_job(
                "init", {"obs:obs_id": "obs", "error": None}
            )
            proc_job = manager.create_job(
                "proc", {"obs:obs_id": "obs", "error": None}
            )
            failure = pp_util.PreprocessFailure.from_errors((
                pp_util.PreprocessErrors.ProcPipeLineRunError,
                "ValueError: proc failed",
                None,
            ))
            pp_util.update_jobdb(manager, [
                pp_util.get_jobdb_update(init_job, failure, "init"),
                pp_util.get_jobdb_update(proc_job, failure, "proc"),
            ])

            result = {job.jclass: job for job in manager.get_jobs()}
            self.assertEqual(result["init"].jstate, jobdb.JState.done)
            self.assertEqual(result["proc"].jstate, jobdb.JState.failed)
            self.assertEqual(
                result["proc"].tags["error"],
                pp_util.PreprocessErrors.ProcPipeLineRunError,
            )


if __name__ == "__main__":
    unittest.main()
