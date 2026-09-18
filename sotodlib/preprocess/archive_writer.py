"""Parallel writers for consolidating preprocessing archive files.

Each writer lane is the sole process allowed to write its lane's HDF5 files.
This preserves the existing consolidated archive layout without funneling all
HDF5 copies through the main process.
"""

from __future__ import annotations

import glob
import hashlib
import multiprocessing
import os
import queue
import re
import time
import traceback
import uuid
from dataclasses import dataclass
from typing import Any

import h5py


@dataclass(frozen=True)
class ArchiveRequest:
    """A temporary preprocessing file ready for archive publication."""

    archive_name: str
    temp_file: str
    db_data: dict[str, Any]
    policy_filename: str
    index_filename: str
    overwrite: bool = False
    recover: bool = False
    max_file_size: int = 10_000_000_000
    request_id: str = ""

    def __post_init__(self):
        if not self.request_id:
            object.__setattr__(self, "request_id", uuid.uuid4().hex)


@dataclass(frozen=True)
class ArchiveResult:
    """Result returned by an archive writer lane."""

    request_id: str
    archive_name: str
    temp_file: str
    db_data: dict[str, Any]
    h5_path: str | None
    lane: int
    archive_file: str | None
    elapsed: float
    error: str | None = None
    traceback: str | None = None

    @property
    def success(self):
        return self.error is None


def lane_for_dataset(dataset: str, lane_count: int) -> int:
    """Return a deterministic archive lane for a dataset name."""
    if lane_count < 1:
        raise ValueError("lane_count must be at least one")
    digest = hashlib.blake2b(dataset.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % lane_count


def request_from_output(
    archive_name: str,
    out_dict: dict[str, Any],
    configs: dict[str, Any],
    overwrite: bool = False,
    recover: bool = False,
) -> ArchiveRequest:
    """Build an :class:`ArchiveRequest` from preprocessing output."""
    archive = configs["archive"]
    return ArchiveRequest(
        archive_name=archive_name,
        temp_file=out_dict["temp_file"],
        db_data=out_dict["db_data"],
        policy_filename=archive["policy"]["filename"],
        index_filename=archive["index"],
        overwrite=overwrite,
        recover=recover,
        max_file_size=archive.get("max_file_size", 10_000_000_000),
    )


class _LaneArchive:
    """Open-file state for one archive in one writer lane."""

    def __init__(self, request: ArchiveRequest, lane: int):
        self.lane = lane
        self.basename = os.path.splitext(request.policy_filename)[0]
        self.index_dir = os.path.dirname(request.index_filename)
        self.max_file_size = request.max_file_size
        self.sequence = None
        self.filename = None
        self.handle = None
        os.makedirs(os.path.dirname(request.policy_filename), exist_ok=True)

    @property
    def pattern(self):
        return f"{self.basename}_lane{self.lane:02d}_*.h5"

    def _filename(self, sequence):
        return f"{self.basename}_lane{self.lane:02d}_{sequence:03d}.h5"

    def _existing_files(self):
        pattern = re.compile(
            rf"{re.escape(self.basename)}_lane{self.lane:02d}_(\d+)\.h5$"
        )
        found = []
        for filename in glob.glob(self.pattern):
            match = pattern.match(filename)
            if match:
                found.append((int(match.group(1)), filename))
        return sorted(found)

    def _open(self, sequence):
        self.close()
        self.sequence = sequence
        self.filename = self._filename(sequence)
        self.handle = h5py.File(self.filename, "a")

    def ensure_open(self, incoming_size):
        if self.handle is None:
            existing = self._existing_files()
            sequence = existing[-1][0] if existing else 0
            self._open(sequence)

        current_size = os.path.getsize(self.filename)
        if (
            len(self.handle) > 0
            and current_size + incoming_size > self.max_file_size
        ):
            self._open(self.sequence + 1)

    def find_dataset(self, dataset):
        """Find a dataset in this lane, used only during recovery."""
        if self.handle is not None and dataset in self.handle:
            return self.filename
        for _, filename in reversed(self._existing_files()):
            if filename == self.filename:
                continue
            with h5py.File(filename, "r") as archive:
                if dataset in archive:
                    return filename
        return None

    def close(self):
        if self.handle is not None:
            self.handle.flush()
            self.handle.close()
            self.handle = None


class _LaneState:
    def __init__(self, lane):
        self.lane = lane
        self.archives = {}

    def _get_archive(self, request):
        key = (
            request.archive_name,
            request.policy_filename,
            request.index_filename,
        )
        if key not in self.archives:
            self.archives[key] = _LaneArchive(request, self.lane)
        return self.archives[key]

    def publish(self, request):
        start = time.monotonic()
        archive = self._get_archive(request)
        dataset = request.db_data["dataset"]

        if request.recover and not request.overwrite:
            filename = archive.find_dataset(dataset)
            if filename is not None:
                return ArchiveResult(
                    request_id=request.request_id,
                    archive_name=request.archive_name,
                    temp_file=request.temp_file,
                    db_data=request.db_data,
                    h5_path=os.path.relpath(filename, start=archive.index_dir),
                    lane=self.lane,
                    archive_file=filename,
                    elapsed=time.monotonic() - start,
                )

        incoming_size = os.path.getsize(request.temp_file)
        archive.ensure_open(incoming_size)
        dest = archive.handle
        staging = f".__copying__.{request.request_id}"

        with h5py.File(request.temp_file, "r") as source:
            if dataset not in source:
                raise KeyError(
                    f"Expected {dataset!r} in temporary file "
                    f"{request.temp_file!r}; found {list(source.keys())!r}"
                )
            if staging in dest:
                del dest[staging]
            source.copy(source[dataset], dest, staging)
            dest.flush()

        if dataset in dest:
            if not request.overwrite:
                del dest[staging]
                dest.flush()
                raise RuntimeError(
                    f"Destination dataset {dataset!r} already exists in "
                    f"{archive.filename!r}"
                )
            del dest[dataset]
        dest.move(staging, dataset)
        dest.flush()

        return ArchiveResult(
            request_id=request.request_id,
            archive_name=request.archive_name,
            temp_file=request.temp_file,
            db_data=request.db_data,
            h5_path=os.path.relpath(archive.filename, start=archive.index_dir),
            lane=self.lane,
            archive_file=archive.filename,
            elapsed=time.monotonic() - start,
        )

    def close(self):
        for archive in self.archives.values():
            archive.close()


def publish_archive_request(request: ArchiveRequest, lane: int = 0):
    """Publish one request synchronously, primarily for compatibility/tests."""
    state = _LaneState(lane)
    try:
        return state.publish(request)
    finally:
        state.close()


def _writer_main(lane, request_queue, result_queue):
    state = _LaneState(lane)
    try:
        while True:
            request = request_queue.get()
            if request is None:
                break
            try:
                result = state.publish(request)
            except Exception as error:
                result = ArchiveResult(
                    request_id=request.request_id,
                    archive_name=request.archive_name,
                    temp_file=request.temp_file,
                    db_data=request.db_data,
                    h5_path=None,
                    lane=lane,
                    archive_file=None,
                    elapsed=0,
                    error=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc(),
                )
            result_queue.put(result)
    finally:
        state.close()


class ArchiveWriterPool:
    """A process pool with exactly one writer for each archive lane."""

    def __init__(self, lane_count=4, queue_depth=8, mp_context="spawn"):
        if lane_count < 1:
            raise ValueError("lane_count must be at least one")
        if queue_depth < 1:
            raise ValueError("queue_depth must be at least one")
        self.lane_count = lane_count
        self._context = multiprocessing.get_context(mp_context)
        self._result_queue = self._context.Queue()
        self._request_queues = [
            self._context.Queue(maxsize=queue_depth) for _ in range(lane_count)
        ]
        self._processes = [
            self._context.Process(
                target=_writer_main,
                args=(lane, self._request_queues[lane], self._result_queue),
                name=f"preproc-archive-writer-{lane}",
            )
            for lane in range(lane_count)
        ]
        self._pending = set()
        self._closed = False
        for process in self._processes:
            process.start()

    @property
    def pending(self):
        return len(self._pending)

    def submit(self, request: ArchiveRequest):
        if self._closed:
            raise RuntimeError("ArchiveWriterPool is closed")
        if request.request_id in self._pending:
            raise ValueError(f"Duplicate request id {request.request_id}")
        lane = lane_for_dataset(request.db_data["dataset"], self.lane_count)
        self._request_queues[lane].put(request)
        self._pending.add(request.request_id)
        return request.request_id

    def get_result(self, block=True, timeout=None):
        result = self._result_queue.get(block=block, timeout=timeout)
        self._pending.discard(result.request_id)
        return result

    def get_available(self):
        results = []
        while True:
            try:
                results.append(self.get_result(block=False))
            except queue.Empty:
                return results

    def close_submissions(self):
        if self._closed:
            return
        self._closed = True
        for request_queue in self._request_queues:
            request_queue.put(None)

    def join(self):
        self.close_submissions()
        for process in self._processes:
            process.join()
        failures = [
            f"{process.name} exited with {process.exitcode}"
            for process in self._processes
            if process.exitcode != 0
        ]
        if failures:
            raise RuntimeError("; ".join(failures))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.join()
        return False
