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
from collections import deque
from dataclasses import dataclass
from functools import partial
from typing import Any

import h5py


_MPI_REQUEST = 27001
_MPI_RESULT = 27002
_MPI_SESSION_CLOSE = 27003
_MPI_SESSION_CLOSED = 27004
_MPI_STOP = 27005
_MPI_STOPPED = 27006


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


def run_mpi_archive_writer_service(comm, lane, root=0):
    """Serve archive requests on a rank reserved from the compute pool."""
    from mpi4py import MPI

    state = _LaneState(lane)
    try:
        while True:
            status = MPI.Status()
            request = comm.recv(
                source=root,
                tag=MPI.ANY_TAG,
                status=status,
            )
            tag = status.Get_tag()
            if tag == _MPI_REQUEST:
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
                comm.send(result, dest=root, tag=_MPI_RESULT)
            elif tag == _MPI_SESSION_CLOSE:
                state.close()
                state = _LaneState(lane)
                comm.send(lane, dest=root, tag=_MPI_SESSION_CLOSED)
            elif tag == _MPI_STOP:
                break
            else:
                raise RuntimeError(
                    f"Archive writer lane {lane} received unknown MPI tag "
                    f"{tag}"
                )
    finally:
        state.close()
    comm.send(lane, dest=root, tag=_MPI_STOPPED)


def split_mpi_archive_writer_ranks(lane_count):
    """Reserve the final MPI ranks as archive writer services.

    This collective must be called by every rank in ``MPI.COMM_WORLD`` before
    the compute executor is constructed.  Writer ranks serve until rank zero
    calls :func:`stop_mpi_archive_writer_ranks` and then return ``is_writer``
    as true.  Compute ranks receive a communicator excluding the writers.
    """
    if not lane_count:
        return None, None, False

    try:
        from mpi4py import MPI
    except ImportError:
        return None, None, False

    world = MPI.COMM_WORLD
    if world.size == 1:
        return None, None, False
    if lane_count >= world.size - 1:
        raise ValueError(
            f"writer_lanes={lane_count} leaves fewer than one compute worker "
            f"in an MPI world of size {world.size}"
        )

    first_writer = world.size - lane_count
    writer_ranks = list(range(first_writer, world.size))
    is_writer = world.rank >= first_writer
    compute_comm = world.Split(
        MPI.UNDEFINED if is_writer else 0,
        key=world.rank,
    )
    if is_writer:
        run_mpi_archive_writer_service(
            world,
            lane=world.rank - first_writer,
        )
        return None, None, True
    return compute_comm, (world, writer_ranks), False


def stop_mpi_archive_writer_ranks(comm, writer_ranks):
    """Stop persistent writer services and drain any outstanding replies."""
    from mpi4py import MPI

    for rank in writer_ranks:
        comm.send(None, dest=rank, tag=_MPI_STOP)

    stopped = set()
    while len(stopped) < len(writer_ranks):
        status = MPI.Status()
        comm.recv(
            source=MPI.ANY_SOURCE,
            tag=MPI.ANY_TAG,
            status=status,
        )
        tag = status.Get_tag()
        if tag == _MPI_STOPPED:
            stopped.add(status.Get_source())
        elif tag not in (_MPI_RESULT, _MPI_SESSION_CLOSED):
            raise RuntimeError(
                f"Unexpected MPI tag {tag} while stopping archive writers"
            )


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
        while True:
            try:
                self._request_queues[lane].put(request, timeout=1)
                break
            except queue.Full:
                self.check_health()
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

    def iter_results(self):
        while self._pending:
            try:
                yield self.get_result(timeout=1)
            except queue.Empty:
                self.check_health()

    def check_health(self):
        failures = [
            f"{process.name} exited with {process.exitcode}"
            for process in self._processes
            if process.exitcode not in (None, 0)
        ]
        if failures:
            raise RuntimeError("; ".join(failures))

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

    def terminate(self):
        self._closed = True
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        for process in self._processes:
            process.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is None:
            self.join()
        else:
            self.terminate()
        return False


class MPIArchiveWriterPool:
    """Archive writer pool backed by ranks reserved from ``COMM_WORLD``."""

    def __init__(self, comm, writer_ranks, queue_depth=8):
        if not writer_ranks:
            raise ValueError("At least one MPI archive writer rank is required")
        if queue_depth < 1:
            raise ValueError("queue_depth must be at least one")
        self.comm = comm
        self.writer_ranks = list(writer_ranks)
        self.lane_count = len(self.writer_ranks)
        self.queue_depth = queue_depth
        self._pending = set()
        self._pending_by_lane = [0] * self.lane_count
        self._ready = deque()
        self._closed = False
        self._joined = False

    @property
    def pending(self):
        return len(self._pending)

    def _receive_mpi_result(self, block=True, timeout=None):
        from mpi4py import MPI

        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.comm.iprobe(source=MPI.ANY_SOURCE, tag=_MPI_RESULT):
            if not block:
                raise queue.Empty
            if deadline is not None and time.monotonic() >= deadline:
                raise queue.Empty
            time.sleep(0.01)

        result = self.comm.recv(source=MPI.ANY_SOURCE, tag=_MPI_RESULT)
        if result.request_id not in self._pending:
            raise RuntimeError(
                f"Unexpected archive result {result.request_id} from MPI lane "
                f"{result.lane}"
            )
        self._pending.remove(result.request_id)
        self._pending_by_lane[result.lane] -= 1
        return result

    def _receive_result(self, block=True, timeout=None):
        if self._ready:
            return self._ready.popleft()
        return self._receive_mpi_result(block=block, timeout=timeout)

    def submit(self, request: ArchiveRequest):
        if self._closed:
            raise RuntimeError("MPIArchiveWriterPool is closed")
        if request.request_id in self._pending:
            raise ValueError(f"Duplicate request id {request.request_id}")
        lane = lane_for_dataset(request.db_data["dataset"], self.lane_count)
        while self._pending_by_lane[lane] >= self.queue_depth:
            self._ready.append(self._receive_mpi_result())
        self.comm.send(
            request,
            dest=self.writer_ranks[lane],
            tag=_MPI_REQUEST,
        )
        self._pending.add(request.request_id)
        self._pending_by_lane[lane] += 1
        return request.request_id

    def get_result(self, block=True, timeout=None):
        return self._receive_result(block=block, timeout=timeout)

    def get_available(self):
        results = []
        while True:
            try:
                results.append(self.get_result(block=False))
            except queue.Empty:
                return results

    def iter_results(self):
        while self._ready or self._pending:
            yield self.get_result()

    def check_health(self):
        return None

    def close_submissions(self):
        if self._closed:
            return
        self._closed = True
        for rank in self.writer_ranks:
            self.comm.send(None, dest=rank, tag=_MPI_SESSION_CLOSE)

    def join(self):
        if self._joined:
            return
        self.close_submissions()
        for rank in self.writer_ranks:
            self.comm.recv(source=rank, tag=_MPI_SESSION_CLOSED)
        self._joined = True

    def terminate(self):
        self.close_submissions()
        for _ in self.iter_results():
            pass
        self.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is None:
            for _ in self.iter_results():
                pass
            self.join()
        else:
            self.terminate()
        return False


class ArchivePublisher:
    """Connect archive writer results to rank-zero manifest managers."""

    def __init__(
        self,
        configs,
        db_managers,
        lane_count=4,
        queue_depth=8,
        overwrite=False,
        mpi_comm=None,
        writer_ranks=None,
    ):
        self.configs = configs
        self.db_managers = db_managers
        self.overwrite = overwrite
        if mpi_comm is None:
            self.writers = ArchiveWriterPool(
                lane_count=lane_count,
                queue_depth=queue_depth,
            )
        else:
            if len(writer_ranks) != lane_count:
                raise ValueError(
                    f"Received {len(writer_ranks)} MPI writer ranks for "
                    f"{lane_count} archive lanes"
                )
            self.writers = MPIArchiveWriterPool(
                comm=mpi_comm,
                writer_ranks=writer_ranks,
                queue_depth=queue_depth,
            )
        self._tokens = {}

    def submit(self, archive_name, out_dict, token, recover=False):
        request = request_from_output(
            archive_name,
            out_dict,
            self.configs[archive_name],
            overwrite=self.overwrite,
            recover=recover,
        )
        self._tokens[request.request_id] = token
        self.writers.submit(request)
        return request.request_id

    @staticmethod
    def _after_commit(result, token, callback):
        callback(token, result)
        try:
            os.remove(result.temp_file)
        except FileNotFoundError:
            pass

    def finish(self, on_commit, on_error):
        """Drain writers and stage successful results in manifest batches."""
        try:
            for result in self.writers.iter_results():
                token = self._tokens.pop(result.request_id)
                if result.success:
                    self.db_managers[result.archive_name].add_entry(
                        result.db_data,
                        result.h5_path,
                        replace=self.overwrite,
                        on_commit=partial(
                            self._after_commit,
                            result,
                            token,
                            on_commit,
                        ),
                    )
                else:
                    on_error(token, result)
            self.writers.join()
        except Exception:
            self.writers.terminate()
            raise
