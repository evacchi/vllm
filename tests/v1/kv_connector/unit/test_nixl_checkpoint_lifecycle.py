# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the NIXL checkpoint lifecycle API."""

import contextlib
import queue
import socket
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1 import multi_connector
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    base_scheduler,
    base_worker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
    NixlBaseConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlBaseConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlHandshakePayload,
)


class _TestConnector(NixlBaseConnector):
    def start_load_kv(self, *args, **kwargs):
        raise NotImplementedError


def test_release_for_checkpoint_drops_nixl_wrapper() -> None:
    worker = object.__new__(NixlBaseConnectorWorker)
    worker.nixl_wrapper = object()

    worker.release_for_checkpoint()

    assert worker.nixl_wrapper is None


def test_connector_forwards_checkpoint_lifecycle() -> None:
    worker = SimpleNamespace(
        quiesce=Mock(),
        release_for_checkpoint=Mock(),
    )
    connector = object.__new__(_TestConnector)
    connector.connector_worker = worker

    connector.quiesce(timeout=30.0)
    connector.release_for_checkpoint()

    worker.quiesce.assert_called_once_with(30.0)
    worker.release_for_checkpoint.assert_called_once_with()


def test_multi_connector_forwards_checkpoint_release() -> None:
    first = SimpleNamespace(release_for_checkpoint=Mock())
    second = SimpleNamespace(release_for_checkpoint=Mock())
    connector = object.__new__(multi_connector.MultiConnector)
    connector._connectors = [first, second]

    connector.release_for_checkpoint()

    first.release_for_checkpoint.assert_called_once_with()
    second.release_for_checkpoint.assert_called_once_with()


def test_scheduler_refreshes_after_ip_change(monkeypatch) -> None:
    scheduler = object.__new__(NixlBaseConnectorScheduler)
    scheduler.side_channel_host = "10.0.0.1"
    scheduler._nixl_handshake_listener_t = None
    scheduler._encoded_handshake_data = {}
    monkeypatch.setattr(socket, "gethostname", lambda: "pod")
    monkeypatch.setattr(socket, "gethostbyname", lambda _: "10.0.0.2")

    scheduler.refresh_handshake_endpoint()

    assert scheduler.side_channel_host == "10.0.0.2"


def test_worker_reinitialization_refreshes_scheduler_first(monkeypatch) -> None:
    events = []
    worker = object.__new__(NixlBaseConnectorWorker)
    worker.engine_id = "engine"
    worker.pp_rank = 0
    worker.tp_rank = 0
    worker.nixl_wrapper = None
    worker._registered_kv_caches = {"layer": object()}
    worker._nixl_config = object()
    worker._nixl_wrapper_cls = lambda *_: events.append("new-agent") or object()
    worker._new_handshake_executor = lambda: events.append("new-executor")
    worker.register_kv_caches = lambda _: events.append("register-caches")
    worker._publish_handshake_metadata = lambda: events.append("publish-metadata")
    worker._refresh_local_scheduler_endpoint = lambda: events.append(
        "refresh-scheduler"
    )

    worker.reinitialize()

    assert events == [
        "refresh-scheduler",
        "new-agent",
        "new-executor",
        "register-caches",
        "publish-metadata",
    ]


def test_refresh_local_scheduler_endpoint_uses_registry(monkeypatch) -> None:
    # reinitialize() must talk to the co-located scheduler directly: this
    # milestone's scope excludes P/D and multi-GPU topologies (see
    # RFC-nixl-connector-lifecycle.md), so scheduler and worker always share
    # one EngineCore process and a same-process call is correct and simpler
    # than a ZMQ round trip to a listener address the worker cannot know.
    scheduler = SimpleNamespace(refresh_handshake_endpoint=Mock())
    worker = object.__new__(NixlBaseConnectorWorker)
    worker.engine_id = "engine-1"
    monkeypatch.setattr(
        base_scheduler, "get_local_nixl_scheduler", lambda engine_id: scheduler
    )

    worker._refresh_local_scheduler_endpoint()

    scheduler.refresh_handshake_endpoint.assert_called_once_with()


def test_refresh_local_scheduler_endpoint_requires_local_scheduler(
    monkeypatch,
) -> None:
    worker = object.__new__(NixlBaseConnectorWorker)
    worker.engine_id = "engine-1"
    monkeypatch.setattr(
        base_scheduler, "get_local_nixl_scheduler", lambda engine_id: None
    )

    with pytest.raises(RuntimeError, match="local EngineCore process"):
        worker._refresh_local_scheduler_endpoint()


def test_quiesce_drains_all_lifecycle_work_before_stopping_threads() -> None:
    events = []
    worker = object.__new__(NixlBaseConnectorWorker)
    worker._recving_metadata = {"request": object()}
    worker._quiesce_drained_sending = set()
    worker._quiesce_drained_recving = set()

    def get_finished():
        events.append("get-finished")
        worker._recving_metadata.clear()
        return set(), set()

    worker.get_finished = get_finished
    worker._stop_push_writer_for_lifecycle = lambda: events.append("stop-push-writer")
    worker._stop_handshake_executor = lambda: events.append("stop-handshake-executor")
    worker._discard_push_work_for_lifecycle = lambda: events.append("discard-push-work")
    worker._release_transport_state = lambda: events.append("release-transport")

    worker.quiesce(timeout=1.0)

    assert events == [
        "get-finished",
        "stop-push-writer",
        "stop-handshake-executor",
        "discard-push-work",
        "release-transport",
    ]


def test_quiesce_tracks_handshake_and_push_work() -> None:
    worker = object.__new__(NixlBaseConnectorWorker)
    worker._handshake_futures = {"engine": object()}
    worker._ready_requests = object()
    worker._sending_transfers = {"request": [object()]}
    worker._push_finished_blocks = {"request": [[1]]}
    worker._pending_d_registrations = {"request": {}}
    worker._reg_send_inbox = object()

    assert worker._pending_lifecycle_work() == (
        "_handshake_futures",
        "_ready_requests",
        "_sending_transfers",
        "_push_finished_blocks",
        "_pending_d_registrations",
        "_reg_send_inbox",
    )


def test_quiesce_accumulates_ids_drained_while_stopping() -> None:
    worker = object.__new__(NixlBaseConnectorWorker)
    worker._recving_metadata = {"request": object()}
    worker._quiesce_drained_sending = set()
    worker._quiesce_drained_recving = set()

    def get_finished():
        worker._recving_metadata.clear()
        return {"sent-1"}, {"recv-1"}

    worker.get_finished = get_finished
    worker._stop_push_writer_for_lifecycle = lambda: None
    worker._stop_handshake_executor = lambda: None
    worker._discard_push_work_for_lifecycle = lambda: None
    worker._release_transport_state = lambda: None

    worker.quiesce(timeout=1.0)

    assert worker._quiesce_drained_sending == {"sent-1"}
    assert worker._quiesce_drained_recving == {"recv-1"}


def test_get_finished_reports_ids_drained_during_quiesce() -> None:
    worker = object.__new__(NixlBaseConnectorWorker)
    worker.transfer_topo = object()
    worker.tp_rank = 0
    worker._recving_transfers = {}
    worker._recving_metadata = {}
    worker._failed_recv_reqs = queue.Queue()
    worker._reqs_to_send = {}
    worker._get_new_notifs = lambda: set()
    worker._pop_done_transfers = lambda _transfers: set()
    worker._quiesce_drained_sending = {"sent-1"}
    worker._quiesce_drained_recving = {"recv-1"}

    done_sending, done_recving = worker.get_finished()

    assert done_sending == {"sent-1"}
    assert done_recving == {"recv-1"}
    # Reported exactly once: the accumulator is cleared after being drained.
    assert worker._quiesce_drained_sending == set()
    assert worker._quiesce_drained_recving == set()


def test_quiesce_timeout_resets_quiescing_flag() -> None:
    # No thread/executor teardown happened, so it is safe to resume normal
    # handshake operation after a drain timeout.
    worker = object.__new__(NixlBaseConnectorWorker)
    worker._pending_lifecycle_work = lambda: ("_ready_requests",)
    worker.get_finished = lambda: (set(), set())
    worker._quiesce_drained_sending = set()
    worker._quiesce_drained_recving = set()

    with contextlib.suppress(TimeoutError):
        worker.quiesce(timeout=0.0)

    assert worker._checkpoint_quiescing is False


def test_quiesce_keeps_quiescing_flag_after_teardown_before_abort() -> None:
    # Once the handshake executor and push writer are stopped, only
    # reinitialize() can restore them; the flag must stay set so callers
    # keep hitting the guarded "quiescing" error instead of using torn-down
    # state (e.g. a None executor).
    worker = object.__new__(NixlBaseConnectorWorker)
    pending_calls = [(), ("_ready_requests",)]
    worker._pending_lifecycle_work = lambda: pending_calls.pop(0)
    worker.get_finished = lambda: (set(), set())
    worker._quiesce_drained_sending = set()
    worker._quiesce_drained_recving = set()
    worker._stop_push_writer_for_lifecycle = lambda: None
    worker._stop_handshake_executor = lambda: None

    with contextlib.suppress(RuntimeError):
        worker.quiesce(timeout=1.0)

    assert worker._checkpoint_quiescing is True


def test_publish_handshake_uses_current_pod_ip(monkeypatch) -> None:
    class _Socket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def setsockopt(self, *args):
            pass

        def send(self, message):
            self.message = message

        def recv(self):
            return b"ok"

    socket_context = _Socket()
    paths = []
    worker = object.__new__(NixlBaseConnectorWorker)
    worker.xfer_handshake_metadata = NixlHandshakePayload(
        compatibility_hash="hash", agent_metadata_bytes=b"metadata"
    )
    worker.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_index=0)
    )
    worker.pp_rank = 0
    worker.tp_rank = 0
    monkeypatch.setattr(base_worker.socket, "gethostname", lambda: "pod")
    monkeypatch.setattr(base_worker.socket, "gethostbyname", lambda _: "10.0.0.9")
    monkeypatch.setattr(
        base_worker, "make_zmq_path", lambda *_args: "tcp://10.0.0.9:5600"
    )

    def _zmq_ctx(_socket_type, path):
        paths.append(path)
        return socket_context

    monkeypatch.setattr(base_worker, "zmq_ctx", _zmq_ctx)

    worker._publish_handshake_metadata()

    assert paths == ["tcp://10.0.0.9:5600"]
    assert socket_context.message
