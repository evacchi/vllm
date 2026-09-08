# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the NIXL checkpoint lifecycle API."""

from types import SimpleNamespace
import socket
from unittest.mock import Mock

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_worker
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlBaseConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlHandshakePayload,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
    NixlBaseConnectorScheduler,
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


def test_scheduler_refreshes_after_ip_change(monkeypatch) -> None:
    scheduler = object.__new__(NixlBaseConnectorScheduler)
    scheduler.side_channel_host = "10.0.0.1"
    scheduler._nixl_handshake_listener_t = None
    scheduler._encoded_handshake_data = {}
    monkeypatch.setattr(socket, "gethostname", lambda: "pod")
    monkeypatch.setattr(socket, "gethostbyname", lambda _: "10.0.0.2")

    scheduler.refresh_handshake_endpoint()

    assert scheduler.side_channel_host == "10.0.0.2"


def test_connector_refreshes_scheduler_before_reinitialize() -> None:
    worker = SimpleNamespace(reinitialize=Mock())
    scheduler = SimpleNamespace(refresh_handshake_endpoint=Mock())
    connector = object.__new__(_TestConnector)
    connector.connector_worker = worker
    connector.connector_scheduler = scheduler

    connector.reinitialize()

    scheduler.refresh_handshake_endpoint.assert_called_once_with()
    worker.reinitialize.assert_called_once_with()


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
