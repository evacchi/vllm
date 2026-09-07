# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the NIXL checkpoint lifecycle API."""

from types import SimpleNamespace
from unittest.mock import Mock

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlBaseConnector,
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
