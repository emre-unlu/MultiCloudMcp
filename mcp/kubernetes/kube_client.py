"""
Shared helpers for initializing Kubernetes API clients once per process.
"""

import threading
from typing import Callable

from kubernetes import client, config

_lock = threading.Lock()
_loaded = False


def _ensure_config_loaded() -> None:
    """Load kubeconfig exactly once in a thread-safe way."""
    global _loaded
    if _loaded:
        return
    with _lock:
        if not _loaded:
            config.load_kube_config()
            _loaded = True


def _build(factory: Callable[[], object]) -> object:
    _ensure_config_loaded()
    return factory()


def core_v1() -> client.CoreV1Api:
    return _build(client.CoreV1Api)


def apps_v1() -> client.AppsV1Api:
    return _build(client.AppsV1Api)


def custom_objects() -> client.CustomObjectsApi:
    return _build(client.CustomObjectsApi)
