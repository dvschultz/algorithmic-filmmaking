"""Derived media cache publication, reader retention, and safe eviction."""

from contextlib import contextmanager

import pytest

from core.artifacts import ArtifactStore
from core.media_cache import MediaCache


def test_cache_verifies_payload_and_keeps_reader_alive_after_prune(tmp_path):
    cache = MediaCache(tmp_path / "cache", tmp_path / "artifacts")
    source = tmp_path / "render.mp4"
    source.write_bytes(b"rendered")
    result = cache.publish("sequence", "key", source)
    assert result.path.read_bytes() == b"rendered"
    hit = cache.get("sequence", "key")
    assert hit is not None and hit.path == result.path
    assert cache.prune(keep_latest=0) == 1
    artifacts = ArtifactStore(tmp_path / "artifacts")
    assert artifacts.collect() == []
    result.lease.close()
    assert artifacts.collect() == []
    hit.lease.close()
    assert len(artifacts.collect()) == 1
    assert source.read_bytes() == b"rendered"


def test_cache_damage_is_a_miss_and_unknown_files_survive(tmp_path):
    cache = MediaCache(tmp_path / "cache", tmp_path / "artifacts")
    source = tmp_path / "render.mp4"
    source.write_bytes(b"rendered")
    result = cache.publish("sequence", "key", source)
    result.path.write_bytes(b"damaged")
    assert cache.get("sequence", "key") is None
    unknown = tmp_path / "cache" / "unknown.mp4"
    unknown.write_bytes(b"user media")
    cache.prune(keep_latest=0)
    assert unknown.read_bytes() == b"user media"


def test_pruning_is_per_namespace_and_retains_latest(tmp_path):
    cache = MediaCache(tmp_path / "cache", tmp_path / "artifacts")
    source = tmp_path / "render.mp4"
    for namespace, key in [("a", "old"), ("a", "new"), ("b", "only")]:
        source.write_bytes(f"{namespace}-{key}".encode())
        cache.publish(namespace, key, source).lease.close()
    assert cache.prune(keep_latest=1) == 1
    assert cache.get("a", "old") is None
    assert cache.get("a", "new") is not None
    assert cache.get("b", "only") is not None


def test_prune_during_lookup_keeps_reader_payload(tmp_path, monkeypatch):
    cache = MediaCache(tmp_path / "cache", tmp_path / "artifacts")
    source = tmp_path / "render.mp4"
    source.write_bytes(b"rendered")
    cache.publish("sequence", "key", source).lease.close()
    available = cache.artifacts.available

    def prune_then_verify(ref):
        assert cache.prune(0) == 1
        return available(ref)

    monkeypatch.setattr(cache.artifacts, "available", prune_then_verify)
    result = cache.get("sequence", "key")
    assert result is not None and result.path.read_bytes() == b"rendered"
    result.lease.close()
    assert len(cache.artifacts.collect()) == 1


def test_uncertain_media_publication_keeps_recoverable_output(tmp_path, monkeypatch):
    cache = MediaCache(tmp_path / "cache", tmp_path / "artifacts")
    source = tmp_path / "render.mp4"
    source.write_bytes(b"rendered")
    connection = cache._connection

    @contextmanager
    def uncertain_close():
        with connection() as db:
            yield db
        raise OSError("uncertain close")

    monkeypatch.setattr(cache, "_connection", uncertain_close)
    with pytest.raises(OSError, match="uncertain close"):
        cache.publish("sequence", "key", source)
    assert cache.artifacts.collect() == []
    recovered = MediaCache(tmp_path / "cache", tmp_path / "artifacts")
    result = recovered.get("sequence", "key")
    assert result is not None and result.path.read_bytes() == b"rendered"
