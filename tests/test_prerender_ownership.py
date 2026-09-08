"""Sequence and undo ownership outlive disposable prerender cache entries."""

from pathlib import Path

import pytest

from core.artifacts import ArtifactLease, ArtifactStore
from core.media_cache import MediaCache
from core.project import Project
from core.project_export import export_project_bundle
from models.clip import Clip, Source


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    monkeypatch.setattr("core.paths.get_artifact_store_dir", lambda: root)
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source")
    source = Source(file_path=source_path, fps=30)
    clip = Clip(source_id=source.id, start_frame=0, end_frame=30)
    project = Project.new()
    project.add_source(source)
    project.add_clips([clip])
    project.add_to_sequence([clip.id])
    rendered = tmp_path / "render.mp4"
    rendered.write_bytes(b"rendered")
    cache = MediaCache(tmp_path / "cache", root)
    result = cache.publish("transform", "key", rendered)
    entry = project.sequence.get_all_clips()[0]
    entry.prerendered_path = str(result.path)
    entry.prerender_artifact = result.reference
    project.retain_artifacts()
    result.lease.close()
    return project, entry, cache, result.reference


def test_sequence_and_undo_retain_prerender_after_cache_eviction(setup):
    project, entry, cache, ref = setup
    cache.prune(0)
    assert cache.artifacts.read_bytes(ref) == b"rendered"
    project.remove_from_sequence([entry.id])
    assert cache.artifacts.collect() == []
    project.session.undo()
    assert project.sequence.get_all_clips()[0].prerender_artifact == ref
    project.remove_from_sequence([entry.id])
    project.session.reset()
    assert cache.artifacts.collect() == [ref.digest]


def test_trim_clears_prerender_and_undo_restores_its_reference(setup):
    project, entry, cache, ref = setup
    cache.prune(0)
    project.update_sequence_clip(entry.id, out_point=20)
    assert entry.prerendered_path is None and entry.prerender_artifact is None
    assert cache.artifacts.collect() == []
    project.session.undo()
    assert entry.prerender_artifact == ref
    assert Path(entry.prerendered_path).is_file()


def test_closed_project_manifest_retains_and_restores_prerender(setup, tmp_path):
    project, entry, cache, ref = setup
    path = tmp_path / "saved" / "project.json"
    assert project.save(path)
    cache.prune(0)
    project.session.close()
    assert cache.artifacts.collect() == []
    loaded = Project.load(path)
    restored = loaded.sequence.get_all_clips()[0]
    assert restored.prerender_artifact == ref
    assert Path(restored.prerendered_path).read_bytes() == b"rendered"
    cache.artifacts.path_for(ref).unlink()
    damaged = Project.load(path)
    assert damaged.sequence.get_all_clips()[0].prerendered_path is None
    assert damaged.sequence.get_all_clips()[0].prerender_artifact == ref


def test_save_snapshot_retains_prerender_after_editor_and_cache_retire(setup):
    project, entry, cache, ref = setup
    snapshot = project.snapshot_for_save()
    lease = ArtifactLease.for_snapshot(snapshot)
    cache.prune(0)
    project.clear()
    assert cache.artifacts.collect() == []
    lease.close()
    assert cache.artifacts.collect() == [ref.digest]


def test_bundle_restores_prerender_into_fresh_artifact_store(setup, tmp_path, monkeypatch):
    project, entry, cache, ref = setup
    result = export_project_bundle(project, tmp_path / "bundle", include_clips=False)
    assert result.artifacts_copied == 1
    fresh = tmp_path / "fresh-artifacts"
    monkeypatch.setattr("core.paths.get_artifact_store_dir", lambda: fresh)
    project_path = next((tmp_path / "bundle").glob("*.sceneripper"), None)
    if project_path is None:
        project_path = next((tmp_path / "bundle").glob("*.json"))
    loaded = Project.load(project_path)
    restored = loaded.sequence.get_all_clips()[0]
    assert restored.prerender_artifact == ref
    assert Path(restored.prerendered_path).read_bytes() == b"rendered"
    assert ArtifactStore(fresh).read_bytes(ref) == b"rendered"


def test_source_removal_undo_retains_prerender(setup):
    project, entry, cache, ref = setup
    project.session.reset()
    cache.prune(0)
    project.remove_source(entry.source_id)
    assert cache.artifacts.collect() == []
    project.session.undo()
    assert project.sequence.get_all_clips()[0].prerender_artifact == ref


def test_unreadable_prerender_reference_preserves_other_project_data(setup, tmp_path):
    import json

    project, entry, cache, ref = setup
    project.clips[0].notes = "Keep these notes"
    path = tmp_path / "project.json"
    assert project.save(path)
    data = json.loads(path.read_text())
    damaged = {"unknown_version": 9, "sha256": "bad"}
    data["sequences"][0]["tracks"][0]["clips"][0]["prerender_artifact"] = damaged
    path.write_text(json.dumps(data))
    loaded = Project.load(path)
    assert loaded.clips[0].notes == "Keep these notes"
    restored = loaded.sequence.get_all_clips()[0]
    assert restored.prerender_artifact is None and restored.prerendered_path is None
    assert restored.to_dict()["prerender_artifact"] == damaged


def test_registered_prerender_binding_requires_unchanged_managed_file(setup):
    from core.artifacts import bind_prerender_path
    from models.sequence import SequenceClip

    project, entry, cache, ref = setup
    draft = SequenceClip(out_point=30)
    path = cache.artifacts.path_for(ref)
    bind_prerender_path(draft, path, project.artifact_store)
    assert draft.prerender_artifact == ref
    path.write_bytes(b"changed")
    bind_prerender_path(draft, path, project.artifact_store)
    assert draft.prerendered_path is None and draft.prerender_artifact is None


def test_batch_leases_bridge_cache_eviction_until_project_binding(setup, monkeypatch):
    from types import SimpleNamespace
    from core.artifacts import bind_prerender_path
    from core.remix.prerender import prerender_batch, cleanup_transform_cache

    project, entry, cache, ref = setup
    source = project.sources[0]

    def render(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(cmd[cmd.index("-ss") + 1].encode())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("core.remix.prerender.find_binary", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr("core.remix.prerender.subprocess.run", render)
    clips = [Clip(source_id=source.id, start_frame=i, end_frame=30) for i in range(6)]
    batch = prerender_batch([(clip, source, {"hflip": True}) for clip in clips], cache.root)
    paths = [result[2] for result in batch]
    assert len(paths) == 6 and all(path.is_file() for path in paths)
    bind_prerender_path(entry, paths[0], project.artifact_store)
    project.retain_artifacts()
    batch.close()
    cleanup_transform_cache(cache.root, keep_latest=0)
    assert paths[0].is_file()
    assert all(not path.exists() for path in paths[1:])


def test_non_active_sequence_retains_and_restores_prerender(setup, tmp_path):
    from copy import deepcopy
    from models.sequence import Sequence

    project, entry, cache, ref = setup
    other = Sequence(name="Other")
    other.tracks[0].clips.append(deepcopy(entry))
    project.add_sequence(other)
    project.remove_from_sequence([entry.id])
    project.session.reset()
    path = tmp_path / "project.json"
    assert project.save(path)
    cache.prune(0)
    loaded = Project.load(path)
    restored = loaded.sequences[1].get_all_clips()[0]
    assert restored.prerender_artifact == ref
    assert Path(restored.prerendered_path).read_bytes() == b"rendered"
