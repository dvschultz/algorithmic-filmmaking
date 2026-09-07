"""Thumbnail computation must be detached and cancellation-safe."""

from pathlib import Path
from threading import Event

import pytest

from core.project import Project
from models.clip import Clip, Source


@pytest.fixture
def media(tmp_path, monkeypatch):
    from types import SimpleNamespace

    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    project = Project.new()
    source = Source(id="source", file_path=video, fps=30)
    clip = Clip(id="clip", source_id=source.id, start_frame=0, end_frame=30)
    project.add_source(source)
    project.add_clips([clip])
    monkeypatch.setattr(
        "core.settings.load_settings",
        lambda: SimpleNamespace(thumbnail_cache_dir=tmp_path),
    )
    return project, source, clip


def fake_thumbnail(self, **kwargs):
    path = kwargs.get("output_path") or self.cache_dir / "test.jpg"
    path.write_bytes(b"thumbnail")
    return path


def test_worker_does_not_mutate_live_clips(media, tmp_path, monkeypatch):
    from ui.main_window import ThumbnailWorker

    project, source, clip = media
    monkeypatch.setattr(
        "core.thumbnail.ThumbnailGenerator.generate_clip_thumbnail", fake_thumbnail
    )
    worker = ThumbnailWorker(source, [clip], tmp_path)
    worker.start()
    assert worker.wait(10000)
    assert clip.thumbnail_path is None, "worker thread changed the live clip"


def test_cancellation_during_generation_discards_result(media, monkeypatch):
    from core.spine.thumbnails import generate_thumbnails

    project, _, clip = media
    cancel = Event()

    def generate(self, **kwargs):
        path = fake_thumbnail(self, **kwargs)
        cancel.set()
        return path

    monkeypatch.setattr(
        "core.thumbnail.ThumbnailGenerator.generate_clip_thumbnail", generate
    )
    result = generate_thumbnails(project, cancel_event=cancel)
    assert clip.thumbnail_path is None
    assert not result["result"]["succeeded"]


def test_cache_tracks_source_range_dimensions_and_force(media, tmp_path, monkeypatch):
    from core.operations.thumbnails import (
        ThumbnailOptions,
        ThumbnailTask,
        run_thumbnails,
    )

    _, source, clip = media
    calls = []

    def generate(self, **kwargs):
        calls.append(kwargs)
        return fake_thumbnail(self, **kwargs)

    monkeypatch.setattr(
        "core.thumbnail.ThumbnailGenerator.generate_clip_thumbnail", generate
    )

    def run(width=320, force=False):
        return run_thumbnails(
            (ThumbnailTask.capture(clip, source),),
            ThumbnailOptions(tmp_path, width, 180, force),
            Event(),
        )[0]

    first = run()
    assert run().path == first.path and len(calls) == 1
    clip.start_frame = 5
    second = run()
    assert second.path != first.path
    resized = run(width=640)
    assert resized.path != second.path
    source.file_path.write_bytes(b"changed video")
    changed = run()
    assert changed.path != second.path
    forced = run(force=True)
    assert forced.path != changed.path
    assert len(calls) == 5
    assert Path(first.path).read_bytes() == b"thumbnail"
    assert not list(tmp_path.glob(".thumbnail-*"))


@pytest.mark.parametrize(
    "mode",
    [
        "success",
        "cancel",
        "project",
        "session",
        "clip",
        "range",
        "fps",
        "media",
        "prior",
        "reply",
        "restart",
        "duplicate",
        "close",
        "stale_dispatch",
    ],
)
def test_gui_thumbnail_publication_owns_results(tmp_path, mode):
    import os
    import subprocess
    import sys

    code = r"""
import sys,time,threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
from PySide6.QtCore import QObject,QCoreApplication
from core.project import Project
from models.clip import Clip,Source
from ui.workers.thumbnail_worker import ThumbnailWorker
from ui.workers.thumbnail_delivery import ThumbnailDelivery
app=QCoreApplication([])
directory=Path(sys.argv[1]); mode=sys.argv[2]
video=directory/'video.mp4'; video.write_bytes(b'video')
source=Source(id='source',file_path=video,fps=30)
clip=Clip(id='clip',source_id=source.id,start_frame=0,end_frame=30)
project=Project.new(); project.add_source(source); project.add_clips([clip])
assert project.save(directory/'project.json')
before=project.path.read_bytes()
window=QObject(); window.project=project
window._dispatch_gui_reply=SimpleNamespace(is_current=lambda _: True)
entered=threading.Event(); release=threading.Event(); main_thread=threading.get_ident()
ready=[]; completed=[]; observer_threads=[]
project.add_observer(lambda *args: observer_threads.append(threading.get_ident()))
def generate(self, **kwargs):
    entered.set(); assert release.wait(10)
    path=kwargs['output_path']; path.write_bytes(b'thumbnail'); return path
def launch():
    worker=ThumbnailWorker(source,[clip],directory,project=project)
    window.thumbnail_worker=worker
    delivery=ThumbnailDelivery(window,worker,ready=lambda *args: ready.append(args),completed=lambda: completed.append(worker))
    worker.start()
    return worker
with patch('core.thumbnail.ThumbnailGenerator.generate_clip_thumbnail',generate):
    if mode=='stale_dispatch': project.replace_source_clips(source.id,[Clip(id='clip',source_id=source.id,start_frame=0,end_frame=30)])
    worker=launch(); assert entered.wait(5)
    assert clip.thumbnail_path is None
    if mode=='cancel': worker.cancel()
    elif mode=='close':
        from ui.main_window import MainWindow
        window._check_unsaved_changes=lambda: True
        window.status_bar=Mock()
        event=Mock()
        MainWindow.closeEvent(window,event)
        event.ignore.assert_called_once()
    elif mode=='project': window.project=Project.new()
    elif mode=='session': project.clear()
    elif mode=='clip': project.replace_source_clips(source.id,[Clip(id='clip',source_id=source.id,start_frame=0,end_frame=30)])
    elif mode=='range': clip.start_frame=5
    elif mode=='fps': source.fps=60
    elif mode=='media': video.write_bytes(b'changed')
    elif mode=='prior': clip.thumbnail_path=directory/'manual.jpg'; clip.thumbnail_path.write_bytes(b'manual')
    elif mode=='reply': window._dispatch_gui_reply.is_current=lambda _:False
    elif mode=='restart': replacement=launch()
    elif mode=='duplicate': worker.finished.emit(); assert not completed
    release.set()
    deadline=time.monotonic()+15
    while window._active_thumbnail_workers and time.monotonic()<deadline:
        app.processEvents(); time.sleep(.005)
    assert not window._active_thumbnail_workers, 'native worker did not settle'
    assert project.path is None or (directory/'project.json').read_bytes()==before, 'implicit project save'
    assert all(thread==main_thread for thread in observer_threads)
    if mode in ('success','duplicate','restart'):
        assert clip.thumbnail_path and clip.thumbnail_path.is_file()
        assert len(ready)==1 and len(completed)==1
        assert project.is_dirty
    else:
        assert not ready
        if mode=='prior': assert clip.thumbnail_path.read_bytes()==b'manual'
        else: assert clip.thumbnail_path is None
        if mode in ('clip','stale_dispatch'): assert project.clips_by_id['clip'].thumbnail_path is None
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), mode],
        capture_output=True,
        text=True,
        timeout=40,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_thumbnail_output_dimensions(tmp_path):
    import shutil
    import subprocess
    from PIL import Image
    from core.operations.thumbnails import (
        ThumbnailOptions,
        ThumbnailTask,
        run_thumbnails,
    )

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("FFmpeg is not installed")
    video = tmp_path / "real.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=10:duration=2",
            "-c:v",
            "mpeg4",
            str(video),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    source = Source(id="source", file_path=video, fps=10)
    clip = Clip(id="clip", source_id=source.id, start_frame=0, end_frame=20)
    result = run_thumbnails(
        (ThumbnailTask.capture(clip, source),),
        ThumbnailOptions(tmp_path, 320, 180),
        Event(),
    )[0]
    assert result.status == "succeeded", result
    with Image.open(result.path) as image:
        assert image.size == (320, 180)


def test_pending_queue_rejects_same_id_replacement(media, tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock, patch
    from ui.main_window import MainWindow

    project, source, old = media
    replacement = Clip(id=old.id, source_id=source.id, start_frame=0, end_frame=30)
    project.replace_source_clips(source.id, [replacement])
    window = SimpleNamespace(
        project=project,
        clips_by_id=project.clips_by_id,
        sources_by_id=project.sources_by_id,
        _pending_thumbnail_clips=[old],
        settings=SimpleNamespace(thumbnail_cache_dir=tmp_path),
        thumbnail_worker=None,
        _on_thumbnail_ready=Mock(),
        _on_agent_thumbnails_finished=Mock(),
    )
    with (
        patch("ui.main_window.ThumbnailWorker") as factory,
        patch("ui.main_window.ThumbnailDelivery"),
    ):
        MainWindow._on_agent_thumbnails_finished(window)
        factory.assert_not_called()
    assert not window._pending_thumbnail_clips


@pytest.mark.parametrize("fps", [None, "invalid", 0, float("nan")])
def test_invalid_source_fps_does_not_abort_other_thumbnails(
    media, tmp_path, monkeypatch, fps
):
    from core.operations.thumbnails import (
        ThumbnailTask,
        ThumbnailOptions,
        run_thumbnails,
    )

    _, source, clip = media
    good = ThumbnailTask.capture(clip, source)
    source.fps = fps
    bad_clip = Clip(id="bad", source_id=source.id, start_frame=0, end_frame=30)
    bad = ThumbnailTask.capture(bad_clip, source)
    monkeypatch.setattr(
        "core.thumbnail.ThumbnailGenerator.generate_clip_thumbnail", fake_thumbnail
    )
    outcomes = run_thumbnails((bad, good), ThumbnailOptions(tmp_path), Event())
    assert outcomes[0].status == "failed" and outcomes[0].code == "invalid_time_range"
    assert outcomes[1].status == "succeeded"


def test_worker_reports_nonfinite_fps_as_an_outcome(media, tmp_path):
    from ui.workers.thumbnail_worker import ThumbnailWorker

    project, source, clip = media
    source.fps = float("nan")
    worker = ThumbnailWorker(source, [clip], tmp_path, project=project)
    worker.start()
    assert worker.wait(10000)
    assert worker.result[0].code == "invalid_time_range"
