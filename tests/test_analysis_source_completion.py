"""Video analysis completion follows the current source binding."""

import pytest

from core.analysis_availability import operation_is_complete_for_clip
from models.clip import Clip
from tests.analysis_fixtures import verify_clip_analysis


@pytest.mark.parametrize("operation,options", [
    ("gaze", {"gaze": True}),
    ("boundary_embeddings", {"boundary": True}),
    ("extract_text", {"ocr": True}),
])
@pytest.mark.parametrize("change", ["path", "fps", "source_id", "missing"])
def test_video_completion_rejects_changed_source(tmp_path, operation, options, change):
    clip = Clip(source_id="source", start_frame=0, end_frame=30)
    source = verify_clip_analysis(clip, tmp_path, **options)
    assert operation_is_complete_for_clip(operation, clip, source=source)
    if change == "path":
        source.file_path = tmp_path / "replacement.mp4"
        source.file_path.write_bytes(b"replacement")
    elif change == "fps":
        source.fps = 24
    elif change == "source_id":
        source.id = "another-source"
    else:
        source = None
    assert not operation_is_complete_for_clip(operation, clip, source=source)
