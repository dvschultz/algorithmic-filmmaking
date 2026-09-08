"""Combined analysis replies and transcription queues share run ownership."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("mode", ["success", "cancel", "expired", "path"])
def test_pipeline_request_and_thread_lifetimes(tmp_path, mode):
    code = r"""
import sys,time,threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from PySide6.QtCore import QObject,QCoreApplication
from core.project import Project
from core.settings import Settings
from core.operations.transcription import TranscriptionOutcome
from models.clip import Source,Clip
from ui.main_window import MainWindow
from ui.workers.clip_analysis import ClipAnalysisController
from ui.workers.transcription_worker import TranscriptionWorker
from ui.workers.gui_tool_reply import GuiToolReply
from ui.workers.gui_tool_mailbox import GuiToolMailbox
app=QCoreApplication([]); directory=Path(sys.argv[1]); mode=sys.argv[2]
window=QObject(); window.project=project=Project.new(); window.settings=Settings()
window.settings.transcription_backend='faster-whisper'
for i in range(2):
    source=Source(file_path=directory/f'{i}.mp4',fps=30)
    source.file_path.write_bytes(b'video'); project.add_source(source)
    project.add_clips([Clip(source_id=source.id,start_frame=0,end_frame=30)])
mailbox=GuiToolMailbox()
window._chat_worker=SimpleNamespace(_stop_requested=False,
    is_gui_tool_pending=mailbox.is_pending,set_gui_tool_result=mailbox.submit)
reply=GuiToolReply.capture(window,'analyze_all_live',mailbox.begin('analyze_all_live'))
window._dispatch_gui_reply=reply
window._pending_agent_tool_call_id='unrelated'
window._gui_state=Mock(); window.analyze_tab=Mock(); window.progress_bar=Mock()
window.status_bar=Mock(); window.collect_tab=Mock(); window._update_chat_project_state=Mock()
window._build_agent_analysis_result=lambda clips,ops,message,extra: dict(extra,clip_ids=[c.id for c in clips])
reports=[]; started=[]; entered=threading.Event(); release=threading.Event()
def run(self):
    started.append(tuple(t.clip_id for t in self.tasks))
    self.status.emit('Preparing transcription model')
    self.result=tuple(TranscriptionOutcome(t.clip_id,'succeeded',()) for t in self.tasks)
    self.transcription_completed.emit()
    if len(started)==1:
        entered.set(); assert release.wait(10)
controller=ClipAnalysisController(window,project.clips,['transcribe'])
if hasattr(controller, 'status'):
    controller.status.connect(lambda owner,message: MainWindow._on_clip_analysis_status(window,owner,message))
controller.completed.connect(lambda owner,result: reports.append(result))
controller.completed.connect(lambda owner,result: MainWindow._on_clip_analysis_finished(window,owner,result))
with patch.object(TranscriptionWorker,'run',run):
    controller.start(); first=controller.workers['transcribe']; assert entered.wait(5)
    try:
        for _ in range(10): app.processEvents(); time.sleep(.002)
        assert len(started)==1 and not reports
        window.status_bar.showMessage.assert_any_call('Preparing transcription model')
        assert all(c.transcript is None for c in project.clips)
        if mode=='cancel': controller.cancel()
        if mode=='expired': assert mailbox.wait(0) is None
        if mode=='path': project.path=directory/'other.json'
    finally:
        release.set(); assert first.wait(5000)
    deadline=time.monotonic()+5
    while not reports and time.monotonic()<deadline:
        app.processEvents(); time.sleep(.002)
    assert len(reports)==1 and not controller.workers
    window.analyze_tab.set_analyzing.assert_called_once_with(False)
    if mode=='success':
        assert len(started)==2 and all(c.transcript==[] for c in project.clips)
        assert all(s.has_analysis for s in project.sources)
        result=mailbox.wait(0)
        assert result['tool_call_id']==reply.token and result['name']==reply.name
        assert result['result']['success'] is True
        assert result['result']['clip_ids']==[c.id for c in project.clips]
        assert result['result']['shot_type_summary']=={}
        assert result['result']['transcribed_count']==0
        assert result['result']['clip_count']==2
    else:
        assert len(started)==1 and all(c.transcript is None for c in project.clips)
        assert reports[0]['cancelled']
        if mode=='expired': assert mailbox.wait(0) is None
    assert window._pending_agent_tool_call_id=='unrelated'
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), mode],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
