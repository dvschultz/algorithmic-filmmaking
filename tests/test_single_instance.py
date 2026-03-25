"""Tests for packaged-app single-instance launch protection."""

from pathlib import Path


def test_acquire_single_instance_lock_returns_false_when_already_held(monkeypatch, tmp_path):
    """A duplicate packaged launch should fail to acquire the app lock."""
    import core.single_instance as single_instance

    class FakeLock:
        def __init__(self, path: str):
            self.path = path
            self.stale_lock_time = None
            self.unlocked = False

        def setStaleLockTime(self, stale_lock_time: int):
            self.stale_lock_time = stale_lock_time

        def tryLock(self, _timeout: int) -> bool:
            return False

        def unlock(self):
            self.unlocked = True

    monkeypatch.setattr(single_instance, "QLockFile", FakeLock)
    monkeypatch.setattr(single_instance, "get_app_support_dir", lambda: tmp_path)
    single_instance.release_single_instance_lock()

    assert single_instance.acquire_single_instance_lock() is False


def test_release_single_instance_lock_unlocks_held_lock(monkeypatch, tmp_path):
    """Releasing the app lock should unlock the underlying QLockFile."""
    import core.single_instance as single_instance

    created_locks = []

    class FakeLock:
        def __init__(self, path: str):
            self.path = path
            self.stale_lock_time = None
            self.unlocked = False
            created_locks.append(self)

        def setStaleLockTime(self, stale_lock_time: int):
            self.stale_lock_time = stale_lock_time

        def tryLock(self, _timeout: int) -> bool:
            return True

        def unlock(self):
            self.unlocked = True

    monkeypatch.setattr(single_instance, "QLockFile", FakeLock)
    monkeypatch.setattr(single_instance, "get_app_support_dir", lambda: tmp_path)
    single_instance.release_single_instance_lock()

    assert single_instance.acquire_single_instance_lock() is True
    assert created_locks[0].stale_lock_time == 0
    assert Path(created_locks[0].path).parent == tmp_path

    single_instance.release_single_instance_lock()

    assert created_locks[0].unlocked is True
