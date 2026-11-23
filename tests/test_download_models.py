import hashlib
import io
import zipfile

import pytest

from scripts import download_models as downloader


class Response(io.BytesIO):
    def __init__(self, data, status=206, content_range="bytes 2-5/8"):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Range": content_range}


def test_download_range_requires_correct_offset_and_total(monkeypatch):
    monkeypatch.setattr(downloader.urllib.request, "urlopen", lambda *_args, **_kwargs: Response(b"cdef"))
    assert downloader.fetch_range("https://example.test/model", 2, 5, 8, attempts=1) == b"cdef"
    monkeypatch.setattr(downloader.urllib.request, "urlopen", lambda *_args, **_kwargs: Response(b"abcd", content_range="bytes 0-3/8"))
    with pytest.raises(RuntimeError, match="Wrong Content-Range"):
        downloader.fetch_range("https://example.test/model", 2, 5, 8, attempts=1)


def test_truncated_range_is_rejected(monkeypatch):
    monkeypatch.setattr(downloader.urllib.request, "urlopen", lambda *_args, **_kwargs: Response(b"cd"))
    with pytest.raises(RuntimeError, match="Incomplete range"):
        downloader.fetch_range("https://example.test/model", 2, 5, 8, attempts=1)


def test_ignored_range_is_only_accepted_as_whole_initial_object(monkeypatch):
    monkeypatch.setattr(downloader.urllib.request, "urlopen", lambda *_args, **_kwargs: Response(b"abcdefgh", status=200))
    assert downloader.fetch_range("https://example.test/model", 0, 3, 8, attempts=1) == b"abcdefgh"
    with pytest.raises(RuntimeError, match="ignored a non-zero Range"):
        downloader.fetch_range("https://example.test/model", 2, 5, 8, attempts=1)


def test_checksum_catches_wrong_same_length_existing_model(tmp_path, monkeypatch):
    path = tmp_path / "model.task"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("pose_detector.tflite", b"a" * 100001)
        archive.writestr("pose_landmarks_detector.tflite", b"b" * 100001)
    original = path.read_bytes()
    monkeypatch.setitem(downloader.MODELS, "lite", {"bytes": len(original), "md5": hashlib.md5(original, usedforsecurity=False).hexdigest()})
    assert downloader.validate_model(path, "lite")["bytes"] == len(original)
    corrupt = bytearray(original)
    corrupt[100] ^= 0x01
    path.write_bytes(corrupt)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        downloader.validate_model(path, "lite")


def test_failed_download_never_promotes_invalid_file(tmp_path, monkeypatch):
    path = tmp_path / "model.task"
    monkeypatch.setitem(downloader.MODELS, "lite", {"bytes": 8, "md5": "invalid"})
    monkeypatch.setattr(downloader, "fetch_range", lambda *_args, **_kwargs: b"abcdefgh")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        downloader.download_model(path, "lite")
    assert not path.exists()
    assert not path.with_suffix(".part").exists()
