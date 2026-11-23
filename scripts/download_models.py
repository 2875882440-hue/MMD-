"""Fetch pinned official MediaPipe bundles with verified ranges and checksums.

Some HTTP proxies cache partial responses.  A successful HTTP status or a file
larger than 1 MB is therefore not sufficient validation.  Each range, total byte
count, official object checksum, and ZIP member CRC is checked before promotion.
"""
from pathlib import Path
import hashlib
import json
import re
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CHUNK_BYTES = 1024 * 1024
# Version 1 objects. MD5 values and sizes are the official GCS ETag and
# x-goog-stored-content-length metadata, independently inspected on 2026-10-22.
# MD5 is used for transport integrity, never for password/security operations.
MODELS = {
    "lite": {"bytes": 5777746, "md5": "04a75ddf7c811ac7a1a4523266dd7d88",
             "sha256": "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"},
    "full": {"bytes": 9398198, "md5": "83879689d373d143be094c972355e48e",
             "sha256": "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1"},
}


def source_url(variant):
    filename = f"pose_landmarker_{variant}.task"
    return f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{variant}/float16/1/{filename}"


def validate_model(path, variant):
    """Validate existing files too; never trust their names or approximate sizes."""
    expected = MODELS[variant]
    path = Path(path)
    if path.stat().st_size != expected["bytes"]:
        raise RuntimeError(f"{path.name}: incomplete model size ({path.stat().st_size}/{expected['bytes']})")
    data = path.read_bytes()
    md5 = hashlib.md5(data, usedforsecurity=False).hexdigest()
    sha256 = hashlib.sha256(data).hexdigest()
    if md5 != expected["md5"] or (expected.get("sha256") and sha256 != expected["sha256"]):
        raise RuntimeError(f"{path.name}: official model checksum mismatch")
    with zipfile.ZipFile(path) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError(f"{path.name}: damaged ZIP member")
        required = {"pose_detector.tflite", "pose_landmarks_detector.tflite"}
        if not required.issubset(bundle.namelist()):
            raise RuntimeError(f"{path.name}: incomplete MediaPipe task bundle")
        if any(bundle.getinfo(member).file_size < 100_000 for member in required):
            raise RuntimeError(f"{path.name}: unexpected empty model member")
    return {"source": source_url(variant), "bytes": len(data), "sha256": sha256,
            "official_md5": expected["md5"], "validation": "exact size, official object checksum, SHA-256 recorded, ZIP CRC and members"}


def fetch_range(url, start, end, total, attempts=3):
    """Return only a correctly addressed range; reject stale/wrong proxy data."""
    last_error = None
    for attempt in range(attempts):
        # Fresh retry URL bypasses caches that incorrectly ignore Range headers.
        request_url = url if attempt == 0 else f"{url}?motionbridge_retry={start}-{time.time_ns()}"
        request = urllib.request.Request(request_url, headers={
            "Range": f"bytes={start}-{end}", "Accept-Encoding": "identity", "User-Agent": "MotionBridge/0.1.0",
        })
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status == 200:
                    # A server may legitimately ignore Range. Accept a whole
                    # object only on the first chunk, with exact length checking.
                    if start != 0:
                        raise RuntimeError("Server ignored a non-zero Range request")
                    data = response.read(total + 1)
                    if len(data) != total:
                        raise RuntimeError(f"Incomplete full response: {len(data)}/{total}")
                    return data
                if response.status != 206:
                    raise RuntimeError(f"Unexpected HTTP status {response.status}")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if not match or tuple(map(int, match.groups())) != (start, end, total):
                    raise RuntimeError(f"Wrong Content-Range: {response.headers.get('Content-Range')}")
                expected_length = end - start + 1
                data = response.read(expected_length + 1)
                if len(data) != expected_length:
                    raise RuntimeError(f"Incomplete range: {len(data)}/{expected_length}")
                return data
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.3 * (attempt + 1))
    raise RuntimeError(f"Failed model range {start}-{end}: {last_error}") from last_error


def download_model(target, variant):
    target = Path(target)
    url = source_url(variant)
    total = MODELS[variant]["bytes"]
    part = target.with_suffix(".part")
    try:
        with part.open("wb") as output:
            offset = 0
            while offset < total:
                end = min(offset + CHUNK_BYTES, total) - 1
                data = fetch_range(url, offset, end, total)
                output.write(data)
                offset += len(data)
                print(f"  {target.name}: {offset}/{total} bytes", flush=True)
        metadata = validate_model(part, variant)
        part.replace(target)
        return metadata
    finally:
        part.unlink(missing_ok=True)


def main():
    folder = ROOT / "models"
    folder.mkdir(exist_ok=True)
    manifest = {}
    for variant in ("lite", "full"):
        filename = f"pose_landmarker_{variant}.task"
        target = folder / filename
        try:
            manifest[filename] = validate_model(target, variant)
            print("Verified existing", filename, flush=True)
        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
            if target.exists():
                print("Existing model is invalid; downloading a verified replacement:", exc, flush=True)
            print("Downloading", filename, flush=True)
            manifest[filename] = download_model(target, variant)
        print(filename, manifest[filename]["sha256"], flush=True)
    temporary_manifest = folder / "manifest.json.part"
    temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary_manifest.replace(folder / "manifest.json")


if __name__ == "__main__":
    main()
