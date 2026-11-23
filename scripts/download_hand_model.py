"""Install the pinned optional CPU hand model with official object validation."""
from pathlib import Path
import hashlib
import json
import zipfile
from download_models import fetch_range, CHUNK_BYTES

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
SIZE = 7819105
MD5 = '15318430ea3851670fe9914116a9cfad'

def main():
    target = ROOT / 'models/hand_landmarker.task'
    part = target.with_suffix('.part')
    if not target.exists():
        with part.open('wb') as stream:
            for start in range(0, SIZE, CHUNK_BYTES):
                data = fetch_range(URL, start, min(start + CHUNK_BYTES - 1, SIZE - 1), SIZE)
                stream.write(data)
                if len(data) == SIZE:
                    break
        data = part.read_bytes()
    else:
        data = target.read_bytes()
    if len(data) != SIZE or hashlib.md5(data, usedforsecurity=False).hexdigest() != MD5:
        raise RuntimeError('Hand model official checksum/size mismatch')
    with zipfile.ZipFile(target if target.exists() else part) as archive:
        if archive.testzip() is not None or not {'hand_detector.tflite', 'hand_landmarks_detector.tflite'}.issubset(archive.namelist()):
            raise RuntimeError('Invalid hand model task bundle')
    if not target.exists():
        part.replace(target)
    manifest = {'source': URL, 'bytes': SIZE, 'official_md5': MD5,
                'sha256': hashlib.sha256(data).hexdigest(), 'purpose': 'optional wrist observations; not captured finger animation'}
    (ROOT / 'models/hand_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest))

if __name__ == '__main__':
    main()
