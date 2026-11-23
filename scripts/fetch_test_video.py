"""Download the upstream research sample for local validation, not redistribution."""
from pathlib import Path
import hashlib
import json
import urllib.request

root = Path(__file__).resolve().parents[1]
folder = root / "data" / "test-inputs"
folder.mkdir(parents=True, exist_ok=True)
url = "https://raw.githubusercontent.com/zju3dv/GVHMR/main/docs/example_video/tennis.mp4"
target = folder / "tennis.mp4"
with urllib.request.urlopen(url, timeout=90) as response, target.open("wb") as out:
    while block := response.read(1024 * 1024):
        out.write(block)
record = {"source": url, "bytes": target.stat().st_size, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
          "purpose": "Local pipeline validation only; not a bundled commercial sample."}
(folder / "source.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record), flush=True)
