"""Download only public weights linked by GVHMR INSTALL.md; never body models."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/runtime/download"))
import gdown

ASSETS = {
    "gvhmr": ("gvhmr_siga24_release.ckpt", "1c9iCeKFN4Kr6cMPJ9Ss6Jdc3SZFnO5NP", 163508011),
    "hmr2": ("epoch=10-step=25000.ckpt", "1X5hvVqvqI9tvjUCb2oAlZxtgIKD9kvsc", 2709494041),
    "vitpose": ("vitpose-h-multi-coco.pth", "1sR8xZD9wrZczdDVo6zKscNLwvarIRhP5", 2549075546),
    "yolo": ("yolov8x.pt", "1_HGm-lqIH83-M1ML4bAXaqhm_eT2FKo5", 136867539),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("asset", choices=ASSETS)
    parser.add_argument("--source", choices=["official", "hf-mirror"], default="official")
    parser.add_argument("--workers", type=int, choices=[1, 2, 3, 4], default=1)
    args = parser.parse_args()
    name, file_id, size = ASSETS[args.asset]
    target = ROOT / "research/GVHMR/inputs/checkpoints" / args.asset / name
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = next(x for x in json.loads((ROOT / "research/setup/hf-mirror-tree.json").read_text())
                    if x["path"] == f"{args.asset}/{name}") if args.source == "hf-mirror" else None
    url = f"https://drive.google.com/file/d/{file_id}/view"
    if metadata:
        url = "https://huggingface.co/camenduru/GVHMR/resolve/21b32d5/" + quote(f"{args.asset}/{name}")
    if not target.exists():
        partial = target.with_suffix(target.suffix + ".part")
        print(f"Downloading public weight {args.asset} from {args.source}, expected {size} bytes", flush=True)
        if metadata:
            import requests
            # Bound each transfer; long CDN connections repeatedly disconnected on this host.
            def fetch_chunk(bounds):
                start, end = bounds
                for attempt in range(4):
                    try:
                        response = requests.get(url + f"?download=true&mb_chunk={start}",
                                                headers={"Range": f"bytes={start}-{end}"}, timeout=(20, 45))
                        response.raise_for_status()
                        expected_range = f"bytes {start}-{end}/{size}"
                        if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                            raise RuntimeError("Server did not honor requested byte range")
                        if len(response.content) != end - start + 1:
                            raise RuntimeError("Incomplete chunk")
                        return response.content
                    except Exception:
                        if attempt == 3:
                            raise
                        time.sleep(2 * (attempt + 1))
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                while (start := partial.stat().st_size if partial.exists() else 0) < size:
                    bounds = [(offset, min(offset + 8 * 1024 * 1024, size)-1)
                              for offset in range(start, min(start + args.workers*8*1024*1024, size), 8*1024*1024)]
                    # Append only complete contiguous chunks. At most workers chunks live in RAM.
                    for (offset, end), data in zip(bounds, pool.map(fetch_chunk, bounds)):
                        current = partial.stat().st_size if partial.exists() else 0
                        if current != offset:
                            raise RuntimeError("Another process modified this partial download")
                        with partial.open("ab") as stream:
                            stream.write(data)
                        print(f"{args.asset}: {end + 1}/{size} bytes", flush=True)
        else:
            gdown.download(id=file_id, output=str(partial), resume=True, quiet=True, use_cookies=False)
        if partial.stat().st_size != size:
            raise RuntimeError(f"Size mismatch: {partial.stat().st_size} != {size}; kept partial")
        if metadata:
            actual = hashlib.sha256()
            with partial.open("rb") as stream:
                while chunk := stream.read(4 * 1024 * 1024):
                    actual.update(chunk)
            if actual.hexdigest() != metadata["lfs"]["oid"]:
                raise RuntimeError("Mirror LFS SHA256 mismatch; kept partial")
        partial.rename(target)
    if target.stat().st_size != size:
        raise RuntimeError("Existing target has unexpected size; preserving it")
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    if metadata and digest.hexdigest() != metadata["lfs"]["oid"]:
        raise RuntimeError("Existing weight does not match the expected LFS hash; preserved")
    manifest = ROOT / f"research/setup/{args.asset}_weight_manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous.get("sha256") == digest.hexdigest():
            print(json.dumps(previous), flush=True)
            return  # Preserve original provenance on repeated verification.
    record = {"asset": args.asset, "filename": name, "bytes": size, "sha256": digest.hexdigest(),
              "url": url, "source": args.source,
              "official_file_url": f"https://drive.google.com/file/d/{file_id}/view",
              "mirror_lfs_sha256_checked": bool(metadata),
              "source_note": "Public community mirror fallback after official Drive quota error; size matches official listing. Not an upstream signature." if metadata else "GVHMR official INSTALL.md linked Drive folder",
              "license": "Corresponding upstream research licenses apply", "loaded": False}
    manifest.write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record), flush=True)

if __name__ == "__main__":
    main()
