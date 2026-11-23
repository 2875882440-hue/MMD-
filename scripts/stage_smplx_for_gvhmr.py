"""Validate and stage the user's licensed neutral NPZ, without loading pickle."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MEMBER = "models/smplx/SMPLX_NEUTRAL.npz"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    with zipfile.ZipFile(args.archive) as archive:
        data = archive.read(MEMBER)  # zipfile checks this member's CRC
    with zipfile.ZipFile(io.BytesIO(data)) as inner:
        assert inner.testzip() is None
    expected = {"v_template": (10475, 3), "shapedirs": (10475, 3, 400),
                "posedirs": (10475, 3, 486), "J_regressor": (55, 10475),
                "weights": (10475, 55), "kintree_table": (2, 55), "f": (20908, 3)}
    with np.load(io.BytesIO(data), allow_pickle=False) as model:
        for key, shape in expected.items():
            assert model[key].shape == shape, key
            assert np.isfinite(model[key]).all(), key
    target = ROOT / "research/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"
    digest = hashlib.sha256(data).hexdigest()
    if target.exists():
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest, "Existing model differs; preserving it"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
    report = {"source": str(args.archive.resolve()), "member": MEMBER,
              "destination": str(target), "bytes": len(data), "sha256": digest,
              "source_kind": "user-provided official licensed download",
              "checks": ["member CRC", "inner CRC", "required array shapes", "finite numeric arrays"],
              "release_authenticity": "No signed upstream hash available; not certified",
              "redistribution": False, "inference_tested": False}
    report_path = ROOT / "research/setup/smplx_manifest.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
