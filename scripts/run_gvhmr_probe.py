"""Isolated, resumable GVHMR inference. No compiled renderer or production fallback.

Run each stage in a fresh .venv-gvhmr process so model memory is released.
The exported NPZ preserves upstream SMPL-X world orientation and translation.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "research/GVHMR"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["preflight", "track", "pose", "features", "camera", "infer"])
    parser.add_argument("--video", type=Path, default=ROOT / "research/global-motion-probe/dance_12_18s.mp4")
    parser.add_argument("--output", type=Path, default=ROOT / "research/global-motion-probe/gvhmr-windows")
    parser.add_argument("--static-camera", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--feature-precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--keypoint-identity", choices=["raw", "temporal", "temporal_local"], default="raw")
    args = parser.parse_args()
    video = str(args.video.resolve())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(ROOT))
    os.environ["MOTIONBRIDGE_GVHMR_BATCH_SIZE"] = str(args.batch_size)
    os.environ["MOTIONBRIDGE_GVHMR_FEATURE_PRECISION"] = args.feature_precision
    os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "research/runtime/yolo-config")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import torch
    import numpy as np
    import cv2
    torch.set_num_threads(4)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(3)), int(cap.get(4))
    cap.release()
    if length < 2 or abs(fps - 30) > .01:
        raise ValueError("Research runner currently requires an explicitly prepared 30 fps clip")
    config = {"video_sha256": hashlib.sha256(Path(video).read_bytes()).hexdigest(),
              "fps": fps, "frames": length, "width": width, "height": height,
              "static_camera": args.static_camera, "batch_size": args.batch_size,
              "feature_precision": args.feature_precision,
              "keypoint_identity": args.keypoint_identity,
              "source_commit": "ee960bb6e2ea2d381aa97f08e9b71ef320b624b1"}
    manifest = output / "run_config.json"
    if manifest.exists():
        previous_config = json.loads(manifest.read_text())
        previous_config.setdefault("keypoint_identity", "raw")
        if previous_config != config:
            raise ValueError("Output belongs to different input/settings. Use a new output directory.")
    manifest.write_text(json.dumps(config, indent=2), encoding="utf-8")
    telemetry = {"stage": args.stage, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "success": False, "torch": torch.__version__, "python": sys.version,
                 "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                 "gpu_samples": []}
    stop = threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                line = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.free,temperature.gpu",
                                                "--format=csv,noheader,nounits"], text=True,
                                               creationflags=0x08000000, timeout=8).strip()
                telemetry["gpu_samples"].append({"elapsed": round(time.monotonic()-started, 2), "used_free_temp": line})
            except Exception as exc:
                telemetry["monitor_error"] = str(exc)
            stop.wait(2)
    started = time.monotonic()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    def save(value, name):
        tmp = output / (name + ".tmp")
        torch.save(value, tmp)
        tmp.replace(output / name)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable in isolated environment")
        torch.cuda.reset_peak_memory_stats()
        from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K
        from hmr4d.utils.geo_transform import compute_cam_angvel
        # Register inference components only; training callbacks import a renderer.
        import hmr4d.model.gvhmr.gvhmr_pl_demo
        import hmr4d.model.gvhmr.utils.endecoder
        import hmr4d.network.gvhmr.relative_transformer
        from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SimpleVO
        from hydra import initialize_config_module, compose
        from hydra.utils import instantiate
        if args.stage == "preflight":
            from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
            v = torch.tensor([[0.0, 1.2, 0.0]], device="cuda")
            assert torch.allclose(v, matrix_to_axis_angle(axis_angle_to_matrix(v)), atol=1e-5)
            telemetry["transform_cuda_smoke_test"] = True
        elif args.stage == "track":
            tracker = Tracker()
            xyxy = tracker.get_one_track(video).float()
            assert xyxy.shape == (length, 4) and torch.isfinite(xyxy).all()
            save({"bbx_xyxy": xyxy, "bbx_xys": get_bbx_xys_from_xyxy(xyxy, base_enlarge=1.2).float()}, "bbx.pt")
        elif args.stage in ("pose", "features"):
            bbx = torch.load(output / "bbx.pt", map_location="cpu")["bbx_xys"]
            if args.stage == "pose":
                extractor = VitPoseExtractor()
                result = extractor.extract(video, bbx)
                if args.keypoint_identity in ("temporal", "temporal_local"):
                    from motionbridge.body_identity import stabilize_coco_identity
                    save(result, "vitpose_raw.pt")
                    corrected, identity_report = stabilize_coco_identity(result.numpy(), bbx.numpy())
                    result = torch.from_numpy(corrected).float()
                    if args.keypoint_identity == "temporal_local":
                        from motionbridge.body_identity import stabilize_arm_observations
                        corrected, arm_report = stabilize_arm_observations(corrected, bbx.numpy(), fps)
                        result = torch.from_numpy(corrected).float()
                        identity_report['arms'] = arm_report
                    (output / "identity_report.json").write_text(json.dumps(identity_report, indent=2), encoding="utf-8")
                name = "vitpose.pt"
            else:
                extractor = Extractor()
                result = extractor.extract_video_features(video, bbx)
                name = "vit_features.pt"
            assert result.shape[0] == length and torch.isfinite(result).all()
            save(result, name)
            if args.stage == "pose":
                np.savez_compressed(output / "keypoints_2d.npz", keypoints=result.cpu().numpy())
            telemetry["output_shape"] = list(result.shape)
        elif args.stage == "camera":
            if args.static_camera:
                result = np.repeat(np.eye(4)[None], length, axis=0)
            else:
                result = SimpleVO(video, scale=.5, step=8, method="sift").compute()
            assert result.shape == (length, 4, 4) and np.isfinite(result).all()
            save(result, "slam_results.pt")
        else:
            with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
                cfg = compose(config_name="demo", overrides=["video_name=motionbridge_probe"])
            model = instantiate(cfg.model, _recursive_=False)
            state = torch.load(REPO / cfg.ckpt_path, map_location="cpu")["state_dict"]
            missing, extra = model.load_state_dict(state, strict=False)
            telemetry["checkpoint_missing_keys"] = missing
            telemetry["checkpoint_unexpected_keys"] = extra
            if missing:
                raise RuntimeError(f"Missing inference weights: {missing}")
            del state
            gc.collect()
            model = model.eval().cuda()
            traj = torch.load(output / "slam_results.pt", map_location="cpu")
            data = {"length": torch.tensor(length),
                    "bbx_xys": torch.load(output / "bbx.pt")["bbx_xys"],
                    "kp2d": torch.load(output / "vitpose.pt"),
                    "K_fullimg": estimate_K(width, height).repeat(length, 1, 1),
                    "cam_angvel": compute_cam_angvel(torch.from_numpy(traj[:, :3, :3])),
                    "f_imgseq": torch.load(output / "vit_features.pt")}
            with torch.no_grad():
                pred = model.predict(data, static_cam=args.static_camera)
                params = pred["smpl_params_global"]
                joints = model.pipeline.endecoder.fk_v2(**{k: v[None] for k, v in params.items()})[0]
                camera_joints = model.pipeline.endecoder.fk_v2(
                    **{k: v[None] for k, v in pred["smpl_params_incam"].items()})[0]
                skeleton = model.pipeline.endecoder.smplx_model.get_skeleton(params["betas"])[..., :22, :]
            arrays = {"world_" + k: v.detach().cpu().numpy() for k, v in params.items()}
            arrays.update({"camera_" + k: v.detach().cpu().numpy() for k, v in pred["smpl_params_incam"].items()})
            arrays.update(joints_world=joints.detach().cpu().numpy(),
                          joints_camera=camera_joints.detach().cpu().numpy(), rest_joints=skeleton.detach().cpu().numpy(),
                          parents=model.pipeline.endecoder.smplx_model.parents[:22].cpu().numpy(),
                          fps=np.array(fps), K=data["K_fullimg"].numpy())
            if not all(np.isfinite(v).all() for v in arrays.values()):
                raise RuntimeError("Non-finite model output")
            np.savez_compressed(output / "motion_world.npz", **arrays)
            telemetry["world_translation_range_m"] = np.ptp(arrays["world_transl"], axis=0).tolist()
            telemetry["coordinate_system"] = "Upstream GVHMR world: right-handed, Y-up; no front-facing correction or in-place root reset"
        torch.cuda.synchronize()
        telemetry["success"] = True
    except Exception:
        telemetry["error"] = traceback.format_exc()
        raise
    finally:
        stop.set()
        thread.join(timeout=10)
        telemetry["elapsed_seconds"] = round(time.monotonic()-started, 3)
        telemetry["peak_torch_allocated_mib"] = torch.cuda.max_memory_allocated()/2**20
        telemetry["peak_torch_reserved_mib"] = torch.cuda.max_memory_reserved()/2**20
        (output / f"{args.stage}_telemetry.json").write_text(json.dumps(telemetry, indent=2), encoding="utf-8")
        print(json.dumps({k:v for k,v in telemetry.items() if k != "gpu_samples"}, indent=2), flush=True)

if __name__ == "__main__":
    main()
