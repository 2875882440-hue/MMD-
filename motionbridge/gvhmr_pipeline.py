"""Opt-in local GVHMR engine. Torch runs only in the isolated worker environment."""
from pathlib import Path
import json
import os
import subprocess
import time

import cv2
import numpy as np

from .pipeline import (_video_info, _target_dimensions, _run_ffmpeg, _VideoWriter,
                       _draw_body18, _check_cancelled, _creation_flags, MAX_UPLOAD_BYTES)
from .global_motion import gvhmr_to_motion

ROOT = Path(__file__).resolve().parents[1]
ENGINE = "gvhmr_experimental"
MAX_SECONDS = 32
STAGES = [("preflight", "检查独立 GPU 环境"), ("track", "跟踪人物"),
          ("pose", "识别关节与连续左右身份"), ("features", "提取身体图像特征"),
          ("camera", "估计镜头运动"), ("infer", "解算世界朝向与位移")]
ASSETS = ["gvhmr/gvhmr_siga24_release.ckpt", "hmr2/epoch=10-step=25000.ckpt",
          "vitpose/vitpose-h-multi-coco.pth", "yolo/yolov8x.pt",
          "body_models/smplx/SMPLX_NEUTRAL.npz"]


def engine_status(root=None):
    root = Path(root or ROOT)
    paths = [root / ".venv-gvhmr/Scripts/python.exe", root / "research/GVHMR/hmr4d/__init__.py"]
    paths += [root / "research/GVHMR/inputs/checkpoints" / p for p in ASSETS]
    missing = [str(p.relative_to(root)) for p in paths if not p.is_file()]
    return {"installed": not missing, "missing": missing, "experimental": True,
            "device": "NVIDIA CUDA", "fps": 30, "max_seconds": MAX_SECONDS,
            "runtime_checked": False, "character_retarget": True,
            "note": "仅检查本地文件；提交后检查 CUDA。研究用途许可，动作准确性尚有限制。"}


def effective_settings(settings):
    return {**settings, "engine": ENGINE, "sample_fps": 30, "max_seconds": MAX_SECONDS,
            "root_mode": "estimated_world_translation", "feature_precision": "fp16",
            "keypoint_identity": "temporal_local" if settings.get('world_arm_correction',True) else "temporal",
            "world_arm_correction": bool(settings.get('world_arm_correction',True)),
            "world_foot_contacts": bool(settings.get('world_foot_contacts',True)), "smoothing": 0,
            "world_body_collision": bool(settings.get('world_body_collision',True)),
            "world_head_collision": bool(settings.get('world_head_collision',True)),
            "refine_motion": False, "stabilize_rotations": False, "enhance_motion": False}


def run_worker(command, log_path, cancelled, timeout=1800):
    """One owned child per stage; cancellation terminates it and waits for exit."""
    _check_cancelled(cancelled)
    with Path(log_path).open("wb") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, creationflags=_creation_flags())
        started = time.monotonic()
        try:
            while process.poll() is None:
                _check_cancelled(cancelled)
                if time.monotonic() - started > timeout:
                    raise RuntimeError("GVHMR 阶段超时，已停止该推理进程。")
                time.sleep(.15)
            if process.returncode:
                # Keep detailed worker traces on disk, not arbitrary dependency output in UI.
                raise RuntimeError(f"GVHMR 阶段执行失败，详见 {log_path}；不会回退成其他模型结果。")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def coco_to_body18(kp, width, height):
    result = np.zeros((18, 3), dtype=float)
    indices = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    for i, j in enumerate(indices):
        if j >= 0:
            result[i] = kp[j]
    result[1, :2] = (kp[5, :2] + kp[6, :2]) / 2
    result[1, 2] = min(kp[5, 2], kp[6, 2])
    result[:, :2] /= [width, height]
    return result


def process_video(input_path, output, settings, progress, cancelled):
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not engine_status()["installed"]:
        raise RuntimeError("GVHMR 独立环境或模型未安装完整；请查看引擎状态。")
    settings = effective_settings(settings)
    input_path = Path(input_path)
    if input_path.stat().st_size > MAX_UPLOAD_BYTES:
        raise RuntimeError("视频超过 300 MB。")
    info = _video_info(input_path)
    if info["duration"] and info["duration"] > MAX_SECONDS + 1 / 30:
        raise RuntimeError(f"实验引擎当前最多支持 {MAX_SECONDS} 秒，请先裁剪；不会静默截断。")
    width, height = _target_dimensions(info["width"], info["height"])
    progress(2, "统一为 30 FPS")
    temp_source = output / ".source.processing.mp4"
    _run_ffmpeg(["-threads", "2", "-i", str(input_path), "-map", "0:v:0", "-an",
                 "-vf", f"fps=30,scale={width}:{height}:flags=bicubic,setsar=1",
                 "-frames:v", str(MAX_SECONDS * 30 + 1), "-c:v", "libx264", "-threads", "2",
                 "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags",
                 "+faststart", str(temp_source)], cancelled)
    info = _video_info(temp_source)
    if not 2 <= info["frames"] <= MAX_SECONDS * 30:
        raise RuntimeError("解码帧数超出实验引擎限制（2–960 帧）。")
    temp_source.replace(output / "source.mp4")
    work = output / "gvhmr"
    work.mkdir(exist_ok=True)
    common = [str(ROOT / ".venv-gvhmr/Scripts/python.exe"), "-X", "utf8",
              str(ROOT / "scripts/run_gvhmr_probe.py"), "--video", str(output / "source.mp4"),
              "--output", str(work), "--feature-precision", "fp16", "--keypoint-identity", settings['keypoint_identity']]
    telemetry = []
    cached_outputs={'track':['bbx.pt'],'pose':['vitpose.pt','keypoints_2d.npz','identity_report.json'],
                    'features':['vit_features.pt'],'camera':['slam_results.pt'],'infer':['motion_world.npz']}
    for index, (stage, label) in enumerate(STAGES):
        progress(5 + index * 13, label)
        # Always run preflight: its input/settings hash guard must pass before
        # reusing any successful stage after an explicitly resumed task.
        report_path=work/(stage+'_telemetry.json')
        reuse=False
        if stage!='preflight' and report_path.is_file():
            try:
                previous=json.loads(report_path.read_text(encoding='utf-8'))
                reuse=previous.get('success') is True and all((work/f).is_file() and (work/f).stat().st_size>0 for f in cached_outputs[stage])
            except (ValueError,OSError):pass
        if not reuse:
            run_worker(common + ["--stage", stage], work / (stage + ".log"), cancelled)
        report = json.loads((work / (stage + "_telemetry.json")).read_text(encoding="utf-8"))
        if not report["success"]:
            raise RuntimeError(f"GVHMR 阶段没有成功完成：{stage}")
        telemetry.append({**report, 'reused_from_cache':reuse})
    _check_cancelled(cancelled)
    motion = gvhmr_to_motion(work / "motion_world.npz", source=info)
    motion["settings"] = settings
    motion["warnings"] += ["实验模型仍可能出现手臂跳动、遮挡误判和脚滑；不代表动作已准确还原。",
                           "彩色骨架为二维 ViTPose 观测；三维动画为 GVHMR 解算，两者不等同。",
                           "输出预览已统一为 30 FPS 并移除音轨，原始上传文件保留。"]
    with np.load(work / "keypoints_2d.npz", allow_pickle=False) as data:
        kp = data["keypoints"]
    if kp.shape != (motion["num_frames"], 17, 3) or not np.isfinite(kp).all():
        raise RuntimeError("二维观测与动作帧数不匹配。")
    (output / 'motion_raw.json').write_text(json.dumps(motion,ensure_ascii=False),encoding='utf-8')
    if settings['world_arm_correction']:
        from .world_stability import stabilize_world_upper_body
        motion=stabilize_world_upper_body(motion)
        from .world_reprojection import refine_from_archive
        (output / 'motion_pre_image_alignment.json').write_text(json.dumps(motion,ensure_ascii=False),encoding='utf-8')
        motion=refine_from_archive(motion,work/'motion_world.npz',kp)
    if settings['world_foot_contacts']:
        from .world_contacts import refine_world_contacts
        progress(82,'估计支撑脚并修正缓慢高度漂移')
        motion=refine_world_contacts(motion,kp)
        (output / 'motion_pre_collision.json').write_text(json.dumps(motion,ensure_ascii=False),encoding='utf-8')
        from .floor_collision import constrain_world_floor
        motion=constrain_world_floor(motion)
    if settings['world_body_collision']:
        from .body_collision import constrain_body_collision
        progress(84,'按身体比例修正手臂与躯干穿插')
        (output / 'motion_pre_body_collision.json').write_text(json.dumps(motion,ensure_ascii=False),encoding='utf-8')
        motion=constrain_body_collision(motion)
    if settings['world_head_collision']:
        from .head_collision import constrain_head_collision
        progress(84.5,'按头部尺寸修正手臂与头脸穿插')
        (output / 'motion_pre_head_collision.json').write_text(json.dumps(motion,ensure_ascii=False),encoding='utf-8')
        motion=constrain_head_collision(motion)
    progress(85, "生成彩色骨架与识别对照")
    capture = cv2.VideoCapture(str(output / "source.mp4"))
    writers = []
    body = []
    try:
        writers = [_VideoWriter(output / ".pose.processing.mp4", width, height, 30)]
        writers.append(_VideoWriter(output / ".overlay.processing.mp4", width, height, 30))
        for i in range(len(kp)):
            _check_cancelled(cancelled)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("原片中途解码失败。")
            points = coco_to_body18(kp[i], width, height)
            body.append({"frame": i, "time": i / 30, "pose_keypoints_2d":
                         (points * [width, height, 1]).ravel().tolist()})
            writers[0].write(_draw_body18(np.zeros_like(frame), points[:, :2], points[:, 2]))
            writers[1].write(_draw_body18(frame, points[:, :2], points[:, 2]))
        for writer in writers:
            writer.close(cancelled)
    finally:
        capture.release()
        for writer in writers:
            writer.abort()
    _check_cancelled(cancelled)
    for name in ["pose", "overlay"]:
        (output / f".{name}.processing.mp4").replace(output / f"{name}.mp4")
    quality = {"engine": ENGINE, "accuracy_measured": False, "confidence_3d_available": False,
               "root_translation_range_m": motion["global_motion"]["root_translation_range_m"],
               "identity": json.loads((work / "identity_report.json").read_text()),
               "world_stability": motion.get('world_stability',{'enabled':False}),
               "image_arm_refinement": motion.get('image_arm_refinement',{'enabled':False}),
               "floor_collision": motion.get('floor_collision',{'enabled':False}),
               "body_collision": motion.get('body_collision',{'enabled':False}),
               "head_collision": motion.get('head_collision',{'enabled':False}),
               "world_contact_refinement": motion.get('world_contact_refinement',{'enabled':False}),
               "stages": telemetry, "warnings": motion["warnings"]}
    for name, value in [("quality.json", quality), ("pose_keypoints.json", {"format": "OpenPose BODY18", "frames": body}),
                        ("motion.json", motion)]:
        temp = output / ("." + name + ".tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        temp.replace(output / name)
    progress(100, "世界动作解算完成")
    return {"motion": motion, "warnings": motion["warnings"],
            "meta": {**info, "detected_ratio": None, "model": ENGINE, "device": "NVIDIA CUDA",
                     "root_motion": True, "accuracy_measured": False,
                     "processing_seconds": round(time.monotonic() - started, 3)}}
