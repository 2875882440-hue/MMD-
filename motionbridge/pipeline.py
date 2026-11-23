"""CPU-only video pose extraction with bounded decoding and honest quality metadata.

The JSON motion is an approximate, hips-relative body pose.  Missing observations
are interpolated for animation continuity, while their confidence remains low.
This module never downloads models and never starts GPU work.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

import cv2
import imageio_ffmpeg
import numpy as np
from scipy.ndimage import gaussian_filter1d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
MAX_SECONDS = 60.0
MAX_UPLOAD_BYTES = 300 * 1024 * 1024
MAX_DIMENSION = 1280
CONFIDENCE_THRESHOLD = 0.25
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".wmv", ".mpg", ".mpeg", ".ts"}

# OpenPose BODY-18 ordering.  Neck (index 1) is constructed from both shoulders.
BODY18_MP = (0, -1, 12, 14, 16, 11, 13, 15, 24, 26, 28, 23, 25, 27, 5, 2, 8, 7)
BODY18_LIMBS = (
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
)
BODY18_RGB = (
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170), (255, 0, 85),
)


def _check_cancelled(cancelled: Callable[[], bool]) -> None:
    if cancelled():
        raise InterruptedError("用户已取消任务")


def _creation_flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def _target_dimensions(width: int, height: int) -> tuple[int, int]:
    scale = min(1.0, MAX_DIMENSION / max(width, height))
    # H.264 yuv420p requires even dimensions.  Never stretch to a square.
    return max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2)


def _video_info(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError("无法读取视频。请使用正常播放的 MP4、MOV、AVI 或 WebM 文件。")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0 or width > 8192 or height > 8192:
            raise RuntimeError("视频尺寸无效或超过首版支持范围（单边最多 8192 像素）。")
        if not math.isfinite(fps) or not 1 <= fps <= 240:
            raise RuntimeError("无法确定有效的视频帧率（首版支持 1–240 fps 输入）。")
        # The decoded shape accounts for rotation metadata where OpenCV supports it.
        success, frame = capture.read()
        if not success or frame is None:
            raise RuntimeError("视频无法解码，文件可能损坏或使用了不支持的编码。")
        height, width = frame.shape[:2]
        duration = frames / fps if frames > 0 else None
        return {"width": width, "height": height, "fps": fps, "frames": frames, "duration": duration}
    finally:
        capture.release()


def _run_ffmpeg(args: list[str], cancelled: Callable[[], bool]) -> None:
    """Run a bounded FFmpeg task; stderr cannot block the child process."""
    _check_cancelled(cancelled)
    with tempfile.TemporaryFile() as error_file:
        process = subprocess.Popen(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=error_file,
            creationflags=_creation_flags(),
        )
        try:
            while process.poll() is None:
                _check_cancelled(cancelled)
                time.sleep(0.1)
            if process.returncode:
                error_file.seek(0)
                details = error_file.read(6000).decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"视频转换失败：{details[-3500:]}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)


class _VideoWriter:
    """Incremental RGB/BGR -> browser-compatible H.264, with explicit teardown."""

    def __init__(self, path: Path, width: int, height: int, fps: float):
        self.path = path
        self.error_file = tempfile.TemporaryFile()
        self.process = subprocess.Popen(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
             "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2", "-preset", "veryfast",
             "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.error_file,
            creationflags=_creation_flags(),
        )
        self.closed = False

    def write(self, frame: np.ndarray) -> None:
        try:
            self.process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        except (BrokenPipeError, OSError) as exc:
            self.error_file.seek(0)
            details = self.error_file.read(3500).decode("utf-8", errors="replace")
            raise RuntimeError(f"无法写入预览视频：{details}") from exc

    def close(self, cancelled: Callable[[], bool] = lambda: False) -> None:
        if self.closed:
            return
        try:
            if self.process.stdin and not self.process.stdin.closed:
                self.process.stdin.close()
            deadline = time.monotonic() + 45
            while self.process.poll() is None:
                _check_cancelled(cancelled)
                if time.monotonic() > deadline:
                    raise RuntimeError("预览视频编码超时。")
                time.sleep(0.05)
            if self.process.returncode:
                self.error_file.seek(0)
                raise RuntimeError("预览视频编码失败：" + self.error_file.read(3500).decode("utf-8", errors="replace"))
        finally:
            self.abort()

    def abort(self) -> None:
        if self.closed:
            return
        if self.process.poll() is None:
            self.process.kill()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
        if self.process.stdin and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        self.error_file.close()
        self.closed = True


def _longest_false_run(mask: np.ndarray) -> int:
    longest = current = 0
    for present in mask:
        current = 0 if present else current + 1
        longest = max(longest, current)
    return longest


def _fill_and_smooth(points: np.ndarray, confidence: np.ndarray, smoothing: float, fps: float, preserve_details=False) -> tuple[np.ndarray, list[int]]:
    """Fill coordinates, not confidence.  Long gaps/low-confidence joints remain visible in QA."""
    points = np.asarray(points, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 3 or confidence.shape != points.shape[:2]:
        raise ValueError("Coordinate/confidence shape mismatch")
    result = np.empty_like(points)
    timeline = np.arange(len(points))
    uncertain_joints: list[int] = []
    for joint in range(points.shape[1]):
        finite = np.isfinite(points[:, joint]).all(axis=1)
        valid = finite & (confidence[:, joint] >= CONFIDENCE_THRESHOLD)
        if not np.any(valid):
            # Retain the model's uncertain estimate only when no confident sample exists.
            # Confidence is NEVER raised by this fallback.
            valid = finite
            uncertain_joints.append(joint)
        if not np.any(valid):
            raise RuntimeError("缺少完整的人体关键点，无法建立可用骨骼。请使用全身清晰入镜的视频。")
        for axis in range(3):
            result[:, joint, axis] = np.interp(timeline, timeline[valid], points[valid, joint, axis])
    sigma_frames = max(0.0, float(smoothing)) * float(fps) * 0.10
    if sigma_frames > 0.01 and len(points) > 2:
        filtered = gaussian_filter1d(result, sigma=sigma_frames, axis=0, mode="nearest")
        if preserve_details:
            velocity = np.diff(result[..., :2], axis=0)
            norms = np.linalg.norm(velocity, axis=-1)
            coherence = np.sum(velocity[1:] * velocity[:-1], axis=-1) / np.maximum(norms[1:] * norms[:-1], 1e-10)
            reliable = (confidence[1:-1] >= .7) & (confidence[:-2] >= .6) & (confidence[2:] >= .6)
            keep = np.zeros(confidence.shape)
            keep[1:-1] = .8 * reliable * (coherence > .4) * (np.minimum(norms[1:], norms[:-1]) * fps > .06)
            result = filtered * (1 - keep[..., None]) + result * keep[..., None]
        else:
            result = filtered
    return result, uncertain_joints


def _body18(image_points: np.ndarray, confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.zeros((18, 2), dtype=np.float64)
    scores = np.zeros(18, dtype=np.float64)
    for index, source in enumerate(BODY18_MP):
        if source >= 0:
            points[index] = image_points[source, :2]
            scores[index] = confidence[source]
    points[1] = (image_points[11, :2] + image_points[12, :2]) * 0.5
    scores[1] = min(confidence[11], confidence[12])
    return points, scores


def _draw_body18(canvas: np.ndarray, points: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """Standard BODY18 joint/limb colors on black are usable as pose conditioning."""
    height, width = canvas.shape[:2]
    image = canvas.copy()
    pixel = np.rint(points * np.array([width, height])).astype(np.int64)
    # Skip off-screen detections; do not invent edge-clamped body parts.
    visible = (confidence >= CONFIDENCE_THRESHOLD) & np.isfinite(points).all(axis=1)
    visible &= (points[:, 0] >= 0) & (points[:, 0] < 1) & (points[:, 1] >= 0) & (points[:, 1] < 1)
    thickness = max(2, round(min(width, height) / 180))
    radius = max(3, round(min(width, height) / 135))
    layer = image.copy()
    for index, (a, b) in enumerate(BODY18_LIMBS):
        if visible[a] and visible[b]:
            color = BODY18_RGB[index][::-1]
            cv2.line(layer, tuple(pixel[a]), tuple(pixel[b]), color, thickness * 2, cv2.LINE_AA)
    image = cv2.addWeighted(image, 0.4, layer, 0.6, 0)
    for index in range(18):
        if visible[index]:
            cv2.circle(image, tuple(pixel[index]), radius, BODY18_RGB[index][::-1], -1, cv2.LINE_AA)
    return image


def _save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")


def process_video(
    input_path: Path,
    output_dir: Path,
    settings: dict,
    progress: Callable[[float, str], None],
    cancelled: Callable[[], bool],
) -> dict:
    """Extract one person's body motion and previews, raising on unusable inputs."""
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    _check_cancelled(cancelled)
    if not input_path.is_file():
        raise RuntimeError("找不到上传的视频文件。")
    if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise RuntimeError("不支持这个文件格式。请上传 MP4、MOV、AVI、MKV 或 WebM 视频。")
    if input_path.stat().st_size > MAX_UPLOAD_BYTES:
        raise RuntimeError("视频超过首版的 300 MB 上传上限。")
    try:
        fps = int(settings.get("sample_fps", 24))
        smoothing = float(settings.get("smoothing", 0.55))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("采样帧率或平滑强度无效。") from exc
    if not 12 <= fps <= 30 or not math.isfinite(smoothing) or not 0 <= smoothing <= 1:
        raise RuntimeError("采样帧率须为 12–30，平滑强度须为 0–1。")
    model = settings.get("model", "full")
    if model not in {"lite", "full"}:
        raise RuntimeError("不支持的姿态模型。")
    model_path = MODEL_DIR / f"pose_landmarker_{model}.task"
    if not model_path.is_file():
        raise RuntimeError(f"缺少 {model} 姿态模型，请先运行项目安装脚本。")

    progress(2, "检查视频与模型")
    original = _video_info(input_path)
    if original["duration"] is not None and original["duration"] > MAX_SECONDS + 0.02:
        raise RuntimeError(f"视频约 {original['duration']:.1f} 秒，超过首版 60 秒上限。请先剪辑。")
    if original["frames"] == 1:
        raise RuntimeError("文件只有一帧。请上传包含动作的视频。")
    width, height = _target_dimensions(original["width"], original["height"])
    max_frames = int(MAX_SECONDS * fps)
    temporary = {
        "source.mp4": output_dir / ".source.processing.mp4",
        "pose.mp4": output_dir / ".pose.processing.mp4",
        "overlay.mp4": output_dir / ".overlay.processing.mp4",
        "motion.json": output_dir / ".motion.processing.json",
        "quality.json": output_dir / ".quality.processing.json",
        "pose_keypoints.json": output_dir / ".pose_keypoints.processing.json",
        "observations.npz": output_dir / ".observations.processing.npz",
    }
    writers: list[_VideoWriter] = []
    capture = None
    hand_observer = None
    try:
        progress(4, "统一帧率与视频尺寸")
        # Decode one frame over the limit, so a misleading container duration
        # cannot silently truncate a long clip and report success.
        _run_ffmpeg([
            "-threads", "2", "-i", str(input_path), "-map", "0:v:0", "-an",
            "-vf", f"fps={fps},scale={width}:{height}:flags=bicubic,setsar=1",
            "-frames:v", str(max_frames + 1), "-c:v", "libx264", "-threads", "2",
            "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(temporary["source.mp4"]),
        ], cancelled)
        normalized = _video_info(temporary["source.mp4"])
        if normalized["frames"] > max_frames:
            raise RuntimeError("解码后的视频超过 60 秒上限。请先剪辑，软件不会静默截断动作。")
        if normalized["frames"] < 2:
            raise RuntimeError("视频太短，至少需要两个有效画面帧。")
        width, height = normalized["width"], normalized["height"]
        expected_frames = normalized["frames"]

        # Import lazily so simple format/quality helper tests need no ML runtime.
        import mediapipe as mp

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path), delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        world_frames, image_frames, confidence_frames = [], [], []
        hand_candidates = []
        hand_status = {"enabled":False,"reason":"disabled"}
        if settings.get("enhance_motion",False):
            hand_path = MODEL_DIR / 'hand_landmarker.task'
            if hand_path.is_file():
                from .hands import HandObserver
                hand_observer = HandObserver(hand_path)
                hand_status = {"enabled":True}
            else:
                hand_status = {"enabled":False,"reason":"hand_model_missing"}
        detected_frames, usable_frames = [], []
        capture = cv2.VideoCapture(str(temporary["source.mp4"]))
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
            frame_index = 0
            while True:
                _check_cancelled(cancelled)
                success, frame = capture.read()
                if not success:
                    break
                if frame_index >= max_frames:
                    raise RuntimeError("解码帧数超过 60 秒上限。")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                    round(frame_index * 1000 / fps),
                )
                world = np.full((33, 3), np.nan)
                image = np.full((33, 3), np.nan)
                confidence = np.zeros(33)
                detected = usable = False
                if result.pose_landmarks and result.pose_world_landmarks:
                    image = np.array([[p.x, p.y, p.z] for p in result.pose_landmarks[0]], dtype=np.float64)
                    world = np.array([[p.x, p.y, p.z] for p in result.pose_world_landmarks[0]], dtype=np.float64)
                    confidence = np.array([
                        min(float(p.visibility if p.visibility is not None else 0),
                            float(p.presence if p.presence is not None else 0))
                        for p in result.pose_landmarks[0]
                    ], dtype=np.float64)
                    confidence = np.nan_to_num(np.clip(confidence, 0, 1), nan=0, posinf=0, neginf=0)
                    detected = bool(image.shape == (33, 3) and world.shape == (33, 3)
                                    and np.isfinite(image).all() and np.isfinite(world).all())
                    if detected:
                        core_confidence = confidence[[11, 12, 23, 24]]
                        usable = bool(np.mean(core_confidence) >= 0.4 and np.count_nonzero(core_confidence >= 0.3) >= 3)
                    else:
                        image = np.full((33, 3), np.nan)
                        world = np.full((33, 3), np.nan)
                        confidence = np.zeros(33)
                world_frames.append(world)
                hand_candidates.append(hand_observer.observe(rgb,image,confidence) if hand_observer and detected else [])
                image_frames.append(image)
                confidence_frames.append(confidence)
                detected_frames.append(detected)
                usable_frames.append(usable)
                frame_index += 1
                if frame_index % 5 == 0 or frame_index == expected_frames:
                    progress(10 + 65 * min(1, frame_index / expected_frames), f"CPU 提取姿态 · {frame_index}/{expected_frames} 帧")
        capture.release()
        capture = None
        total_frames = len(world_frames)
        if total_frames < 2:
            raise RuntimeError("视频解码未得到足够的画面。")
        if total_frames != expected_frames:
            raise RuntimeError("视频解码不完整，未输出截断的动作。请重新编码原视频后再试。")
        usable = np.asarray(usable_frames, dtype=bool)
        detected = np.asarray(detected_frames, dtype=bool)
        if np.count_nonzero(usable) < max(2, math.ceil(total_frames * 0.05)):
            raise RuntimeError("没有检测到足够清晰的人体姿态。请使用单人全身入镜、光线充足、遮挡较少的视频。")

        progress(77, "插值与平滑（保留原始置信度）")
        confidence = np.stack(confidence_frames)
        raw_image = np.stack(image_frames)
        body_image = raw_image.copy()
        hand_support = np.zeros((total_frames,2))
        if hand_observer:
            from .hands import associate_wrists
            raw_image, hand_support, hand_status = associate_wrists(raw_image, confidence, hand_candidates, fps, width, height)
            hand_observer.close()
            hand_observer = None
        world, uncertain_joints = _fill_and_smooth(np.stack(world_frames), confidence, smoothing, fps)
        from .hands import image_observation_weights
        image_weights = image_observation_weights(confidence, hand_support)
        image_points, _ = _fill_and_smooth(raw_image, image_weights, smoothing, fps,
                                          preserve_details=settings.get('enhance_motion', False))
        longest_gap = _longest_false_run(usable)
        detected_ratio = float(np.mean(usable))
        warnings = [
            "单人身体姿态估计；不恢复可靠的场景位移，也不包含精细手指和表情。开启接地时可能调整骨盆高度。",
            "输出预览视频已统一帧率并移除音轨；原始上传视频保留在本地任务目录。",
            "彩色骨架采用 OpenPose BODY-18 身体布局，不含手部和面部详细关键点；需匹配支持该布局的 AI 工作流。",
        ]
        if detected_ratio < 0.9:
            warnings.append(f"仅 {detected_ratio:.1%} 的帧具有较清晰躯干姿态；缺失/低置信度坐标通过插值补齐，需检查动作。")
        if longest_gap:
            warnings.append(f"最长连续低可信区间约 {longest_gap / fps:.2f} 秒；插值区间不能视为真实捕捉结果。")
        if uncertain_joints:
            warnings.append(f"{len(uncertain_joints)} 个关键点整段缺少可信观测，使用了低置信度估计。")
        feet_visibility = float(np.mean(confidence[:, [27, 28, 31, 32]] >= CONFIDENCE_THRESHOLD))
        if feet_visibility < 0.8:
            warnings.append("脚部经常不可见或置信度较低，腿部/脚部动画可能不准确。")

        source = {
            **original,
            "duration": original["duration"] if original["duration"] is not None else total_frames / fps,
            "normalized_width": width,
            "normalized_height": height,
            "normalized_fps": fps,
            "normalized_duration": total_frames / fps,
        }
        from .exporters import landmarks_to_motion

        progress(79, "校正三维姿态与拟合固定骨长")
        np.savez_compressed(temporary["observations.npz"], world=world, image=image_points, confidence=confidence,
                            hand_support=hand_support, image_weights=image_weights,
                            raw_body_image=body_image, raw_world=np.stack(world_frames), assisted_image=raw_image)
        motion = landmarks_to_motion(world, image_points, confidence, fps, source, dict(settings), hand_support=hand_support)
        motion['hand_assistance'] = hand_status
        if settings.get('enhance_motion',False):
            motion['warnings'].append('手部辅助只校正可识别的手腕；遮挡或身份不明确时跳过，不含手指动画。' if hand_status['enabled'] else '手部模型未安装，本次手腕辅助未启用；运行 scripts/download_hand_model.py 后重新处理。')
            if hand_status['enabled']:
                motion['warnings'].append(f"手部辅助在 {hand_status['candidate_frames']}/{total_frames} 帧发现候选，关联 {hand_status['associated_wrist_samples']} 个手腕观测；这是覆盖统计，不是动作准确率。")
        motion["warnings"] = list(dict.fromkeys([*motion.get("warnings", []), *warnings]))
        motion["is_demo"] = False
        motion["quality"] = {
            "detected_ratio": detected_ratio,
            "raw_detected_ratio": float(np.mean(detected)),
            "longest_low_confidence_seconds": longest_gap / fps,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
        }
        quality = {
            "schema_version": "1.0", "engine": "MediaPipe PoseLandmarker", "device": "CPU", "model": model,
            "frames": total_frames, "fps": fps, "duration": total_frames / fps,
            "usable_frames": int(np.count_nonzero(usable)), "detected_frames": int(np.count_nonzero(detected)),
            **motion["quality"], "feet_visible_ratio": feet_visibility,
            "uncertain_landmarks": uncertain_joints,
            "per_frame": [{"frame": i, "time": round(i / fps, 6), "detected": bool(detected[i]),
                           "usable": bool(usable[i]), "mean_confidence": round(float(np.mean(confidence[i])), 5)}
                          for i in range(total_frames)],
            "warnings": motion["warnings"],
            "method": "Coordinates are interpolated and Gaussian-smoothed; observation confidence is unchanged.",
            "refinement": motion.get("refinement", {}),
            "rotation_stabilization": motion.get("rotation_stabilization", {"enabled": False}),
            "hand_assistance": hand_status,
            "projection_refinement": motion.get('projection_refinement', {'enabled':False}),
            "temporal_finish": motion.get('temporal_finish', {'enabled':False}),
            "foot_contacts": {k:v for k,v in motion.get('foot_contacts',{}).items() if k!='weights'},
        }
        _save_json(temporary["motion.json"], motion)

        progress(81, "生成彩色骨架与对照视频")
        pose_writer = _VideoWriter(temporary["pose.mp4"], width, height, fps)
        writers.append(pose_writer)
        overlay_writer = _VideoWriter(temporary["overlay.mp4"], width, height, fps)
        writers.append(overlay_writer)
        capture = cv2.VideoCapture(str(temporary["source.mp4"]))
        keypoints_frames = []
        for frame_index in range(total_frames):
            _check_cancelled(cancelled)
            success, frame = capture.read()
            if not success:
                raise RuntimeError("预览生成时视频解码中断。")
            body_points, body_confidence = _body18(image_points[frame_index], confidence[frame_index])
            pose_writer.write(_draw_body18(np.zeros_like(frame), body_points, body_confidence))
            overlay_writer.write(_draw_body18(frame, body_points, body_confidence))
            packed = []
            for point, score in zip(body_points, body_confidence):
                packed.extend([round(float(point[0] * width), 3), round(float(point[1] * height), 3), round(float(score), 5)])
            keypoints_frames.append({"frame_index": frame_index, "time": round(frame_index / fps, 6),
                                     "people": [{"pose_keypoints_2d": packed}] if detected[frame_index] else []})
            if frame_index % 10 == 0:
                progress(81 + 16 * (frame_index + 1) / total_frames, f"编码预览 · {frame_index + 1}/{total_frames} 帧")
        capture.release()
        capture = None
        for writer in writers:
            writer.close(cancelled)
        _save_json(temporary["pose_keypoints.json"], {
            "schema_version": "1.0", "layout": "OpenPose BODY_18", "width": width, "height": height,
            "fps": fps, "coordinate_units": "pixels", "frames": keypoints_frames,
            "note": "Coordinates are smoothed/interpolated; confidence retains original observation quality. No detailed hand or face landmarks.",
        })
        quality["processing_seconds"] = round(time.monotonic() - started, 3)
        _save_json(temporary["quality.json"], quality)
        _check_cancelled(cancelled)
        for filename, path in temporary.items():
            path.replace(output_dir / filename)
        progress(98, "姿态与预览生成完成")
        return {
            "motion": motion,
            "meta": {"fps": fps, "frames": total_frames, "duration": total_frames / fps,
                     "detected_ratio": detected_ratio, "width": width, "height": height,
                     "processing_seconds": quality["processing_seconds"], "model": model, "device": "CPU"},
            "warnings": motion["warnings"],
        }
    finally:
        if hand_observer is not None:
            hand_observer.close()
        if capture is not None:
            capture.release()
        for writer in writers:
            writer.abort()
        # Only explicit temporary files directly within this job output directory.
        for path in temporary.values():
            if path.resolve().parent == output_dir and path.is_file():
                path.unlink(missing_ok=True)
