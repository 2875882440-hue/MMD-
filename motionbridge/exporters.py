"""Portable body skeleton solving and animation exports.

The solver uses NumPy only so the exact same rotations can run in Blender.
MediaPipe landmarks are estimates, not a production mocap rig: the root stays
in place and terminal hand/head twists are inherited rather than invented.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import time
import warnings

import numpy as np


JOINT_NAMES = [
    "Hips", "Spine", "Chest", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
]
PARENTS = [-1, 0, 1, 2, 3, 2, 5, 6, 7, 2, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
COORDINATE_SYSTEM = "Y_UP_RIGHT_HANDED_METERS"


def _length(a: np.ndarray, b: np.ndarray, fallback: float, lo: float, hi: float, confidence=None) -> float:
    lengths = np.linalg.norm(a - b, axis=-1)
    valid = np.isfinite(lengths) & (lengths > 0.005)
    if confidence is not None:
        valid &= np.asarray(confidence) >= .6
    values = lengths[valid]
    return float(np.clip(np.median(values) if len(values) else fallback, lo, hi))


def default_rest_positions() -> np.ndarray:
    """A 1.72 m approximate human T-pose with its hips at the origin."""
    return _rest_positions(.095, .19, .49, .17, [.28, .28], [.25, .25], [.43, .43], [.43, .43], [.17, .17])


def _rest_positions(hip_half, shoulder_half, torso, head, upper_arms, forearms, thighs, shins, feet):
    out = np.zeros((len(JOINT_NAMES), 3), dtype=float)
    out[1] = [0, torso * .33, 0]
    out[2] = [0, torso * .70, 0]
    out[3] = [0, torso + head * .2, 0]
    out[4] = [0, torso + head, 0]
    for side, sign, arm, leg in [(0, 1, 5, 13), (1, -1, 9, 17)]:
        out[arm + 1] = [sign * shoulder_half, torso, 0]
        out[arm] = (out[2] + out[arm + 1]) * .5
        out[arm + 2] = out[arm + 1] + [sign * upper_arms[side], 0, 0]
        out[arm + 3] = out[arm + 2] + [sign * forearms[side], 0, 0]
        out[leg] = [sign * hip_half, 0, 0]
        out[leg + 1] = out[leg] + [0, -thighs[side], 0]
        out[leg + 2] = out[leg + 1] + [0, -shins[side], 0]
        dy = min(.055, feet[side] * .3)
        out[leg + 3] = out[leg + 2] + [0, -dy, math.sqrt(max(feet[side] ** 2 - dy ** 2, 1e-5))]
    return out


def landmarks_to_motion(world_landmarks, image_landmarks, confidences, fps, source, settings, hand_support=None):
    """Convert observations to a fixed-root skeleton, optionally image-aligned.

    Image coordinates constrain camera-relative pose, never scene translation.
    Newly processed videos default to refinement; legacy files keep their solver.
    """
    world = np.asarray(world_landmarks, dtype=float)
    confidence = np.asarray(confidences, dtype=float)
    if world.ndim != 3 or world.shape[1:] != (33, 3) or not len(world):
        raise ValueError("world_landmarks must have shape (frames, 33, 3)")
    if confidence.shape != world.shape[:2]:
        raise ValueError("confidence must have shape (frames, 33)")
    if not np.isfinite(world).all() or not np.isfinite(confidence).all():
        raise ValueError("Motion landmarks must be finite after gap filling")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Motion fps must be positive")
    from .hands import image_observation_weights
    support = np.zeros((len(world), 2)) if hand_support is None else hand_support
    image_weights = image_observation_weights(confidence, support) if settings.get('enhance_motion', False) else confidence

    width = source.get("normalized_width", source.get("width", 0))
    height = source.get("normalized_height", source.get("height", 0))
    refined = bool(settings.get("refine_motion", True) and width > 0 and height > 0)
    refinement = {"enabled": refined, "solver": "direction_v1"}
    if refined:
        from .refinement import align_image_observations
        world, alignment = align_image_observations(world, image_landmarks, image_weights, width, height, fps)
        refinement.update({"solver": "endpoint_ik_v2", "image_alignment": alignment})

    world = world * np.array([1., -1., -1.])
    hip = (world[:, 23] + world[:, 24]) * .5
    world -= hip[:, None, :]
    hip = np.zeros_like(hip)
    shoulder = (world[:, 11] + world[:, 12]) * .5
    head = (world[:, 7] + world[:, 8]) * .5
    frames = np.zeros((len(world), len(JOINT_NAMES), 3))
    conf = np.zeros((len(world), len(JOINT_NAMES)))
    hip_c = np.minimum(confidence[:, 23], confidence[:, 24])
    shoulder_c = np.minimum(confidence[:, 11], confidence[:, 12])
    head_c = np.minimum(confidence[:, 7], confidence[:, 8])
    frames[:, 0] = hip
    frames[:, 1] = shoulder * .33
    frames[:, 2] = shoulder * .70
    frames[:, 3] = shoulder + (head - shoulder) * .2
    frames[:, 4] = head
    conf[:, 0] = hip_c
    conf[:, 1] = conf[:, 2] = np.minimum(hip_c, shoulder_c)
    conf[:, 3] = np.minimum(shoulder_c, head_c)
    conf[:, 4] = head_c
    for arm, leg, landmarks in [(5, 13, (11, 13, 15, 23, 25, 27, 31)), (9, 17, (12, 14, 16, 24, 26, 28, 32))]:
        sho, elbow, wrist, hip_i, knee, ankle, toe = landmarks
        frames[:, arm] = (frames[:, 2] + world[:, sho]) * .5
        conf[:, arm] = np.minimum(conf[:, 2], confidence[:, sho])
        for joint, landmark in [(arm + 1, sho), (arm + 2, elbow), (arm + 3, wrist),
                                (leg, hip_i), (leg + 1, knee), (leg + 2, ankle), (leg + 3, toe)]:
            frames[:, joint] = world[:, landmark]
            conf[:, joint] = confidence[:, landmark]

    def segment(a, b, fallback, lo, hi):
        return _length(world[:, a], world[:, b], fallback, lo, hi,
                       np.minimum(confidence[:, a], confidence[:, b]) if refined else None)

    rest = _rest_positions(
        segment(23, 24, .19, .10, .46) * .5,
        segment(11, 12, .38, .20, .70) * .5,
        _length(shoulder, hip, .49, .25, .8, np.minimum(shoulder_c, hip_c) if refined else None),
        _length(head, shoulder, .17, .08, .38, np.minimum(head_c, shoulder_c) if refined else None),
        [segment(a, b, .28, .12, .52) for a, b in [(11, 13), (12, 14)]],
        [segment(a, b, .25, .10, .48) for a, b in [(13, 15), (14, 16)]],
        [segment(a, b, .43, .20, .75) for a, b in [(23, 25), (24, 26)]],
        [segment(a, b, .43, .18, .72) for a, b in [(25, 27), (26, 28)]],
        [segment(a, b, .17, .07, .34) for a, b in [(27, 31), (28, 32)]],
    )
    motion = {
        "schema_version": "1.0", "fps": float(fps), "num_frames": len(frames),
        "joint_names": JOINT_NAMES.copy(), "parents": PARENTS.copy(),
        "rest_positions": rest.round(7).tolist(),
        "frames": [{"time": round(i / fps, 7), "positions": frame.round(7).tolist(),
                    "confidence": np.clip(conf[i], 0, 1).round(4).tolist()} for i, frame in enumerate(frames)],
        "source": source, "settings": settings,
        "coordinate_system": COORDINATE_SYSTEM, "root_mode": "in_place", "is_demo": False,
        "warnings": ["基础三维为单目估计；根骨固定，未恢复真实场景位移。", "仅身体骨骼，未捕捉精细手指、表情与可靠的肢体轴向扭转。"],
    }
    motion["refinement"] = refinement
    if settings.get('enhance_motion', False):
        for f, frame in enumerate(motion['frames']):
            weights = conf[f].copy()
            weights[[8,12]] = image_weights[f,[15,16]]
            frame['observation_weights'] = weights.round(4).tolist()
    motion["solver"] = refinement["solver"]
    if refined:
        _, _, before_fk = _solve_direction_rotations(motion)
        _, _, solved = solve_rotations(motion)
        endpoints = [8, 12, 15, 19]
        reliable = conf[:, endpoints] >= .6
        before_error = np.linalg.norm(before_fk[:, endpoints] - frames[:, endpoints], axis=-1)
        after_error = np.linalg.norm(solved[:, endpoints] - frames[:, endpoints], axis=-1)
        refinement["endpoint_fit"] = {
            "before_mean_m": float(before_error[reliable].mean()) if reliable.any() else None,
            "after_mean_m": float(after_error[reliable].mean()) if reliable.any() else None,
            "note": "Fit to estimated targets, not physical accuracy. Unreachable targets are clamped without stretching.",
        }
        # The browser and every exporter now consume the same fixed-length pose.
        for frame, positions in zip(motion["frames"], solved):
            frame["positions"] = positions.round(7).tolist()
        motion["positions_are_solved"] = True
        if not refinement["image_alignment"]["applied"]:
            motion["warnings"].append("二维校正缺少可信躯干标定，已回退到三维估计；端点骨骼拟合仍启用。")
        motion["warnings"].append("动作校正使用二维观测和固定骨长拟合；仍可能受遮挡和单目深度误差影响。")
    if settings.get("stabilize_rotations", refined):
        from .stability import stabilize_motion
        motion = stabilize_motion(motion)
    if settings.get("enhance_motion", False) and refined:
        from .projection import refine_projected_arms
        motion = refine_projected_arms(motion, image_landmarks, image_weights, width, height)
        from .contacts import lock_support_feet
        motion = lock_support_feet(motion, image_landmarks, confidence, width, height)
        if settings.get('stabilize_rotations', True):
            from .temporal import finish_motion
            motion = finish_motion(motion)
    elif settings.get("enhance_motion", False):
        motion['foot_contacts']={'enabled':False,'reason':'image_refinement_unavailable'}
        motion['warnings'].append('支撑脚校正需要开启动作校正并有可用图像尺寸；本次未执行脚锁。')
    return motion


def validate_motion(motion):
    names = motion.get("joint_names", [])
    parents = np.asarray(motion.get("parents", []), dtype=int)
    rest = np.asarray(motion.get("rest_positions", []), dtype=float)
    poses = np.asarray([f["positions"] for f in motion.get("frames", [])], dtype=float)
    if len(names) < 2 or len(names) != len(set(names)) or len(parents) != len(names):
        raise ValueError("Motion skeleton is incomplete or has duplicate joint names")
    if parents[0] != -1 or any(p < 0 or p >= i for i, p in enumerate(parents) if i):
        raise ValueError("Skeleton must have one root and topologically ordered parents")
    if rest.shape != (len(names), 3) or poses.ndim != 3 or poses.shape[1:] != rest.shape or not len(poses):
        raise ValueError("Motion frame shape does not match its skeleton")
    if not np.isfinite(rest).all() or not np.isfinite(poses).all():
        raise ValueError("Motion contains non-finite positions")
    if any(np.linalg.norm(rest[i] - rest[p]) < 1e-6 for i, p in enumerate(parents) if i):
        raise ValueError("Skeleton contains a zero-length segment")
    fps = float(motion.get("fps", 0))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Motion frame rate must be positive")
    if motion.get("coordinate_system", COORDINATE_SYSTEM) != COORDINATE_SYSTEM:
        raise ValueError("Unsupported coordinate system")
    return names, parents, rest, poses, fps


def _swing(a, b):
    """Shortest-arc rotation, including parallel and opposite vectors."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    an, bn = np.linalg.norm(a), np.linalg.norm(b)
    if min(an, bn) < 1e-10:
        return np.eye(3)
    a, b = a / an, b / bn
    cosine = float(np.clip(a @ b, -1., 1.))
    if cosine > 1 - 1e-10:
        return np.eye(3)
    if cosine < -1 + 1e-8:
        reference = np.eye(3)[int(np.argmin(np.abs(a)))]
        axis = np.cross(a, reference)
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    v = np.cross(a, b)
    k = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
    return np.eye(3) + k + k @ k / (1 + cosine)


def _solve_direction_rotations(motion):
    """Return (local, world, FK positions), using a constant rest skeleton.

    A joint's rotation maps its outgoing rest offsets to observed directions.
    Branching torso joints use a proper-rotation Kabsch fit. A single child
    constrains swing only; its unconstrained twist is inherited from its parent.
    """
    names, parents, rest, poses, _ = validate_motion(motion)
    children = [[i for i, p in enumerate(parents) if p == j] for j in range(len(names))]
    world = np.zeros((len(poses), len(names), 3, 3))
    local = np.zeros_like(world)
    solved = np.zeros_like(poses)
    for frame_index, pose in enumerate(poses):
        for joint, parent in enumerate(parents):
            inherited = world[frame_index, parent] if parent >= 0 else np.eye(3)
            links = children[joint]
            valid = [c for c in links if np.linalg.norm(pose[c] - pose[joint]) > 1e-7]
            orientation = inherited.copy()
            if valid:
                a = np.array([rest[c] - rest[joint] for c in valid])
                b = np.array([pose[c] - pose[joint] for c in valid])
                a /= np.linalg.norm(a, axis=1)[:, None]
                b /= np.linalg.norm(b, axis=1)[:, None]
                if len(valid) >= 2 and np.linalg.matrix_rank(a, tol=1e-5) >= 2 and np.linalg.matrix_rank(b, tol=1e-5) >= 2:
                    u, _, vt = np.linalg.svd(a.T @ b)
                    correction = np.diag([1., 1., np.linalg.det(vt.T @ u.T)])
                    orientation = vt.T @ correction @ u.T
                else:
                    orientation = _swing(inherited @ a[0], b[0]) @ inherited
            world[frame_index, joint] = orientation
            local[frame_index, joint] = inherited.T @ orientation if parent >= 0 else orientation
            if parent < 0:
                solved[frame_index, joint] = pose[joint]
            else:
                solved[frame_index, joint] = solved[frame_index, parent] + world[frame_index, parent] @ (rest[joint] - rest[parent])
    return local, world, solved


def solve_rotations(motion):
    """Backward-compatible solver; v2 fits reliable limb endpoints with IK."""
    if "world_rotations" in motion:
        names, parents, rest, poses, fps = validate_motion(motion)
        world = np.asarray(motion["world_rotations"], dtype=float)
        if world.shape != (len(poses), len(names), 3, 3) or not np.isfinite(world).all():
            raise ValueError("Invalid baked rotation track shape or nonfinite values")
        if not np.allclose(np.swapaxes(world, -1, -2) @ world, np.eye(3), atol=1e-6) or not np.allclose(np.linalg.det(world), 1., atol=1e-6):
            raise ValueError("Baked tracks must contain proper rotation matrices")
        from .stability import fk_positions
        solved = fk_positions(world, parents, rest, poses[:, 0])
        if not np.allclose(solved, poses, atol=2e-6):
            raise ValueError("Baked rotations and preview positions disagree")
        local = world.copy()
        for joint, parent in enumerate(parents):
            if parent >= 0:
                local[:, joint] = np.swapaxes(world[:, parent], -1, -2) @ world[:, joint]
        return local, world, solved
    initial = _solve_direction_rotations(motion)
    if motion.get("solver") != "endpoint_ik_v2" or motion.get("positions_are_solved", False):
        return initial
    from .refinement import refine_limb_endpoints
    names, _, rest, poses, _ = validate_motion(motion)
    confidence = np.asarray([f.get("observation_weights", f.get("confidence", [1.] * len(names))) for f in motion["frames"]], float)
    if confidence.shape != poses.shape[:2] or not np.isfinite(confidence).all():
        raise ValueError("Invalid solver confidence")
    fitted = refine_limb_endpoints(poses, initial[2], rest, names, np.clip(confidence, 0, 1))
    targets = {**motion, "frames": [{**f, "positions": p} for f, p in zip(motion["frames"], fitted)]}
    return _solve_direction_rotations(targets)


def export_bvh(motion, output_path):
    """Write hierarchy-local Z-X-Y Euler channels; positions/offsets in meters."""
    from scipy.spatial.transform import Rotation

    names, parents, rest, poses, fps = validate_motion(motion)
    local, _, _ = solve_rotations(motion)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # Euler gimbal lock is expected; rotations remain valid.
        angles = Rotation.from_matrix(local.reshape(-1, 3, 3)).as_euler("ZXY", degrees=False).reshape(len(poses), len(names), 3)
    angles = np.rad2deg(np.unwrap(angles, axis=0))
    children = [[i for i, p in enumerate(parents) if p == j] for j in range(len(names))]
    lines, order = ["HIERARCHY"], []

    def emit(joint, depth):
        indent = "  " * depth
        order.append(joint)
        safe_name = "".join(c if c.isalnum() or c in "_:.-" else "_" for c in names[joint])
        lines.append(f"{indent}{'ROOT' if parents[joint] < 0 else 'JOINT'} {safe_name}")
        lines.append(indent + "{")
        offset = rest[joint] if parents[joint] < 0 else rest[joint] - rest[parents[joint]]
        lines.append(indent + "  OFFSET " + " ".join(f"{v:.7f}" for v in offset))
        lines.append(indent + ("  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation" if parents[joint] < 0 else "  CHANNELS 3 Zrotation Xrotation Yrotation"))
        if children[joint]:
            for child in children[joint]:
                emit(child, depth + 1)
        else:
            terminal = rest[joint] - rest[parents[joint]]
            terminal = terminal / max(np.linalg.norm(terminal), 1e-10) * .055
            lines.extend([indent + "  End Site", indent + "  {", indent + "    OFFSET " + " ".join(f"{v:.7f}" for v in terminal), indent + "  }"])
        lines.append(indent + "}")

    emit(0, 0)
    lines.extend(["MOTION", f"Frames: {len(poses)}", f"Frame Time: {1. / fps:.10f}"])
    for i in range(len(poses)):
        # BVH root channels add to the ROOT offset; avoid applying it twice.
        values = list(poses[i, 0] - rest[0])
        for joint in order:
            values.extend(angles[i, joint])
        lines.append(" ".join(f"{float(value):.7f}" for value in values))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_demo_motion(fps=24, seconds=4):
    """Deterministic synthetic arm wave used only to test the software pipeline."""
    fps, seconds = int(fps), float(seconds)
    if not 1 <= fps <= 60 or not 0 < seconds <= 60:
        raise ValueError("Unsupported demo duration or fps")
    rest = default_rest_positions()
    frames = []
    for i in range(round(fps * seconds)):
        t = i / fps
        local = np.tile(np.eye(3), (len(JOINT_NAMES), 1, 1))
        # T-pose shoulders lowered into a relaxed stance, with a visible left wave.
        angle = -.40 + .50 * math.sin(t * 2 * math.pi / seconds)
        for joint, theta in [(6, angle), (10, .95), (7, .5 + .6 * math.sin(t * 4 * math.pi / seconds))]:
            c, s = math.cos(theta), math.sin(theta)
            local[joint] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        # Small upper-body turn reveals three-dimensional export playback.
        theta = .16 * math.sin(t * 2 * math.pi / seconds)
        c, s = math.cos(theta), math.sin(theta)
        local[0] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        positions, orientations = np.zeros_like(rest), np.zeros_like(local)
        for joint, parent in enumerate(PARENTS):
            if parent < 0:
                orientations[joint] = local[joint]
            else:
                orientations[joint] = orientations[parent] @ local[joint]
                positions[joint] = positions[parent] + orientations[parent] @ (rest[joint] - rest[parent])
        frames.append({"time": round(t, 7), "positions": positions.round(7).tolist(), "confidence": [1.] * len(JOINT_NAMES)})
    return {
        "schema_version": "1.0", "fps": fps, "num_frames": len(frames),
        "joint_names": JOINT_NAMES.copy(), "parents": PARENTS.copy(),
        "rest_positions": rest.tolist(), "frames": frames,
        "coordinate_system": COORDINATE_SYSTEM, "root_mode": "in_place", "is_demo": True,
        "source": {"width": 0, "height": 0, "fps": fps, "duration": seconds},
        "settings": {"sample_fps": fps, "synthetic_demo": True},
        "warnings": ["这是程序生成的内置动作演示，用于检查预览与导出，不代表视频动作识别效果。"],
    }


def export_blender(motion_path, output_dir, blender_path, progress=None, cancelled=None):
    """Run Blender in an isolated background process and require round-trip QA."""
    motion_path, output_dir = Path(motion_path).resolve(), Path(output_dir).resolve()
    blender = Path(blender_path)
    if not blender.is_file():
        raise RuntimeError("未找到 Blender，请在配置中选择有效的 blender.exe。")
    if not motion_path.is_file():
        raise RuntimeError("动作文件不存在，无法导出 Blender。")
    output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parents[1] / "scripts" / "blender_export.py"
    logfile = output_dir / "blender_export.log"
    report_path = output_dir / "blender_verification.json"
    report_path.unlink(missing_ok=True)
    command = [str(blender), "--background", "--factory-startup", "--threads", "4", "--python-exit-code", "23", "--python", str(script), "--", str(motion_path), str(output_dir)]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    if progress:
        progress(88, "正在创建骨骼与内置角色，导出 Blender / FBX")
    started = time.monotonic()
    with logfile.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, creationflags=creationflags)
        try:
            while process.poll() is None:
                if cancelled and cancelled():
                    raise InterruptedError("已取消 Blender 导出")
                if time.monotonic() - started > 600:
                    raise RuntimeError("Blender 导出超过 10 分钟，请查看 blender_export.log。")
                time.sleep(.15)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            raise
    if process.returncode != 0 or not report_path.is_file():
        tail = logfile.read_text(encoding="utf-8", errors="replace")[-2500:]
        raise RuntimeError(f"Blender 导出失败（退出码 {process.returncode}）。详情见 blender_export.log。\n{tail}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for filename in ("motion.blend", "motion.fbx"):
        if not (output_dir / filename).is_file() or (output_dir / filename).stat().st_size < 1000:
            raise RuntimeError(f"Blender 未生成有效的 {filename}")
    if not report.get("passed"):
        raise RuntimeError("Blender / FBX 回读验证未通过，请查看 blender_verification.json。")
    if progress:
        progress(98, "Blender / FBX 已生成并通过回读验证")
    return report
