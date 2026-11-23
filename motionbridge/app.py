"""Local HTTP application. Jobs and artifacts stay inside this project."""
from __future__ import annotations

import copy
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import urlparse
import uuid
import zipfile

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("MOTIONBRIDGE_DATA", str(ROOT / "data" / "physics-jobs")))
if not DATA.resolve().is_relative_to(ROOT.resolve()):
    raise RuntimeError("Physics开发副本禁止使用工程外的任务目录。")
DATA.mkdir(parents=True, exist_ok=True)
(ROOT / "logs").mkdir(exist_ok=True)
MAX_UPLOAD = 300 * 1024 * 1024
MAX_QUEUE = 4
OUTPUT_NAMES = {
    "source.mp4": "处理后原片", "pose.mp4": "彩色骨架视频", "overlay.mp4": "识别叠加预览",
    "motion.json": "关节动作数据", "quality.json": "识别质量报告", "motion.bvh": "BVH 骨骼动画",
    "motion_raw.json": "GVHMR 模型输出（去抖与接地前）",
    "motion_pre_collision.json": "脚底碰撞约束前动作（已去抖与接地）",
    "motion_pre_body_collision.json": "手臂避碰前动作（保留已有脚底修正）",
    "motion_pre_head_collision.json": "头脸避碰前动作（保留已有躯干与脚底修正）",
    "motion_pre_image_alignment.json": "二维肘腕对齐前动作（已去抖）",
    "pose_keypoints.json": "OpenPose BODY18 关键点",
    "observations.npz": "姿态观测缓存（校正前）",
    "motion.blend": "Blender 角色工程", "motion.fbx": "FBX 角色动画", "mannequin.mp4": "立体人偶视频",
    "blender_verification.json": "Blender 导出验证", "blender_export.log": "Blender 日志",
    "Nahida_MotionBridge_Dance.blend": "纳西妲动作替换工程",
    "Rin_MotionBridge_Dance.blend": "远坂凛动作替换工程",
    "character_preview.png": "目标角色静帧预览", "retarget_report.json": "角色映射与处理记录",
    "saved_project_audit.json": "角色全帧验证", "character_export.log": "角色处理日志",
    "character_audit.log": "角色验证日志",
}
log = logging.getLogger("motionbridge")
if not log.handlers:
    handler = RotatingFileHandler(ROOT / "logs" / "application.log", maxBytes=3_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)

app = FastAPI(title="MotionBridge", version=__version__, docs_url="/api/docs", redoc_url=None)
lock = threading.RLock()
pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="motionbridge")
jobs: dict[str, dict] = {}
cancel_events: dict[str, threading.Event] = {}


def now() -> str:
    # Windows Python installations may not ship an IANA zone database.
    from datetime import timedelta, timezone
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def find_blender() -> str | None:
    config_file = ROOT / "config.json"
    configured = None
    if config_file.exists():
        try:
            configured = json.loads(config_file.read_text(encoding="utf-8-sig")).get("blender_path")
        except (OSError, ValueError):
            pass
    candidates = [os.environ.get("MOTIONBRIDGE_BLENDER"), configured, shutil.which("blender"),
                  r"D:\STEAM\steamapps\common\Blender\blender.exe",
                  r"D:\AI Analysis Knowledge Base\.tools\blender-4.5.0-windows-x64\blender.exe"]
    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    candidates += [str(p) for p in (program_files / "Blender Foundation").glob("Blender*/blender.exe")]
    return next((str(p) for p in candidates if p and Path(p).is_file()), None)


def persist(job: dict) -> None:
    directory = DATA / job["id"]
    directory.mkdir(parents=True, exist_ok=True)
    temp = directory / "job.json.tmp"
    temp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(directory / "job.json")


for saved in DATA.glob("*/job.json"):
    try:
        record = json.loads(saved.read_text(encoding="utf-8"))
        if record.get("status") in {"queued", "processing"}:
            record.update(status="failed", stage="上次运行中断", error="应用在任务完成前关闭，请重新导入视频。", updated_at=now())
            persist(record)
        jobs[record["id"]] = record
    except (OSError, ValueError, KeyError):
        log.exception("Cannot restore job %s", saved)


@app.middleware("http")
async def local_only(request: Request, call_next):
    host = request.headers.get("host", "").split(":")[0].lower()
    if host not in {"localhost", "127.0.0.1", "[", "testserver"}:
        return JSONResponse({"detail": "此应用仅允许本机访问。"}, status_code=403)
    origin = request.headers.get("origin")
    if origin and urlparse(origin).hostname not in {"127.0.0.1", "localhost", "::1"}:
        return JSONResponse({"detail": "拒绝外部网页调用本地任务接口。"}, status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


def get_job(job_id: str) -> dict:
    with lock:
        if job_id not in jobs:
            raise HTTPException(404, "任务不存在。")
        return copy.deepcopy(jobs[job_id])


def update(job_id: str, **changes) -> None:
    with lock:
        jobs[job_id].update(changes, updated_at=now())
        persist(jobs[job_id])


def output_files(job_id: str) -> list[dict]:
    folder = DATA / job_id / "outputs"
    return [{"name": name, "label": label, "size": (folder / name).stat().st_size,
             "url": f"/api/jobs/{job_id}/files/{name}"}
            for name, label in OUTPUT_NAMES.items() if (folder / name).is_file() and (folder / name).stat().st_size > 0]


def new_job(name: str, settings: dict, is_demo: bool = False) -> dict:
    with lock:
        if sum(j["status"] in {"queued", "processing"} for j in jobs.values()) >= MAX_QUEUE:
            raise HTTPException(429, "队列已满，请等待当前任务完成。")
        job_id = uuid.uuid4().hex
        job = {"id": job_id, "name": name[:180], "status": "queued", "stage": "排队等待", "progress": 0,
               "created_at": now(), "updated_at": now(), "error": None, "warnings": [], "settings": settings,
               "meta": {}, "files": [], "is_demo": is_demo, "software_version": __version__}
        jobs[job_id] = job
        cancel_events[job_id] = threading.Event()
        (DATA / job_id / "outputs").mkdir(parents=True)
        persist(job)
        return copy.deepcopy(job)


def run_job(job_id: str, input_path: Path | None = None, export_only: bool = False) -> None:
    event = cancel_events[job_id]
    output = DATA / job_id / "outputs"
    job = get_job(job_id)

    def progress(percent, stage):
        if event.is_set():
            raise InterruptedError("任务已取消")
        # Reserve the last stage for artifacts and optional Blender export.
        update(job_id, progress=round(max(0, min(float(percent), 99)), 1), stage=stage)

    try:
        if event.is_set():
            raise InterruptedError("任务已取消")
        update(job_id, status="processing", stage="准备处理", error=None)
        from .exporters import export_bvh, export_blender
        if not export_only:
            if job["is_demo"]:
                from .exporters import create_demo_motion
                motion = create_demo_motion()
                motion["is_demo"] = True
                if not any("演示" in w for w in motion.setdefault("warnings", [])):
                    motion["warnings"].append("这是程序生成的内置动作，仅用于预览与导出演示，不代表视频识别效果。")
                result = {"motion": motion, "meta": {"fps": motion["fps"], "frames": motion["num_frames"],
                    "duration": motion["num_frames"] / motion["fps"], "detected_ratio": None}, "warnings": motion["warnings"]}
                (output / "motion.json").write_text(json.dumps(motion, ensure_ascii=False), encoding="utf-8")
                progress(65, "生成内置演示动作")
            else:
                if job["settings"].get("engine") == "gvhmr_experimental":
                    from .gvhmr_pipeline import process_video
                else:
                    from .pipeline import process_video
                result = process_video(input_path, output, job["settings"], lambda p, s: progress(p * .78, s), event.is_set)
            export_bvh(result["motion"], output / "motion.bvh")
            update(job_id, meta=result["meta"], warnings=result.get("warnings", []), files=output_files(job_id))
            progress(82, "骨骼动画已导出")
        if export_only or job["settings"].get("exports_3d", True):
            blender = find_blender()
            if blender:
                try:
                    progress(85, "生成 Blender 角色与 FBX")
                    export_blender(output / "motion.json", output, blender,
                                   progress=lambda p, s: progress(85 + min(float(p), 100) * .13, s), cancelled=event.is_set)
                except InterruptedError:
                    raise
                except Exception as exc:
                    if export_only:
                        raise
                    log.exception("Blender optional export failed: %s", job_id)
                    warnings = get_job(job_id)["warnings"] + [f"基础动捕已完成；Blender 导出失败，可重试：{str(exc)[:300]}"]
                    update(job_id, warnings=warnings)
            else:
                warnings = get_job(job_id)["warnings"] + ["未找到 Blender；已保留 BVH 和动作数据。安装或配置 Blender 后可补导出。"]
                update(job_id, warnings=warnings)
        if event.is_set():
            raise InterruptedError("任务已取消")
        update(job_id, status="completed", stage="处理完成", progress=100, files=output_files(job_id))
        log.info("Job completed id=%s demo=%s meta=%s", job_id, job["is_demo"], get_job(job_id)["meta"])
    except InterruptedError:
        update(job_id, status="cancelled", stage="已取消", files=output_files(job_id))
        log.info("Job cancelled id=%s", job_id)
    except Exception as exc:
        log.exception("Job failed id=%s", job_id)
        update(job_id, status="failed", stage="处理失败", error=str(exc)[:1200], files=output_files(job_id))


@app.get("/api/health")
def health():
    from .gvhmr_pipeline import engine_status
    blender = find_blender()
    models = {kind: (ROOT / "models" / f"pose_landmarker_{kind}.task").is_file() for kind in ("lite", "full")}
    return {"version": __version__, "instance_id": "motionbridge-physics-dev", "project_root": str(ROOT), "pid": os.getpid(), "model_ready": models["full"], "models": models,
            "engines": {"gvhmr_experimental": engine_status(ROOT)}, "blender_ready": bool(blender),
            "blender_path": blender, "device": "CPU", "limits": {"max_seconds": 60, "max_upload_mb": 300},
            "features": {"secondary_motion": False, "secondary_workbench": True, "secondary_default_enabled": False, "secondary_profile_inspection": True, "pose_video": True, "bvh": True, "blender": bool(blender), "fbx": bool(blender),
                         "root_motion": False, "fingers": False, "custom_retarget": False,
                         "motion_refinement": True, "rotation_stabilization": True, "character_presets": True,
                         "contact_ik": True, "image_constrained_arms": True,
                         "hand_assistance": (ROOT / "models/hand_landmarker.task").is_file()}}


@app.get("/api/characters")
def characters():
    from .retarget import character_profiles
    return {"characters": [{k: v for k, v in p.items() if k not in {"scene_path", "weights_scene_path"}}
                           for p in _character_profiles(ROOT, find_blender())]}


@app.get("/api/jobs")
def list_jobs():
    with lock:
        return {"jobs": copy.deepcopy(sorted(jobs.values(), key=lambda j: j["created_at"], reverse=True))}


@app.get("/api/jobs/{job_id}")
def read_job(job_id: str):
    return get_job(job_id)


@app.post("/api/jobs", status_code=202)
async def upload_job(file: UploadFile = File(...), smoothing: float = Form(.55), sample_fps: int = Form(24),
                     model: str = Form("full"), exports_3d: bool = Form(True), refine_motion: str = Form("true"),
                     stabilize_rotations: str = Form("true"), enhance_motion: str = Form("true"),
                     engine: str = Form("mediapipe"), world_arm_correction: str = Form("true"),
                     world_foot_contacts: str = Form("true"), world_body_collision: str = Form("true"),
                     world_head_collision: str = Form("true")):
    if any(v.lower() not in {'true','false'} for v in [world_arm_correction,world_foot_contacts,world_body_collision,world_head_collision]):
        raise HTTPException(422, "世界动作校正开关须为 true 或 false。")
    if engine not in {"mediapipe", "gvhmr_experimental"}:
        raise HTTPException(422, "不支持的识别引擎。")
    if not 0 <= smoothing <= 1 or not 12 <= sample_fps <= 30 or model not in {"lite", "full"}:
        raise HTTPException(422, "无效参数：平滑 0–1、帧率 12–30、模型 lite/full。")
    # FastAPI interprets absent bool form fields as unchecked, even with True
    # defaults. Explicit parsing preserves default-on API behavior for old clients.
    if refine_motion.lower() not in {"true", "false"}:
        raise HTTPException(422, "动作校正参数须为 true 或 false。")
    if stabilize_rotations.lower() not in {"true", "false"}:
        raise HTTPException(422, "旋转稳定参数须为 true 或 false。")
    if enhance_motion.lower() not in {"true", "false"}:
        raise HTTPException(422, "手脚增强参数须为 true 或 false。")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
        raise HTTPException(415, "请选择 MP4、MOV、AVI、MKV 或 WebM 视频。")
    if engine == "mediapipe" and not (ROOT / "models" / f"pose_landmarker_{model}.task").is_file():
        raise HTTPException(409, "姿态模型尚未安装，请运行 安装或修复.bat。")
    settings = {"smoothing": smoothing, "sample_fps": sample_fps, "model": model, "exports_3d": exports_3d,
                "root_mode": "in_place", "max_seconds": 60, "refine_motion": refine_motion.lower() == "true",
                "stabilize_rotations": stabilize_rotations.lower() == "true", "enhance_motion": enhance_motion.lower() == "true"}
    settings["engine"] = engine
    settings.update(world_arm_correction=world_arm_correction.lower()=='true',
                    world_body_collision=world_body_collision.lower()=='true',
                    world_head_collision=world_head_collision.lower()=='true',
                    world_foot_contacts=world_foot_contacts.lower()=='true')
    if engine == "gvhmr_experimental":
        from .gvhmr_pipeline import engine_status, effective_settings
        if not engine_status(ROOT)["installed"]:
            raise HTTPException(409, "GVHMR 独立环境或模型未安装完整。")
        settings = effective_settings(settings)
    job = new_job(file.filename or "video", settings)
    folder = DATA / job["id"] / "input"
    folder.mkdir()
    path = folder / f"source{suffix}"
    total = 0
    try:
        with path.open("wb") as stream:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD:
                    raise HTTPException(413, "视频超过 300 MB，请先裁剪或压缩。")
                stream.write(chunk)
        if total == 0:
            raise HTTPException(400, "上传文件为空。")
    except Exception as exc:
        path.unlink(missing_ok=True)
        update(job["id"], status="failed", stage="上传失败", error=str(getattr(exc, "detail", exc)))
        raise
    finally:
        await file.close()
    pool.submit(run_job, job["id"], path)
    log.info("Upload id=%s name=%s bytes=%s", job["id"], job["name"], total)
    return get_job(job["id"])


@app.post("/api/demo", status_code=202)
def demo():
    job = new_job("内置动作演示 · 非视频识别", {"exports_3d": False, "root_mode": "in_place"}, is_demo=True)
    pool.submit(run_job, job["id"])
    return job


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str):
    job = get_job(job_id)
    if job["status"] not in {"queued", "processing"}:
        return job
    cancel_events[job_id].set()
    update(job_id, stage="正在取消…")
    return get_job(job_id)


@app.post("/api/jobs/{job_id}/export-blender", status_code=202)
def export_again(job_id: str):
    with lock:
        job = get_job(job_id)
        if job.get("kind") == "retarget":
            raise HTTPException(409, "角色任务请返回源视频任务重新替换。")
        if job["status"] in {"queued", "processing"}:
            raise HTTPException(409, "任务仍在运行。")
        if not (DATA / job_id / "outputs" / "motion.json").is_file():
            raise HTTPException(409, "没有可导出的动作数据，请先完成识别。")
        if not find_blender():
            raise HTTPException(409, "未找到 Blender，请在 config.json 设置 blender_path。")
        if sum(j["status"] in {"queued", "processing"} for j in jobs.values()) >= MAX_QUEUE:
            raise HTTPException(429, "队列已满。")
        cancel_events[job_id] = threading.Event()
        update(job_id, status="queued", stage="等待 Blender 导出", progress=0, error=None)
    pool.submit(run_job, job_id, export_only=True)
    return get_job(job_id)


@app.post("/api/jobs/{job_id}/reprocess", status_code=202)
def reprocess(job_id: str, engine: str | None = Form(None), world_arm_correction: str | None = Form(None),
              world_foot_contacts: str | None = Form(None), world_body_collision: str | None = Form(None),
              world_head_collision: str | None = Form(None)):
    """Run current inference/refinement on a copy; never mutate historical jobs."""
    original = get_job(job_id)
    if original["is_demo"] or original.get("kind") == "retarget" or original["status"] in {"queued", "processing"}:
        raise HTTPException(409, "请选择已结束的真实视频任务。")
    sources = [p for p in (DATA / job_id / "input").glob("source.*")
               if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}]
    if len(sources) != 1:
        raise HTTPException(409, "原始视频不可用，请重新上传。")
    settings = {**original["settings"], "refine_motion": True, "stabilize_rotations": True, "enhance_motion": True}
    for key,value in [('world_arm_correction',world_arm_correction),('world_foot_contacts',world_foot_contacts),('world_body_collision',world_body_collision),('world_head_collision',world_head_collision)]:
        if value is not None:
            if value.lower() not in {'true','false'}:
                raise HTTPException(422,'世界动作校正开关须为 true 或 false。')
            settings[key]=value.lower()=='true'
    selected = engine or settings.get("engine", "mediapipe")
    if selected not in {"mediapipe", "gvhmr_experimental"}:
        raise HTTPException(422, "不支持的识别引擎。")
    settings["engine"] = selected
    if selected == "gvhmr_experimental":
        from .gvhmr_pipeline import engine_status, effective_settings
        if not engine_status(ROOT)["installed"]:
            raise HTTPException(409, "GVHMR 独立环境或模型未安装完整。")
        settings = effective_settings(settings)
    elif original["settings"].get("engine") == "gvhmr_experimental":
        settings.update(root_mode="in_place", max_seconds=60, smoothing=.55, model="full")
    job = new_job(original["name"] + " · 新版重算", settings)
    try:
        folder = DATA / job["id"] / "input"
        folder.mkdir()
        target = folder / ("source" + sources[0].suffix)
        shutil.copy2(sources[0], target)
        update(job["id"], reprocessed_from=job_id, software_version=__version__)
        pool.submit(run_job, job["id"], target)
    except Exception:
        update(job["id"], status="failed", stage="重新处理失败", error="无法复制原片或提交任务；旧任务未改动。")
        raise
    return get_job(job["id"])


def run_character_job(job_id, original_id, profile, blender, validate_only=False):
    event = cancel_events[job_id]
    output = DATA / job_id / "outputs"
    original = get_job(original_id)
    source = DATA / original_id / "outputs"
    def progress(percent, stage):
        if event.is_set():
            raise InterruptedError('任务已取消')
        update(job_id, status="processing", progress=percent, stage=stage)
    try:
        inputs = DATA / job_id / 'input'
        if validate_only:
            from .retarget import audit_existing_character
            report=audit_existing_character(ROOT,blender,profile,inputs/'motion.json',output,event.is_set,progress)
        else:
            progress(5, '复制源动作，保留原任务')
            inputs.mkdir(exist_ok=True)
            for filename in ['motion.json', 'motion.fbx']:
                if (source / filename).is_file():shutil.copy2(source / filename, inputs / filename)
            for filename in ['motion.json', 'source.mp4']:
                if (source / filename).is_file():shutil.copy2(source / filename, output / filename)
            update(job_id, files=output_files(job_id), meta=original.get('meta', {}))
            if not (inputs / 'motion.fbx').exists():
                from .exporters import export_blender
                progress(10, '为角色替换生成源 FBX')
                export_blender(inputs / 'motion.json', inputs, blender, cancelled=event.is_set)
            from .retarget import export_character
            if profile.get('id')=='rin':
                from .secondary_rin import export_character
            report = export_character(ROOT, blender, profile, inputs / 'motion.json', output, event.is_set, progress)
        if event.is_set():
            raise InterruptedError('任务已取消')
        update(job_id, status='completed', stage='角色动作替换完成', progress=100,
               files=output_files(job_id), character_validation=report,
               warnings=original.get('warnings', []) + ['网页骨架窗口仍显示源动作；目标角色请看静帧或打开角色工程。',
                         '仅支持此处配置的角色骨架；未捕捉精细手指和表情。衣物与发饰避碰使用有限几何修正，不是布料物理，仍可能穿插。'
                         if profile.get('surface_clearance',True) else
                         '仅支持此处配置的角色骨架；未捕捉精细手指和表情，衣裙为蒙皮变形，未启用衣物与发饰避碰。',
                         '世界根位移按腿长比例传递；启用世界接地时按估计支撑修正高度并进行有限腿 IK，仍可能脚滑、离地或肢体跳变。'
                         if original['settings'].get('engine') == 'gvhmr_experimental' else
                         '接地与旋转稳定为估算处理，仍可能脚滑、穿插或丢失快速动作细节。'])
    except InterruptedError:
        update(job_id, status='cancelled', stage='已取消', files=output_files(job_id))
    except Exception as exc:
        log.exception('Character job failed: %s', job_id)
        update(job_id, status='failed', stage='角色替换失败', error=str(exc)[:1200], files=output_files(job_id))


@app.post('/api/jobs/{job_id}/retarget', status_code=202)
def retarget_job(job_id: str, character: str = Form('nahida'),surface_clearance: str = Form('true')):
    from .retarget import character_profiles
    if surface_clearance.lower() not in {'true','false'}:raise HTTPException(422,'衣物与发饰避碰开关须为 true 或 false。')
    original = get_job(job_id)
    if original.get('kind') == 'retarget' or original['status'] != 'completed':
        raise HTTPException(409, '请选择已完成的源动作任务。')
    blender = find_blender()
    profile = next((p for p in _character_profiles(ROOT, blender) if p['id'] == character), None)
    if not profile:
        raise HTTPException(422, '不支持此角色预设。')
    if not profile['ready']:
        raise HTTPException(409, profile['reason'])
    profile={**profile,'surface_clearance':surface_clearance.lower()=='true' and character!='rin'}
    if not (DATA / job_id / 'outputs' / 'motion.json').is_file():
        raise HTTPException(409, '源动作数据不可用。')
    # New immutable source relationship; repeated requests create distinct outputs.
    job = new_job(('远坂凛 · ' if character=='rin' else '纳西妲 · ') + original['name'], {'character': character, 'exports_3d': False,
                  'surface_clearance':profile['surface_clearance'],
                  'engine': original['settings'].get('engine','mediapipe'),
                  'root_mode': original['settings'].get('root_mode','in_place')}, original['is_demo'])
    update(job['id'], kind='retarget', source_job_id=job_id, character=character)
    pool.submit(run_character_job, job['id'], job_id, profile, blender)
    return get_job(job['id'])


@app.post('/api/jobs/{job_id}/retarget/retry-validation',status_code=202)
def retry_character_validation(job_id: str):
    from .retarget import character_profiles,validate_character_checkpoint
    job=get_job(job_id)
    if job.get('kind')!='retarget' or job['status'] not in {'failed','cancelled'}:
        raise HTTPException(409,'请选择已生成工程但验证未完成的角色任务。')
    if not job.get('source_job_id') or job['source_job_id'] not in jobs:
        raise HTTPException(409,'源动作任务记录缺失，无法恢复验证。')
    blender=find_blender()
    profile=next((p for p in character_profiles(ROOT,blender) if p['id']==job.get('character')),None)
    if not profile or not profile['ready']:raise HTTPException(409,'本机角色或Blender配置不可用。')
    profile={**profile,'surface_clearance':job['settings'].get('surface_clearance',False)}
    try:validate_character_checkpoint(profile,DATA/job_id/'input/motion.json',DATA/job_id/'outputs')
    except (OSError,ValueError,KeyError,TypeError) as exc:raise HTTPException(409,str(exc)) from exc
    with lock:
        if jobs[job_id]['status'] not in {'failed','cancelled'}:raise HTTPException(409,'此任务已经在处理。')
        if sum(j['status'] in {'queued','processing'} for j in jobs.values())>=MAX_QUEUE:raise HTTPException(429,'任务队列已满。')
        history=list(jobs[job_id].get('validation_history',[]))
        history.append({'at':now(),'previous_status':job['status'],'previous_error':job.get('error'),'validator_version':__version__})
        cancel_events[job_id]=threading.Event()
        update(job_id,status='queued',stage='等待重新验证角色工程',progress=84,error=None,
               validation_history=history,validation_retries=len(history))
    pool.submit(run_character_job,job_id,job['source_job_id'],profile,blender,validate_only=True)
    return get_job(job_id)


@app.get("/api/jobs/{job_id}/motion")
def motion_data(job_id: str):
    get_job(job_id)
    path = DATA / job_id / "outputs" / "motion.json"
    if not path.is_file():
        raise HTTPException(404, "动作数据尚未就绪。")
    return FileResponse(path, media_type="application/json")


@app.get("/api/jobs/{job_id}/files/{filename}")
def get_file(job_id: str, filename: str):
    get_job(job_id)
    if filename not in OUTPUT_NAMES:
        raise HTTPException(404, "文件不存在。")
    path = DATA / job_id / "outputs" / filename
    if not path.is_file():
        raise HTTPException(404, "文件尚未生成。")
    if path.suffix == ".mp4":
        return FileResponse(path, media_type="video/mp4")
    return FileResponse(path, filename=filename)


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str):
    job = get_job(job_id)
    if job["status"] in {"queued", "processing"}:
        raise HTTPException(409, "请等待任务结束后下载。")
    files = output_files(job_id)
    reference=DATA/job_id/'outputs/surface_solver_reference.npz'
    if job.get('kind')=='retarget' and reference.is_file():files.append({'name':reference.name})
    if not files:
        raise HTTPException(404, "没有可下载的输出。")
    archive = DATA / job_id / "MotionBridge-output.zip"
    # A per-job archive lock prevents simultaneous downloads from racing the writer.
    with lock:
        newest = max((DATA / job_id / "outputs" / f["name"]).stat().st_mtime for f in files)
        if not archive.exists() or archive.stat().st_mtime < newest:
            temporary = archive.with_suffix(".tmp")
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for item in files:
                    if item["name"] != "source.mp4":
                        z.write(DATA / job_id / "outputs" / item["name"], item["name"])
                z.writestr("READ_ME.txt", "MotionBridge " + __version__ + "\n单人身体动画；根运动模式：" + job["settings"].get("root_mode", "见动作源") + "；不含精细手指/表情。\n" + "\n".join(job["warnings"]))
            temporary.replace(archive)
    return FileResponse(archive, filename=f"MotionBridge-{job_id[:8]}.zip", media_type="application/zip")


@app.get("/")
def index():
    return FileResponse(ROOT / "motionbridge" / "static" / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=ROOT / "motionbridge" / "static"), name="static")


# secondary workbench dev6: same single worker queue, separate run state.
def _character_profiles(root, blender):
    from .retarget import character_profiles
    from .secondary_rin import profile
    return character_profiles(root, blender) + profile(root, blender)

_secondary_instances = {}
def secondary_workbench():
    from .secondary_workbench import Workbench
    key = (str(ROOT.resolve()), str(DATA.resolve()))
    with lock:
        if key not in _secondary_instances:
            _secondary_instances[key] = Workbench(ROOT, DATA, pool)
        return _secondary_instances[key]

from .secondary_workbench import router as secondary_router
app.include_router(secondary_router(secondary_workbench))
