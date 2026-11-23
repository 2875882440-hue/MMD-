"""Local, explicitly configured character presets and cancellable Blender jobs."""
import json
import hashlib
import math
import os
from pathlib import Path
import subprocess
import shutil
import time
import uuid


def character_profiles(root, blender):
    try:
        config = json.loads((Path(root) / 'config.json').read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        config = {}
    configured = config.get('characters', {}).get('nahida', {})
    source, weights = configured.get('scene_path'), configured.get('weights_scene_path')
    ready = bool(blender and source and weights and Path(source).is_file() and Path(weights).is_file())
    return [{'id': 'nahida', 'label': '纳西妲 · 已验证 MMD 骨架', 'ready': ready,
             'reason': '' if ready else '需要 Blender 及已配置的纳西妲原工程；在 config.json 设置 characters.nahida。',
             'scene_path': source, 'weights_scene_path': weights}]


def run_script(blender, script, arguments, log_path, cancelled, timeout=600):
    command = [str(blender), '--background', '--factory-startup', '--disable-autoexec', '--threads', '4',
               '--python-exit-code', '23', '--python', str(script), '--', *map(str, arguments)]
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    with Path(log_path).open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        start = time.monotonic()
        try:
            while process.poll() is None:
                if cancelled():
                    raise InterruptedError('角色动作替换已取消')
                if time.monotonic() - start > timeout:
                    raise RuntimeError('角色处理超时，请查看角色日志。')
                time.sleep(.15)
            if cancelled():
                raise InterruptedError('角色动作替换已取消')
            if process.returncode:
                raise RuntimeError(f'Blender 角色处理失败（{process.returncode}），请查看角色日志。')
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            raise


def character_timeouts(frame_count, surface_enabled):
    """Budget measured per-frame cloth work without disabling cancellation."""
    if type(frame_count) is not int or frame_count < 1:
        raise ValueError('角色动作必须包含至少一帧')
    bake = max(2400, 600 + 4 * frame_count) if surface_enabled else 600
    audit = max(900, math.ceil(180 + frame_count))
    return bake, audit


def export_character(root, blender, profile, motion_path, output, cancelled, progress):
    scripts = Path(root) / 'scripts'
    motion = json.loads(Path(motion_path).read_text(encoding='utf-8'))
    bake_timeout, audit_timeout = character_timeouts(len(motion['frames']), profile.get('surface_clearance', True))
    progress(35, '替换纳西妲动作、约束衣物穿插与网格拉扯')
    run_script(blender, scripts / 'retarget_nahida.py',
               [profile['scene_path'], motion_path, profile['weights_scene_path'], output,str(profile.get('surface_clearance',True)).lower()],
               output / 'character_export.log', cancelled, timeout=bake_timeout)
    progress(85, '重新打开角色工程，检查全帧连续性')
    run_script(blender, scripts / 'audit_retarget_review.py', [output], output / 'character_audit.log', cancelled,
               timeout=audit_timeout)
    report = json.loads((output / 'saved_project_audit.json').read_text(encoding='utf-8'))
    if not report.get('passed'):
        raise RuntimeError('角色工程验证未通过，请查看角色验证记录。')
    for filename in ['Nahida_MotionBridge_Dance.blend', 'character_preview.png']:
        path = output / filename
        if not path.is_file() or path.stat().st_size < 1000:
            raise RuntimeError('角色文件或预览未生成。')
    return report


def validate_character_checkpoint(profile,motion_path,output):
    """Only resume the matching generated project; never rebuild or change it."""
    output=Path(output).resolve();motion_path=Path(motion_path).resolve()
    for name in ['Nahida_MotionBridge_Dance.blend','retarget_report.json']:
        path=output/name
        if not path.is_file() or path.stat().st_size<100:
            raise ValueError('角色工程尚未完整保存，请从源视频重新生成。')
    report=json.loads((output/'retarget_report.json').read_text(encoding='utf-8'))
    if (Path(report['output']).resolve()!=output/'Nahida_MotionBridge_Dance.blend'
        or Path(report['motion']).resolve()!=motion_path
        or Path(report['original_project']).resolve()!=Path(profile['scene_path']).resolve()):
        raise ValueError('工程检查点与本任务或当前角色配置不匹配。')
    if not motion_path.is_file() or hashlib.sha256(motion_path.read_bytes()).hexdigest()!=report['motion_sha256']:
        raise ValueError('源动作已改变，不能复用此角色工程。')
    if hashlib.sha256(Path(profile['scene_path']).read_bytes()).hexdigest()!=report['original_sha256_before']:
        raise ValueError('原角色工程已改变，不能复用此检查点。')
    enabled=bool(report.get('surface_clearance',{}).get('enabled',False))
    if enabled!=profile.get('surface_clearance',False):raise ValueError('衣物约束设置与检查点不匹配。')
    motion=json.loads(motion_path.read_text(encoding='utf-8'))
    if len(motion['frames'])!=report['frames']:raise ValueError('检查点帧数不一致。')
    character_timeouts(len(motion['frames']),enabled)
    return report


def audit_existing_character(root,blender,profile,motion_path,output,cancelled,progress):
    """Retry only the saved-project audit, preserving failed evidence."""
    output=Path(output);checkpoint=validate_character_checkpoint(profile,motion_path,output)
    history=output/'validation_history'/uuid.uuid4().hex
    history.mkdir(parents=True)
    for name in ['saved_project_audit.json','character_audit.log']:
        if (output/name).exists():shutil.copy2(output/name,history/name)
    target=output/'Nahida_MotionBridge_Dance.blend';before=hashlib.sha256(target.read_bytes()).hexdigest()
    progress(85,'重新验证已保存的角色工程')
    run_script(blender,Path(root)/'scripts/audit_retarget_review.py',[output],output/'character_audit.log',cancelled,
               timeout=character_timeouts(checkpoint['frames'],profile.get('surface_clearance',False))[1])
    report=json.loads((output/'saved_project_audit.json').read_text(encoding='utf-8'))
    if not report.get('passed'):raise RuntimeError('角色工程验证仍未通过，请查看验证记录。')
    if hashlib.sha256(target.read_bytes()).hexdigest()!=before:raise RuntimeError('验证期间角色工程发生改变。')
    if not (output/'character_preview.png').is_file() or (output/'character_preview.png').stat().st_size<1000:
        raise RuntimeError('角色静帧未生成。')
    return report
