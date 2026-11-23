"""Trusted local connector settings. The HTTP client selects a registered provider by name."""
from pathlib import Path
from .secondary_cache import read_json

def mmd_settings(root):
    path = Path(root) / 'config.json'
    config = read_json(path).get('blender_plugins', {}).get('mmd_tools', {}) if path.is_file() else {}
    exe = Path(config.get('blender_path') or '')
    manifest = exe.parent / 'portable/extensions/user_default/mmd_tools/blender_manifest.toml'
    ready = exe.is_absolute() and exe.is_file() and manifest.is_file()
    return {'exe': exe, 'ready': ready, 'fallback': config.get('fallback', True) is True,
            'fallback_mode': 'surface' if config.get('fallback_mode')=='surface' else 'hair',
            'reason': '' if ready else '隔离版 Blender 或 MMD Tools 未找到。'}

def mmd_command(root, job, pair, phase='bake'):
    settings = mmd_settings(root)
    if not settings['ready']:
        raise ValueError(settings['reason'])
    return [str(settings['exe']), '-b', '--offline-mode', '--disable-autoexec', '--threads', '3',
            '--python-exit-code', '23', '--python', str(Path(root)/'scripts/run_mmd_plugin.py'),
            '--', '--phase', phase, '--scene', job['source']['scene'], '--source-sha', job['source_sha256'],
            '--out', str(pair)]
