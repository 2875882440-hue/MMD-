from pathlib import Path
from test_secondary_workbench import work,start
from motionbridge.secondary_cache import atomic_json
from motionbridge.blender_plugins import mmd_settings

def test_optional_plugin_missing_falls_back(work,monkeypatch):
    manager,c,_,_=work
    assert c.get('/api/secondary/catalog').json()['plugins']['mmd_tools']['ready'] is False
    job=start(work,mode='mmd_tools',preview=False).json()
    calls=[]
    def capture(jid,cmd,managed):
        calls.append(cmd)
        return 13
    monkeypatch.setattr(manager,'_call',capture)
    manager.execute(job['id'],False)
    result=manager.get(job['id'])
    assert Path(result['output']).name=='internal-fallback' and not Path(result['output']).exists()
    assert result['actual_provider']=='internal' and '未找到' in result['fallback_reason']
    assert len(calls)==1 and any('simulate_secondary_workbench_hair.py' in s for s in calls[0])
    assert result['status']=='failed' and not result.get('result')

def test_plugin_fallback_can_be_disabled(work):
    manager,_,_,_=work
    atomic_json(manager.root/'config.json',{'blender_plugins':{'mmd_tools':{'fallback':False}}})
    job=start(work,mode='mmd_tools',preview=False).json()
    manager.execute(job['id'],False)
    result=manager.get(job['id'])
    assert result['status']=='failed' and '未找到' in result['error']
    assert not result.get('actual_provider')


def test_partial_bake_stays_private_until_verified(work):
    manager, client, _, _ = work
    job = start(work, mode='mmd_tools').json()
    pair = Path(job['output']) / 'mmd-test'
    pair.mkdir(parents=True)
    (pair / 'after.blend').write_bytes(b'unverified')
    endpoint = '/api/secondary/runs/' + job['id']
    assert client.get(endpoint).json()['files'] == []
    manager._update(job['id'], result={'pair': str(pair), 'case': str(pair)}, protection_passed=True)
    assert client.get(endpoint).json()['files'][0]['name'] == 'after.blend'
