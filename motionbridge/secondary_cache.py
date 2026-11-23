"""Content-addressed secondary-motion artifacts. No simulation or app import."""
from __future__ import annotations
from pathlib import Path,PurePosixPath
import contextlib,ctypes,hashlib,json,os,re,shutil,time,uuid

SCHEMA=1

def file_sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def key_for(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def atomic_json(path,value,*,durable=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    with tmp.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);stream.flush()
        if durable:os.fsync(stream.fileno())
    for attempt in range(20):
        try:os.replace(tmp,path);break
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05)

def read_json(path):
    for attempt in range(20):
        try:return json.loads(Path(path).read_text(encoding='utf-8'))
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05)

def relative_name(value):
    if not isinstance(value,str) or '\\' in value or ':' in value:raise ValueError('Invalid artifact path')
    p=PurePosixPath(value)
    if p.is_absolute() or not p.parts or any(x in ('.','..') for x in p.parts):raise ValueError('Invalid artifact path')
    return p

def checked_file(root,name):
    p=Path(root).joinpath(*relative_name(name).parts)
    if p.is_symlink() or not p.resolve().is_relative_to(Path(root).resolve()):raise ValueError('Artifact escapes its directory')
    return p

def process_identity(pid=None):
    pid=int(pid or os.getpid())
    if os.name!='nt':
        try:os.kill(pid,0);return str(pid)
        except ProcessLookupError:return None
        except PermissionError:return str(pid)
    from ctypes import wintypes as w
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];kernel.OpenProcess.restype=w.HANDLE
    kernel.GetExitCodeProcess.argtypes=[w.HANDLE,ctypes.POINTER(w.DWORD)]
    kernel.GetProcessTimes.argtypes=[w.HANDLE,*([ctypes.POINTER(w.FILETIME)]*4)]
    kernel.CloseHandle.argtypes=[w.HANDLE]
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:return None
    try:
        status=w.DWORD()
        if not kernel.GetExitCodeProcess(handle,ctypes.byref(status)) or status.value!=259:return None
        times=[w.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle,*[ctypes.byref(t) for t in times]):return None
        return str((times[0].dwHighDateTime<<32)|times[0].dwLowDateTime)
    finally:kernel.CloseHandle(handle)

class BusyError(RuntimeError):pass

@contextlib.contextmanager
def lease(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    owner={'pid':os.getpid(),'identity':process_identity(),'token':uuid.uuid4().hex,'started':time.time()}
    for _ in range(3):
        try:
            with path.open('x',encoding='utf-8') as stream:json.dump(owner,stream)
            break
        except FileExistsError:
            try:old=json.loads(path.read_text(encoding='utf-8'))
            except (ValueError,OSError):raise BusyError('Unreadable lock; inspect before retrying')
            if old.get('identity') and process_identity(old.get('pid'))==old['identity']:raise BusyError(f'Already running: PID {old["pid"]}')
            # Retain stale owner evidence; this is a checked same-parent rename.
            target=path.with_name(path.name+'.stale-'+uuid.uuid4().hex)
            assert path.resolve().parent==target.resolve().parent
            try:path.rename(target)
            except FileNotFoundError:continue
    else:raise BusyError('Concurrent lock changes')
    try:yield owner
    finally:
        try:
            if json.loads(path.read_text(encoding='utf-8')).get('token')==owner['token']:path.unlink()
        except FileNotFoundError:pass

class ArtifactCache:
    def __init__(self,root):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)

    def entry(self,stage,key):
        if not re.fullmatch('[a-z0-9_-]{1,40}',stage) or not re.fullmatch('[a-f0-9]{64}',key):raise ValueError('Invalid stage/cache key')
        return self.root/'objects'/stage/key

    def verify(self,stage,key):
        path=self.entry(stage,key)
        try:
            m=json.loads((path/'manifest.json').read_text(encoding='utf-8'))
            if m.get('schema')!=SCHEMA or m.get('stage')!=stage or m.get('key')!=key or not m.get('files'):return None
            for name,record in m['files'].items():
                p=checked_file(path/'payload',name)
                if not p.is_file() or p.stat().st_size!=record['bytes'] or file_sha(p)!=record['sha256']:return None
            actual={p.relative_to(path/'payload').as_posix() for p in (path/'payload').rglob('*') if p.is_file()}
            if actual!=set(m['files']):return None
            return m
        except (OSError,ValueError,KeyError,TypeError):return None

    @contextlib.contextmanager
    def reserve(self,stage,key):
        self.entry(stage,key)
        with lease(self.root/'locks'/f'{stage}-{key}.json'):yield

    def publish(self,stage,key,source,names,*,provenance=None,prepare=None):
        target=self.entry(stage,key);source=Path(source).resolve()
        existing=self.verify(stage,key)
        if existing:return existing
        if target.exists():
            quarantine=self.root/'quarantine'/f'{stage}-{key}-{uuid.uuid4().hex}';quarantine.parent.mkdir(parents=True,exist_ok=True)
            assert target.resolve().is_relative_to(self.root) and quarantine.resolve().is_relative_to(self.root)
            target.rename(quarantine)
        temp=self.root/'pending'/uuid.uuid4().hex;payload=temp/'payload';payload.mkdir(parents=True)
        try:
            for name in sorted(set(names)):
                origin=checked_file(source,name);dest=checked_file(payload,name)
                if not origin.is_file():raise ValueError(f'Missing artifact: {name}')
                dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(origin,dest)
            if prepare:prepare(source,payload)
            files={}
            for p in sorted(payload.rglob('*')):
                if p.is_file():
                    name=p.relative_to(payload).as_posix();checked_file(payload,name)
                    files[name]={'sha256':file_sha(p),'bytes':p.stat().st_size}
            if not files:raise ValueError('Cannot cache empty output')
            m={'schema':SCHEMA,'stage':stage,'key':key,'files':files,'provenance':provenance or {},
               'meaning':'artifact integrity/completed computation only; not physics quality approval'}
            atomic_json(temp/'manifest.json',m);target.parent.mkdir(parents=True,exist_ok=True)
            temp.rename(target)
            if self.verify(stage,key) is None:raise ValueError('Published cache failed verification')
            return m
        except BaseException:
            # Preserve the partial snapshot for diagnosis, never index it.
            if temp.exists():atomic_json(temp/'failed.json',{'stage':stage,'key':key,'failed':True})
            raise

    def restore(self,stage,key,destination):
        m=self.verify(stage,key)
        if m is None:return None
        destination=Path(destination).resolve()
        if destination.exists():raise ValueError('Refuse to overwrite existing working artifacts')
        destination.mkdir(parents=True)
        for name in m['files']:
            origin=checked_file(self.entry(stage,key)/'payload',name);target=checked_file(destination,name)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(origin,target)
        return m

def stage_keys(stamp,profile,code,*,input_override=None):
    """Changing cloth parameters must not invalidate hair, and conversely."""
    structural={k:v for k,v in profile.items() if k not in ('hair_dynamics','surface_dynamics')}
    surface=profile.get('surface_dynamics',{})
    classification={k:surface.get(k) for k in ('body_materials','ornament_bone_tokens','maximum_rigid_extent_m')}
    capture=input_override or key_for({'schema':SCHEMA,'stage':'capture','stamp':stamp,'profile':structural,'classification':classification,'code':code['capture']})
    cloth=key_for({'stage':'cloth','capture':capture,'settings':surface,'code':code['cloth']})
    hair=key_for({'stage':'hair','capture':capture,'settings':profile.get('hair_dynamics',{}),'floor':surface.get('floor_z_m',0),'code':code['hair']})
    return {'capture':capture,'cloth':cloth,'hair':hair}
