"""Owned subprocess cancellation and durable, revisioned cache-stage execution."""
from pathlib import Path
import json,os,re,shutil,subprocess,time,uuid
from .secondary_cache import atomic_json,file_sha,key_for,lease,process_identity,checked_file,BusyError,read_json

class Cancelled(RuntimeError):pass

def snapshot(directory,names):
    return {n:{'sha256':file_sha(checked_file(directory,n)),'bytes':checked_file(directory,n).stat().st_size} for n in names}

def matches(directory,files):
    try:return bool(files) and all(checked_file(directory,n).is_file() and checked_file(directory,n).stat().st_size==v['bytes'] and file_sha(checked_file(directory,n))==v['sha256'] for n,v in files.items())
    except (OSError,ValueError,KeyError):return False

class RunSession:
    def __init__(self,root,spec,cache=None,*,resume=False):
        self.root=Path(root).resolve();self.cache=cache;self.spec=spec;self.revision=key_for(spec)
        if self.root.exists():
            if not resume:raise ValueError('Output exists; use --resume for a managed run')
            self.state=read_json(self.root/'run.json')
            if self.state.get('schema')!=1 or not self.state.get('run_id'):raise ValueError('Not a managed secondary run')
        else:
            if resume:raise ValueError('No managed run to resume')
            self.root.mkdir(parents=True);self.state={'schema':1,'run_id':uuid.uuid4().hex,'stages':{},'revisions':{},'status':'created'};self.save();atomic_json(self.root/'control.json',{'run_id':self.state['run_id']})
        self.active_stage=None;self.owner=None

    def save(self,*,durable=True):atomic_json(self.root/'run.json',self.state,durable=durable)

    def __enter__(self):
        self._lease=lease(self.root/'run.lock');self.owner=self._lease.__enter__()
        for old in self.state['stages'].values():
            if old.get('child_identity') and not old.get('child_finished',True) and process_identity(old.get('child_pid'))==old['child_identity']:
                self._lease.__exit__(None,None,None)
                raise BusyError(f'Previous owned child is still running: {old["child_pid"]}')
        if not (self.root/'control.json').exists():atomic_json(self.root/'control.json',{'run_id':self.state['run_id']})
        # Resume is explicit; retain the previous cancellation request.
        p=self.root/'cancel.json'
        if p.exists():p.rename(self.root/f'cancel.previous-{uuid.uuid4().hex}.json')
        self.state['status']='running';self.state['owner']=self.owner;self.state['current_revision']=self.revision
        self.state['revisions'].setdefault(self.revision,{'spec':self.spec,'started':time.time()});self.save()
        self.revision_root=self.root/'revisions'/self.revision[:16]
        marker=self.revision_root/'revision.json'
        if marker.is_file() and json.loads(marker.read_text(encoding='utf-8')).get('key')!=self.revision:
            self.revision_root=self.root/'revisions'/self.revision
            marker=self.revision_root/'revision.json'
        self.revision_root.mkdir(parents=True,exist_ok=True);atomic_json(marker,{'key':self.revision})
        return self

    def __exit__(self,kind,error,trace):
        self.state['status']='cancelled' if isinstance(error,(Cancelled,KeyboardInterrupt)) else ('failed' if error else 'completed')
        self.state['finished']=time.time();self.state['error']=str(error) if error else None;self.save()
        return self._lease.__exit__(kind,error,trace)

    @staticmethod
    def cancel(root):
        root=Path(root).resolve();state=read_json(root/'control.json' if (root/'control.json').exists() else root/'run.json')
        atomic_json(root/'cancel.json',{'run_id':state['run_id'],'requested':time.time()},durable=False)

    def check_cancel(self):
        p=self.root/'cancel.json'
        if p.exists() and read_json(p).get('run_id')==self.state['run_id']:raise Cancelled('Cancelled; partial outputs retained')

    def execute(self,command,log_name=None):
        self.check_cancel();logdir=self.root/'logs';logdir.mkdir(exist_ok=True)
        logfile=logdir/(log_name or f'{self.state["stages"][self.active_stage]["name"]}-{uuid.uuid4().hex[:8]}.log')
        env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
        with logfile.open('x',encoding='utf-8') as stream:
            child=subprocess.Popen(list(map(str,command)),stdout=stream,stderr=subprocess.STDOUT,env=env,
                                   creationflags=(getattr(subprocess,'CREATE_NO_WINDOW',0) if os.name=='nt' else 0))
            row=self.state['stages'][self.active_stage];row.update(child_pid=child.pid,child_identity=process_identity(child.pid),log=str(logfile));self.save(durable=False)
            try:
                while child.poll() is None:self.check_cancel();time.sleep(.15)
                if child.returncode:raise RuntimeError(f'Stage exited {child.returncode}; see {logfile}')
            except BaseException:
                if child.poll() is None:
                    child.terminate()
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
                raise
            finally:row['child_finished']=True;self.save(durable=False)
        return logfile

    def utility(self,command,name='preflight'):
        identifier=name+'-'+uuid.uuid4().hex[:10];self.active_stage=identifier
        self.state['stages'][identifier]={'name':name,'status':'running','started':time.time()};self.save()
        try:
            result=self.execute(command);self.state['stages'][identifier]['status']='completed';self.save();return result
        except BaseException:
            self.state['stages'][identifier]['status']='failed';self.save();raise

    def stage(self,name,key,parent,producer,files,*,prepare=None,provenance=None):
        self.check_cancel()
        if not re.fullmatch('[a-z0-9_-]{1,40}',name):raise ValueError('Invalid stage name')
        identifier=f'{self.revision}:{name}';previous=self.state['stages'].get(identifier)
        if previous:
            if previous.get('child_identity') and not previous.get('child_finished',True) and process_identity(previous.get('child_pid'))==previous.get('child_identity'):
                raise RuntimeError('Previous stage child is still running; do not duplicate it')
            if previous.get('key')==key and matches(previous['output'],previous.get('files',{})):
                previous['last_access']='working_reuse';self.save();return Path(previous['output'])
        parent=Path(parent).resolve()
        if not parent.is_relative_to(self.root):raise ValueError('Stage output must stay within this owned run')
        output=parent/f'{name}-{uuid.uuid4().hex[:10]}'
        row={'name':name,'key':key,'output':str(output),'status':'pending','started':time.time()}
        if previous:row['previous_attempt']=previous
        self.state['stages'][identifier]=row;self.active_stage=identifier;self.save()
        def perform():
            if self.cache:
                m=self.cache.restore(name,key,output)
                if m:
                    row.update(status='cached',files=m['files'],last_access='cache_hit',finished=time.time());self.save();return output
            row['status']='running';self.save();producer(output,self.execute)
            self.check_cancel();names=files(output) if callable(files) else files
            recorded=snapshot(output,names)
            if self.cache:
                m=self.cache.publish(name,key,output,names,provenance=provenance,prepare=prepare)
                # Canonical path-only blend rewrites are applied to this new
                # owned output, never to input scenes or previous revisions.
                for n,v in m['files'].items():
                    target=checked_file(output,n)
                    if not target.is_file() or file_sha(target)!=v['sha256']:
                        target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_name(target.name+'.canonical-'+uuid.uuid4().hex)
                        shutil.copyfile(checked_file(self.cache.entry(name,key)/'payload',n),tmp);os.replace(tmp,target)
                recorded=m['files']
            row.update(status='completed',files=recorded,last_access='computed',finished=time.time());self.save();return output
        try:
            if self.cache:
                with self.cache.reserve(name,key):return perform()
            return perform()
        except BaseException as error:
            row.update(status='cancelled' if isinstance(error,(Cancelled,KeyboardInterrupt)) else 'failed',error=str(error));self.save();raise
