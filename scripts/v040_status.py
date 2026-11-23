"""Record existing v0.4 jobs; never submit duplicate work."""
from pathlib import Path
import json
import ctypes
import subprocess
import csv
import httpx

root=Path(__file__).resolve().parents[1]
p=root/'data/release-v040/final_jobs.json'
ids=json.loads(p.read_text(encoding='utf-8'))
with httpx.Client(trust_env=False,timeout=10) as c:
    jobs=c.get('http://127.0.0.1:8765/api/jobs').json()['jobs']
    child=next((j for j in jobs if j.get('kind')=='retarget' and j.get('source_job_id')==ids['dance']),None)
    if child:
        ids['character']=child['id']
        p.write_text(json.dumps(ids,indent=2),encoding='utf-8')
    for name,ident in ids.items():
        j=next(j for j in jobs if j['id']==ident)
        print(name,ident,j['status'],j['stage'],j.get('error'),flush=True)
        (p.parent/(name+'_job.json')).write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8')
class Memory(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ['total','avail','totalPage','availPage','totalVirtual','availVirtual','extended']]
m=Memory()
m.length=ctypes.sizeof(m)
ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
print('Memory used %',m.load,'available GB',round(m.avail/1024**3,2),'commit available GB',round(m.availPage/1024**3,2))
listing=subprocess.run(['tasklist','/FO','CSV'],capture_output=True,text=True,errors='replace')
for row in csv.reader(listing.stdout.splitlines()):
    if row and row[0].lower() in {'blender.exe','python.exe','chrome.exe','node.exe'}:
        print('process',row[0],row[1],row[-1])
