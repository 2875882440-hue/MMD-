const el=id=>document.getElementById(id),active=new Set(['queued','processing','cancelling']);
const state={catalog:null,runs:[],selected:null,busy:false,polling:false};
const labels={queued:'排队等待',processing:'正在处理',cancelling:'正在取消',cancelled:'已取消',interrupted:'上次运行中断',failed:'失败',completed:'计算完成',quality_rejected:'计算完成 · 物理质量未通过'};
Object.assign(labels,{running:'进行中',pending:'等待处理',cached:'已缓存'});
const stageLabels={preflight:'检查源工程','hair-basic':'发束求解与保存验证',capture:'捕获角色表面',cloth:'衣物求解',hair:'耦合发束','bundle-coupled':'合并与审核','bundle-preserved':'保留原头发审核','cloth-guarded':'布片回退','bundle-guarded':'回退后审核','export-blender':'Blender工程','export-skeleton':'骨骼FBX','export-rigid':'硬饰FBX','export-alembic':'Alembic','export-rollback':'回退包'};
const show=(id,yes)=>el(id).classList.toggle('hidden',!yes);
async function api(url,body){const r=await fetch('/api/secondary'+url,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const result=await r.json();if(!r.ok)throw Error(typeof result.detail==='string'?result.detail:JSON.stringify(result.detail));return result;}
function error(message){el('secondaryError').textContent=message;show('secondaryError',true);}
function options(id,rows,placeholder){const select=el(id),value=select.value;select.replaceChildren();for(const row of rows){const option=document.createElement('option');option.value=row.id;option.textContent=row.label;option.disabled=row.ready===false;select.append(option);}if(!rows.length){const option=document.createElement('option');option.textContent=placeholder;option.value='';select.append(option);}if(rows.some(r=>r.id===value))select.value=value;}
function controls(){const profile=state.catalog?.profiles.find(p=>p.id===el('secondaryProfile').value),mode=el('secondaryMode').value;const running=state.runs.some(r=>active.has(r.status));const ready=!!profile?.[mode]&&!!el('secondarySource').value;
 el('secondaryStart').disabled=!el('secondaryEnabled').checked||!ready||running||state.busy;
 el('secondaryHint').textContent=!el('secondaryEnabled').checked?'衣发开关关闭，原动作流程保持。':!ready?(mode==='surface'&&profile?.surface_reason?profile.surface_reason:'请选择有配置的角色与动作工程。'):mode==='surface'?'衣服与耦合碰撞仍为未验收实验；会保留真实质量检查和失败记录。':'使用模型发束链施力；原身体动作受独立保护。';
 if(mode==='mmd_tools'&&el('secondaryEnabled').checked){const p=state.catalog?.plugins?.mmd_tools;el('secondaryHint').textContent=p?.ready?'使用远坂凛原始 PMX 的作者物理，并稳定发束。':(p?.reason||'插件未配置。')+(p?.fallback?(p.fallback_mode==='surface'?' 将回退到内置衣物求解器。':' 将回退到内置头发求解，衣服沿用输入。'):' 请检查本机插件配置。');}
 for(const id of ['secondaryProfile','secondarySource','secondaryMode'])el(id).disabled=state.busy||running;
 const job=state.selected;show('secondaryCancel',!!job&&active.has(job.status));show('secondaryResume',!!job&&['failed','cancelled','interrupted'].includes(job.status));
 const canExport=job&&job.kind!=='export'&&['completed','quality_rejected'].includes(job.status)&&job.result;show('secondaryExportArea',!!canExport);
 show('secondaryPreviewRefresh',!!canExport);el('secondaryPreviewRefresh').disabled=running||state.busy;
 el('secondaryExport').disabled=!canExport||running||state.busy||!el('secondaryExperimental').checked||!document.querySelector('input[name=secondaryFormat]:checked');
 for(const id of ['secondaryCancel','secondaryResume'])el(id).disabled=state.busy;
}
function sources(){options('secondarySource',(state.catalog?.sources||[]).filter(p=>p.profile===el('secondaryProfile').value&&!p.export_only),'先生成远坂凛角色工程');controls();}
async function refreshCatalog(){state.catalog=await api('/catalog');options('secondaryProfile',state.catalog.profiles,'暂无角色配置');sources();}
function select(job){state.selected=job;el('secondaryExperimental').checked=false;render();}
function render(){const signature=JSON.stringify([state.selected?.id,state.runs.map(r=>[r.id,r.status])]);if(signature!==state.listSignature){state.listSignature=signature;el('secondaryRuns').replaceChildren();for(const job of state.runs){const button=document.createElement('button');button.className='secondary-run'+(job.id===state.selected?.id?' active':'');button.textContent=`${job.kind==='export'?'导出':job.kind==='mmd_tools'?'MMD 插件':job.kind==='hair'?'头发':'衣服与碰撞'} · ${labels[job.status]||job.status} · ${new Date(job.created*1000).toLocaleTimeString('zh-CN')}`;button.onclick=()=>select(job);el('secondaryRuns').append(button);}if(!state.runs.length)el('secondaryRuns').textContent='还没有衣发任务';}
 const job=state.selected;if(!job){controls();return;}
 el('secondaryStatus').textContent=`${labels[job.status]||job.status} · ${job.source.label}`;
 el('secondaryQuality').textContent=job.kind==='export'?`格式验证：${job.format_validated?'通过':'尚未完成'}；完整碰撞：${job.physics_quality==='rejected'?'未通过':'未验收'}`:`主动作保护：${job.protection_passed?'通过':'等待验证'}；完整碰撞：${job.physics_quality==='rejected'?'未通过':job.physics_quality==='accepted'?'本次门限通过':'未验收'}`;
 if(job.fallback_reason)el('secondaryQuality').textContent+=(job.kind==='hair'?'；已回退内置头发求解，衣服沿用输入：':'；已回退内置衣物求解：')+job.fallback_reason;
 if(job.selection_gate_passed&&job.physics_quality==='rejected')el('secondaryQuality').textContent+='；回退候选满足本次统计门限，残余穿插仍在';
 if(job.selection_mode==='cloth_component_backoff'&&job.cloth_components){const counts=job.cloth_components;el('secondaryQuality').textContent+=`。分布片回退：${counts.accepted_components}/${counts.accepted_components+counts.rejected_components} 布片保留物理，头发沿用输入`;}else if(job.selection_mode==='source_hair')el('secondaryQuality').textContent+='。本次头发沿用输入，衣物使用实验候选';
 el('secondaryStages').replaceChildren();for(const stage of job.stages||[]){const li=document.createElement('li');li.textContent=`${stageLabels[stage.name]||stage.name} · ${stage.last_access==='working_reuse'?'复用本次已完成阶段':stage.last_access==='cache_hit'?'复用已验证缓存':labels[stage.status]||stage.status}${stage.frame_progress?' · '+stage.frame_progress.frames+'/'+stage.frame_progress.total+' 帧':''}`;el('secondaryStages').append(li);}
 show('secondaryError',!!job.error);if(job.error)el('secondaryError').textContent=job.error;
 const files=job.files||[],before=files.filter(isPreviewBefore).sort((a,b)=>a.relative.localeCompare(b.relative));
 show('secondaryPreview',before.length>0);el('secondaryFrame').max=Math.max(0,before.length-1);if(+el('secondaryFrame').value>=before.length)el('secondaryFrame').value=0;renderFrame();
 el('secondaryOutputs').replaceChildren();const links=job.kind==='export'&&job.format_validated?[{name:'下载完整导出包（含缓存）',url:`/api/secondary/runs/${job.id}/download`}]:files.filter(f=>f.selected_result&&['physics-validation.json','surface-audit.json'].includes(f.name));
 for(const f of links){const a=document.createElement('a');a.href=f.url;a.textContent=f.name==='physics-validation.json'?'下载动作保持验证报告':f.name==='surface-audit.json'?'下载真实表面审核报告':f.name;a.download='';el('secondaryOutputs').append(a);}const log=document.createElement('a');log.href=`/api/secondary/runs/${job.id}/log`;log.textContent='查看处理日志';log.target='_blank';log.rel='noopener';el('secondaryOutputs').append(log);controls();}
function isPreviewBefore(f){return state.selected?.preview_relative&&f.relative.startsWith(state.selected.preview_relative+'/before/')&&f.name.endsWith('.png');}
async function renderFrame(){const files=state.selected?.files||[],before=files.filter(isPreviewBefore).sort((a,b)=>a.relative.localeCompare(b.relative));if(!before.length)return;const selected=before[+el('secondaryFrame').value||0],after=files.find(f=>f.relative===selected.relative.replace('/before/','/after/'));if(!after)return;
 const frame=parseInt(selected.name),target=state.selected.id+selected.id+after.id;if(state.frameTarget===target)return;state.frameTarget=target;el('secondaryFrameLabel').textContent=`正在载入第 ${frame} 帧…`;
 try{await Promise.all([selected,after].map(f=>new Promise((resolve,reject)=>{const img=new Image();img.onload=resolve;img.onerror=()=>reject(Error('对照图片加载失败，请刷新任务。'));img.src=f.url;})));
  if(state.frameTarget!==target)return;for(const [id,f] of [['secondaryBefore',selected],['secondaryAfter',after]]){el(id).src=f.url;el(id).dataset.frame=String(frame);}el('secondaryFrameLabel').textContent=`实际第 ${frame} 帧 · ${before.length} 组关键帧`;
 }catch(e){if(state.frameTarget===target){state.frameTarget=null;error(e.message);}}
}
async function refresh(){const result=await api('/runs');state.runs=result.runs;if(state.selected)state.selected=state.runs.find(r=>r.id===state.selected.id)||state.selected;else if(state.runs.length)state.selected=state.runs[0];render();}
async function action(fn){if(state.busy)return;state.busy=true;controls();show('secondaryError',false);try{const job=await fn();state.selected=job;await refresh();}catch(e){error(e.message);}finally{state.busy=false;controls();}}
el('secondaryEnabled').onchange=controls;el('secondaryProfile').onchange=sources;el('secondarySource').onchange=controls;el('secondaryMode').onchange=controls;el('secondaryFrame').oninput=renderFrame;el('secondaryExperimental').onchange=controls;
document.querySelectorAll('input[name=secondaryFormat]').forEach(c=>c.onchange=controls);
el('secondaryRefresh').onclick=async()=>{try{await refreshCatalog();await refresh();}catch(e){error(e.message);}};
el('secondaryStart').onclick=()=>action(()=>api('/runs',{source_id:el('secondarySource').value,profile:el('secondaryProfile').value,enabled:el('secondaryEnabled').checked,mode:el('secondaryMode').value,preview:true}));
el('secondaryCancel').onclick=()=>action(()=>api(`/runs/${state.selected.id}/cancel`,{}));el('secondaryResume').onclick=()=>action(()=>api(`/runs/${state.selected.id}/resume`,{}));
el('secondaryPreviewRefresh').onclick=()=>action(()=>api(`/runs/${state.selected.id}/refresh-preview`,{}));
el('secondaryExport').onclick=()=>action(()=>api(`/runs/${state.selected.id}/export`,{formats:[...document.querySelectorAll('input[name=secondaryFormat]:checked')].map(c=>c.value),experimental:el('secondaryExperimental').checked}));
try{await refreshCatalog();await refresh();}catch(e){error(e.message);}
setInterval(async()=>{if(state.polling||state.busy||document.hidden)return;state.polling=true;try{await refresh();if(!state.runs.some(r=>active.has(r.status))&&!el('secondaryProfile').matches(':focus')&&!el('secondarySource').matches(':focus')&&Date.now()-(state.catalogTime||0)>12000){await refreshCatalog();state.catalogTime=Date.now();}}catch(e){error(e.message);}finally{state.polling=false;}},2000);
