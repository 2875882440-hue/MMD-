"""Read-only regression evidence review. Does not run or approve physics.

Native short/long runs must use an accepted model adapter, which the current
production pipeline does not have. Archived measurements remain diagnostics;
their integrity and gates can be checked without recomputing a failed run.
"""
import hashlib
import json
import math
from pathlib import Path
from .secondary_guard import attachment_repair_gate, prewarm_stop_gate
from .secondary_segmentation import require_review

CHANNELS=('body_cloth','body_hair','cloth_hair','accessory_body','accessory_cloth','accessory_hair')
LABELS=dict(zip(CHANNELS,('身体/衣料','身体/头发','衣料/头发','饰品/身体','饰品/衣料','饰品/头发')))


def fingerprint(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def regression_readiness(plan):
    blockers=[]
    try:require_review(plan)
    except ValueError as error:blockers.append(str(error))
    rules=plan.get('secondary_contact_rules',{})
    blockers.extend(rules.get('blockers',[]))
    # Even a legacy/no-policy plan is not proof of the new native adapter.
    blockers.append('native_surface_joint_and_contact_adapter_pending' if plan.get('source_surface_map') else 'new_native_character_adapter_not_implemented')
    return {'native_execution_allowed':False,'blockers':list(dict.fromkeys(blockers)),
        'segmentation':plan.get('surface_segmentation',{}).get('review_gate'),
        'source_surface_map':plan.get('source_surface_map'),
        'joint_solver':rules.get('joint_solver'),'prewarm':rules.get('prewarm'),
        'sequence':['model_mapping_and_native_adapter','short_90_frames_179_samples',
                    'long_900_frames_1799_samples','same_camera_visual_review'],
        'next_action':'complete_native_semantic_attachment_and_contact_mapping',
        'short_run_started':False,'long_run_started':False,'video_generated':False}


def _number(value,name,integer=False):
    if type(value) not in ((int,) if integer else (int,float)) or not math.isfinite(value) or value<0:
        raise ValueError('Invalid '+name)
    return value


def _validate(meta,audit,cloth,surface):
    start=meta['frame_start'];end=meta['frame_end'];count=meta['frames']
    if any(type(x) is not int for x in (start,end,count)) or count<1 or end-start+1!=count:raise ValueError('Invalid source frame range')
    if _number(meta['fps'],'fps')<=0:raise ValueError('Invalid source fps')
    expected=[start+i*.5 for i in range(2*count-1)]
    if audit.get('frame_stride')!=1:raise ValueError('Sparse sampling cannot replace integer and half frames')
    if audit.get('attachment_audit_required') is not True:raise ValueError('Missing attachment audit')
    if surface['source_sha256']!=meta['source_sha256']:raise ValueError('Source fingerprint mismatch')
    if surface['frames']!=count or cloth['frames']!=count or cloth['fps']!=meta['fps']:raise ValueError('Mixed clip or frame count')
    for side in ('before','after'):
        row=audit[side];samples=_number(row['samples'],'samples',True)
        if samples!=len(expected) or [f['frame'] for f in row['per_frame']]!=expected:raise ValueError('Incomplete or misaligned sample grid')
        for channel in CHANNELS:
            values=[_number(f[channel],channel,True) for f in row['per_frame']]
            for suffix,actual in (('crossing_pairs',sum(values)),('crossing_frames',sum(v>0 for v in values)),('max_pairs',max(values))):
                if _number(row[channel+'_'+suffix],channel,True)!=actual:raise ValueError('Per-frame totals disagree: '+channel+' '+side)
        depths=[_number(f['dynamic_floor_depth_m'],'floor depth') for f in row['per_frame']]
        if _number(row['dynamic_floor_frames_over_1mm'],'floor frames',True)!=sum(d>.001 for d in depths):raise ValueError('Floor frame count mismatch')
        if abs(_number(row['dynamic_floor_max_depth_m'],'floor max')-max(depths))>1e-12:raise ValueError('Floor maximum mismatch')
        _number(row['dynamic_floor_vertex_samples_over_1mm'],'floor vertices',True)
        if row['dynamic_floor_vertex_samples_over_1mm']<row['dynamic_floor_frames_over_1mm']:raise ValueError('Invalid floor vertex count')
        p=audit['protection'][side]
        if p['samples']!=len(expected) or p['bones']!=48 or p['matrix_sha256']!=meta['protected_matrix_sha256']:
            raise ValueError('Protected bone evidence mismatch')
        if p['all_source_actions_unchanged'] is not True or p['original_skin_and_shape_keys_unchanged'] is not True:raise ValueError('Original action or skin/morph changed')
        if p['source_actions']!=len(meta['original_actions']):raise ValueError('Source action coverage mismatch')
    if audit.get('protected_body_boot_max_world_change_m')!=0 or meta.get('shape_keys')!=81:raise ValueError('Rin body/shape evidence mismatch')
    if audit.get('protection_passed') is not True:raise ValueError('Protection failed')
    return expected


def review_measurements(meta,audit,cloth,surface):
    """Validate an existing Rin report; never promote it to a new native run."""
    result={'evidence_valid':False,'numeric_regression_passed':False,'publication_accepted':False,
        'scope':'archived_report_review_not_new_geometry_measurement','reasons':[],
        'channels':{},'new_simulation':False,'visual_verified':False}
    try:
        grid=_validate(meta,audit,cloth,surface)
        result.update(evidence_valid=True,frames=meta['frames'],samples=len(grid),fps=meta['fps'])
        contact=attachment_repair_gate(audit)
        warm=prewarm_stop_gate(cloth,{'prewarm_p95_limit_m_s':.02,'prewarm_max_limit_m_s':.1})
        raw_warm=cloth['prewarm_convergence']
        if raw_warm.get('trailing_frames',0)/meta['fps']<.5:warm['accepted']=False;warm['rejection_reasons'].append('warmup_window_too_short')
        shape=surface['garment_deformation_gate'];limit=_number(shape['relative_limit'],'edge limit')
        error=_number(shape['maximum_additional_relative_edge_error'],'edge error')
        tolerance=_number(shape['absolute_tolerance_m'],'edge absolute tolerance')
        shape_pass=(shape.get('passed') is True and 0<limit<=.25 and error<=limit and tolerance<=5e-6
                    and _number(shape['frames_over_budget'],'shape frames',True)==0
                    and _number(shape['edge_samples_over_budget'],'shape samples',True)==0)
        result['reasons'].extend(contact['rejection_reasons']+warm['rejection_reasons'])
        if not shape_pass:result['reasons'].append('deformation_gate_failed_or_relaxed')
        result.update(contact_gate=contact,prewarm_gate=warm,deformation_gate_passed=shape_pass,
                      prewarm_measurements=raw_warm)
        for channel in CHANNELS:
            before=audit['before'][channel+'_crossing_pairs'];after=audit['after'][channel+'_crossing_pairs']
            rows=[{'frame':a['frame'],'clip_seconds':(a['frame']-meta['frame_start'])/meta['fps'],
                   'before':b[channel],'after':a[channel],'increase':a[channel]-b[channel]}
                  for b,a in zip(audit['before']['per_frame'],audit['after']['per_frame'])]
            worse=[r for r in rows if r['increase']>0]
            result['channels'][channel]={'before_pairs':before,'after_pairs':after,'delta_pairs':after-before,
                'percent_change':(after-before)/before*100 if before else None,
                'worsened_samples':len(worse),'largest_increases':sorted(worse,key=lambda r:(-r['increase'],r['frame']))[:5]}
        result['numeric_regression_passed']=contact['accepted'] and warm['accepted'] and shape_pass
    except (KeyError,TypeError,ValueError,OverflowError) as error:
        result['evidence_valid']=False;result['numeric_regression_passed']=False
        result['reasons'].append('invalid_or_incomplete_evidence: '+str(error))
    result['reasons']=list(dict.fromkeys(result['reasons']))
    return result


def review_archived_case(root,case,pair):
    root=Path(root).resolve();case=Path(case).resolve();pair=Path(pair).resolve()
    if not case.is_relative_to(root/'output/secondary_motion') or not pair.is_relative_to(case):raise ValueError('Use existing evidence within the development project')
    paths={'input':case/'input.json','audit':pair/'surface-audit.json',
           'cloth':case/'cloth/cloth-validation.json','surface':pair/'surface-validation.json'}
    loaded={k:json.loads(p.read_text(encoding='utf8')) for k,p in paths.items()}
    result=review_measurements(loaded['input'],loaded['audit'],loaded['cloth'],loaded['surface'])
    result.update(case=str(case),pair=str(pair),evidence_sha256={str(p):fingerprint(p) for p in paths.values()})
    source=Path(loaded['input']['source']).resolve()
    if not source.is_relative_to(root) or not source.is_file() or fingerprint(source)!=loaded['input']['source_sha256']:
        result['evidence_valid']=False;result['numeric_regression_passed']=False;result['reasons'].append('source_file_missing_or_changed')
    # Reuse of the original before is legitimate; reuse of a worse after is not.
    previous=loaded['audit'].get('baseline_reused_from')
    if previous:
        previous=Path(previous).resolve()
        if not previous.is_relative_to(case):raise ValueError('Baseline must belong to the same capture')
        old=json.loads(previous.read_text(encoding='utf8'));old_surface=previous.parent/'surface-validation.json'
        origin=json.loads(old_surface.read_text(encoding='utf8'))
        same=(old['before']==loaded['audit']['before'] and old['protection']['before']==loaded['audit']['protection']['before']
              and origin['source_sha256']==loaded['input']['source_sha256'])
        if not same:result['evidence_valid']=False;result['numeric_regression_passed']=False;result['reasons'].append('reused_before_mismatch')
        result['evidence_sha256'].update({str(p):fingerprint(p) for p in (previous,old_surface)})
    result['source_file_verified']=not ('source_file_missing_or_changed' in result['reasons'])
    return result


def markdown_report(report):
    lines=['# 第六步回归前置检查与存量报告复核','',
        '**当前未执行新版短片、长片或对照视频。存量结果不能作为新版验收。**','',
        '## 当前阻塞','']
    labels={'semantic_segmentation_not_accepted':'真实表面分类尚未审核通过',
            'model_attachments_and_surface_mapping_missing':'模型连接点与物理表面映射缺失',
            'production_joint_solver_and_native_mapping_not_accepted':'完整联合求解与原生模型映射未验收',
            'new_native_character_adapter_not_implemented':'尚无将新核心用于真实角色的原生适配',
            'cloth_prewarm_unsettled_or_unverified':'衣物预热不稳定或未验证'}
    for c,label in LABELS.items():labels[c+'_crossing_pairs']=label+'交叉总数恶化'
    lines.extend('- '+labels.get(str(x),str(x)) for x in report['readiness']['blockers'])
    for name,case in report['historical_cases'].items():
        lines+=['','## '+name,'',f"报告数据核对：{'通过' if case['evidence_valid'] else '未通过'}；数值门槛：{'通过' if case['numeric_regression_passed'] else '未通过'}。",
            '','| 通道 | before | after | 变化 | 恶化采样数 |','| --- | ---: | ---: | ---: | ---: |']
        for channel,row in case['channels'].items():
            delta='基线为0' if row['percent_change'] is None else f"{row['percent_change']:+.2f}%"
            lines.append(f"| {LABELS[channel]} | {row['before_pairs']} | {row['after_pairs']} | {delta} | {row['worsened_samples']} |")
        lines+=['','拒绝原因：'+'；'.join(labels.get(r,r) for r in case['reasons']),
            '','高风险采样（时间相对本测试片段起点，不等于原始长视频时间）：']
        for channel in ('accessory_body','accessory_cloth','accessory_hair'):
            rows=case['channels'].get(channel,{}).get('largest_increases',[])
            lines.append('- '+LABELS[channel]+'：'+('；'.join(f"帧{r['frame']:g} / {r['clip_seconds']:.3f}s，增加{r['increase']}对" for r in rows) or '未记录恶化采样'))
        warm=case.get('prewarm_measurements',{})
        lines+=['',f"存量预热最大速度：{warm.get('maximum_vertex_speed_m_s')} m/s；P95：{warm.get('maximum_frame_p95_m_s')} m/s。"]
    lines+=['','## 放行顺序','',
        '完成真实模型分类/绑定和原生适配 → 90帧/179采样短片 → 900帧/1799采样长片 → 同相机、完整取景视频与人工视觉检查。',
        '短片失败不得启动长片；先前已存在的长片报告可以只读诊断。任何一步失败不自动试参。',
        '饰品三个通道分别不恶化、衣物/身体严格改善，且主动作、预热、形变和地面全部通过。',
        '合计交叉数不等于逐帧零穿模或动作准确率；数值通过也不能代替同相机视觉验收。',
        '下一项具体工作：完成Rin真实语义分割、连接点、主副面与接触域映射，再补齐原生求解适配。']
    return '\n'.join(lines)+'\n'
