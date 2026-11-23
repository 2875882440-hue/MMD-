import copy
import numpy as np
import pytest
from test_secondary_coupled_surface import fixture
from motionbridge.secondary_coupled_surface import solve_coupled_surface
from motionbridge.secondary_coupled_contacts import sweep_aabb_pairs
from motionbridge.secondary_contact_stream import iter_aabb_blocks,StreamingSourceContacts,geometry_key,segment_weights
from motionbridge.secondary_joint_contacts import segment_closest


@pytest.mark.parametrize('size',[1,2,7,4096])
def test_block_pairs_exactly_equal_reference_across_boundaries(size):
    rng=np.random.default_rng(21);x=rng.uniform(-1,1,(150,3));tri=np.arange(150).reshape(-1,3)
    chunks=list(iter_aabb_blocks(x,tri,.002,size));pairs=np.concatenate(chunks)
    expected=sweep_aabb_pairs(x,tri,.002,100000)
    assert len(pairs)==len(set(map(tuple,pairs)))==len(expected)
    assert set(map(tuple,pairs))==set(expected)
    assert max(map(len,chunks))<=size


def test_more_than_old_budget_without_raising_resident_budget():
    # 451 identical boxes have 101475 unique pairs. This is enumeration only.
    tri=np.arange(451*3).reshape(-1,3);x=np.tile([[0,0,0],[1,0,0],[0,1,0]],(451,1))
    count=0;seen=set()
    for block in iter_aabb_blocks(x,tri,.001,127):
        assert len(block)<=127
        for p in map(tuple,block):assert p not in seen;seen.add(p)
        count+=len(block)
    assert count==451*450//2>100000


def test_tied_sort_keys_and_closed_boundary_pairs_once():
    x=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,.001],[1,0,.001],[0,1,.001]],float)
    pairs=list(iter_aabb_blocks(x,np.array([[0,1,2],[3,4,5]]),.001,1))
    assert np.concatenate(pairs).tolist()==[[0,1]]


def test_float32_inputs_use_float64_for_both_reference_and_scan():
    rng=np.random.default_rng(131);x=rng.uniform(-.01,.01,(90,3)).astype(np.float32);tri=np.arange(90).reshape(-1,3)
    expected=sweep_aabb_pairs(x.astype(np.float64),tri,.002,100000)
    actual=np.concatenate(list(iter_aabb_blocks(x,tri,.002,3)))
    assert set(map(tuple,actual))==set(expected)


@pytest.mark.parametrize('size',[0,4097,True])
def test_no_budget_expansion(size):
    with pytest.raises(ValueError):list(iter_aabb_blocks(np.zeros((3,3)),np.array([[0,1,2]]),.001,size))


def test_batched_edges_match_scalar_nearest_features():
    rng=np.random.default_rng(8);a,b,c,d=rng.normal(size=(4,30,3));s,t,bad=segment_weights(a,b,c,d)
    assert not bad.any()
    for i in range(30):
        _,_,p,q=segment_closest(a[i],b[i],c[i],d[i])
        np.testing.assert_allclose(np.linalg.norm(a[i]+s[i]*(b[i]-a[i])-c[i]-t[i]*(d[i]-c[i])),np.linalg.norm(p-q),atol=1e-12)


def contacts(f,**kw):return StreamingSourceContacts(f,gap=.002,floor_z=-.1,block_size=1,**kw)


def test_blocked_scan_still_finishes_all_pairs_and_channels():
    f,_=fixture();x=f.source.copy();x[7:]=[[-.1,.05,-.1],[.2,.05,-.1],[.05,.05,.1]]
    f.source=x;f.x=x.copy();c=contacts(f);rows=[]
    r=c.scan(x,on_block=lambda s,p:rows.append((s,p.copy())))
    assert 'stop_missing_authored_exit_side' in r['blocking_reasons']
    assert r['audit']['scan_complete'] and r['audit']['proper_crossings']>0
    assert sum(len(p) for _,p in rows)==len(sweep_aabb_pairs(x,f.mapping.triangles,.002,100000))


def test_immutable_overlap_retained_but_fixed_motion_change_stops():
    f,_=fixture();f.weights[:4]=0;f.metrics['tab'][:]=0
    x=f.source.copy();x[7:]=[[-.1,.05,-.1],[.2,.05,-.1],[.05,.05,.1]]
    f.source=x;f.x=x.copy();c=contacts(f);r=c.scan(x)
    assert r['audit']['exit_decisions']['retain_immutable_baseline']>0
    assert not r['blocking_reason']
    changed=x.copy();changed[7,0]+=.001
    assert 'stop_fixed_source_changed' in c.scan(changed)['blocking_reasons']


def test_streamed_corrections_and_final_global_recheck():
    f,_=fixture();f.x[:4,2]-=.019;f.states['tab'][1][2]-=.019;c=contacts(f)
    out,r=solve_coupled_surface(f,c)
    assert r['candidate_committed'],r
    assert out[:7,2].min()>=.002-2e-6
    assert r['candidate_shape_audit']['attachment_residual_m']<=2e-6
    assert r['candidate_contact_audit']['scan_complete']
    assert c.scan(out)['audit']['maximum_deficit_m']<=2e-6


def test_block_size_does_not_change_scan_totals():
    f,_=fixture();f.x[:4,2]-=.019;f.states['tab'][1][2]-=.019;x=f.positions()
    a=contacts(f).scan(x)
    b=StreamingSourceContacts(f,gap=.002,floor_z=-.1,block_size=4096).scan(x)
    for k in ('candidate_pairs','proper_crossings','proximity_features','constraint_count','maximum_deficit_m','bonded_features','exit_decisions'):
        assert a['audit'][k]==b['audit'][k]


def test_late_block_rejection_rolls_back_earlier_corrections():
    f,_=fixture();f.x[:4,2]-=.019;f.states['tab'][1][2]-=.019;c=contacts(f);original=f.positions()
    real=c.blocks
    def injected(x):
        yield from real(x)
        yield {'audit':{'pairs':0,'constraint_count':0,'maximum_deficit_m':0},'pairs':np.empty((0,2),int),
               'constraints':[],'blocking_reasons':['stop_test_late_conflict']}
    c.blocks=injected;r=c.project_all(f)
    assert r['blocking_reason'];np.testing.assert_array_equal(f.positions(),original)


def policy(f,**changes):
    rule={'moving_face':0,'target_face':3,'side':1,'max_depth_m':.002,'evidence':'synthetic explicit upper side'}
    rule.update(changes)
    return {'schema_version':1,'geometry_sha256':geometry_key(f.mapping),'rules':[rule]}


def test_authored_exit_generates_only_declared_side_and_is_bounded():
    f,_=fixture();c=StreamingSourceContacts(f,gap=.0001,floor_z=-.1,exit_policy=policy(f))
    x=f.source.copy();x[0,2]=-.0005
    decision,rows=c._exit(0,3,x)
    assert decision=='push_authored_side' and rows
    assert all(row['gradients'][0][2]>0 for row in rows)
    x[0,2]=-.003;assert c._exit(0,3,x)[0]=='stop_authored_exit_depth_exceeded'


def test_signed_rule_persists_after_crossing_disappears():
    f,_=fixture();c=StreamingSourceContacts(f,gap=.0001,floor_z=-.1,exit_policy=policy(f))
    x=f.source.copy();x[:4,2]=-.0005
    tail=[b for b in c.blocks(x) if b['audit'].get('kind')=='authored_exit']
    assert len(tail)==1 and len(tail[0]['constraints'])==3
    assert all(row['gradients'][0][2]>0 for row in tail[0]['constraints'])


@pytest.mark.parametrize('case',['fingerprint','side','depth','evidence','moving_target'])
def test_unreviewed_or_unsupported_exit_rejected(case):
    f,_=fixture();p=policy(f)
    if case=='fingerprint':p['geometry_sha256']='bad'
    elif case=='side':p['rules'][0]['side']=0
    elif case=='depth':p['rules'][0]['max_depth_m']=.003
    elif case=='evidence':p['rules'][0]['evidence']=''
    else:p['rules'][0].update(moving_face=3,target_face=0)
    with pytest.raises(ValueError):contacts(f,exit_policy=p)
