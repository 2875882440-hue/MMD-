import numpy as np
import pytest
from test_secondary_coupled_surface import fixture
from test_secondary_contact_stream import policy
from motionbridge.secondary_contact_stream import StreamingSourceContacts

@pytest.mark.parametrize('distance,expected',[
    (-8.913845660306673e-5,'push_authored_side'),
    (-.002,'push_authored_side'),
    (-.002001,'stop_authored_exit_depth_exceeded'),
    (.0005,'push_authored_side')])
def test_depth_is_not_penetration_plus_body_gap(distance,expected):
    f,_=fixture();c=StreamingSourceContacts(f,gap=.001,body_gap=.002,floor_z=-.1,exit_policy=policy(f))
    x=f.source.copy();x[0,2]=distance;original=x.copy()
    decision,rows=c._exit(0,3,x)
    assert decision==expected
    if rows:
        row=next(r for r in rows if r['ids']==[0])
        assert row['value']==pytest.approx(distance-.002)
        assert row['offset']==pytest.approx(.002)
    np.testing.assert_array_equal(x,original)

def test_gap_changes_target_but_not_depth_admission():
    f,_=fixture();x=f.source.copy();x[0,2]=-.0015
    values=[]
    for gap in (.001,.002):
        c=StreamingSourceContacts(f,gap=.001,body_gap=gap,floor_z=-.1,exit_policy=policy(f))
        decision,rows=c._exit(0,3,x);assert decision=='push_authored_side'
        values.append(rows[0]['value'])
    assert values[1]-values[0]==pytest.approx(-.001)

def test_fixed_vertex_is_still_not_pushable():
    f,_=fixture();f.weights[0]=0
    c=StreamingSourceContacts(f,gap=.001,body_gap=.002,floor_z=-.1,exit_policy=policy(f))
    x=f.source.copy();x[0,2]=-.0001
    assert c._exit(0,3,x)[0]=='stop_fixed_vertex_blocks_exit'
