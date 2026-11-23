import pytest
import numpy as np
from motionbridge.secondary_attachment_regions import compile_regions
def config():return dict(fixed=[10],driven=[11],free=[12],evidence='reviewed test',pin_weights=dict(fixed=1,driven=.35,free=0))
def test_explicit_regions():np.testing.assert_array_equal(compile_regions(config(),[10,11,12],[[0,1,2]]),[1,.35,0])
def test_overlap_rejected():
    c=config();c['free']=[11,12]
    with pytest.raises(ValueError):compile_regions(c,[10,11,12],[[0,1,2]])
def test_missing_point_rejected():
    with pytest.raises(ValueError):compile_regions(config(),[10,11,12,13],[[0,1,2]])
def test_evidence_required():
    c=config();c['evidence']=''
    with pytest.raises(ValueError):compile_regions(c,[10,11,12],[[0,1,2]])
def test_free_component_rejected():
    c=config();c['free']=[12,13,14,15]
    with pytest.raises(ValueError,match='Unanchored'):compile_regions(c,[10,11,12,13,14,15],[[0,1,2],[3,4,5]])
def test_strengths_cannot_silently_disable_attachment():
    c=config();c['pin_weights']['fixed']=.5
    with pytest.raises(ValueError):compile_regions(c,[10,11,12],[[0,1,2]])
