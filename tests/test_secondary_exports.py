import copy
import pytest
from motionbridge.secondary_export_jobs import format_key,validate_export_receipt

@pytest.fixture
def stamp():return {'source_sha256':'source-a','dependencies':{'cache.mdd':'hash-a'},'blender_version':'test','blender_build':'build','frame_start':1,'frame_end':900,'rigid_object_count':86}

def test_export_cache_is_invalidated_by_actual_source_dependencies_and_options(stamp):
 key=format_key(stamp,'alembic',{'samples':2},'code')
 changed=copy.deepcopy(stamp);changed['dependencies']['cache.mdd']='changed'
 assert format_key(changed,'alembic',{'samples':2},'code')!=key
 assert format_key(stamp,'alembic',{'samples':1},'code')!=key
 assert format_key(stamp,'alembic',{'samples':2},'new code')!=key
 assert format_key(stamp,'skeleton',{'samples':2},'code')!=key

def test_character_readback_depends_on_the_actual_skeleton_file(stamp):
 a=format_key(stamp,'fbx-character',{},'code',{'skeleton_key':'same','skeleton_file_sha256':'a'})
 b=format_key(stamp,'fbx-character',{},'code',{'skeleton_key':'same','skeleton_file_sha256':'b'})
 assert a!=b

def test_valid_format_receipt_does_not_certify_physics(stamp):
 report={'passed':True,'source_sha256':'source-a','sample_frames':1799,'source_file_unchanged':True,'physics_quality_approval':False}
 assert validate_export_receipt(report,stamp) is None
 report['physics_quality_approval']=True
 with pytest.raises(ValueError):validate_export_receipt(report,stamp)

@pytest.mark.parametrize('change',[{'source_sha256':'other-source'},{'sample_frames':23},{'passed':False},{'source_file_unchanged':False}])
def test_partial_foreign_or_mutated_exports_cannot_enter_completed_cache(stamp,change):
 report={'passed':True,'source_sha256':'source-a','sample_frames':1799,'source_file_unchanged':True,**change}
 with pytest.raises(ValueError):validate_export_receipt(report,stamp)

def test_skip_rigid_is_only_allowed_when_source_really_has_none(stamp):
 report={'passed':True,'source_sha256':'source-a','skipped':True}
 with pytest.raises(ValueError):validate_export_receipt(report,stamp,kind='rigid')
 stamp['rigid_object_count']=0
 validate_export_receipt(report,stamp,kind='rigid')
 with pytest.raises(ValueError):validate_export_receipt(report,stamp,kind='alembic')
