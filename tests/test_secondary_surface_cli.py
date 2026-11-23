import importlib.util
from pathlib import Path
import pytest

@pytest.fixture
def cli(tmp_path,monkeypatch):
 path=Path(__file__).resolve().parents[1]/'scripts/simulate_secondary_surface.py'
 spec=importlib.util.spec_from_file_location('surface_cli_guard',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 monkeypatch.setattr(module,'ROOT',tmp_path)
 scene=tmp_path/'scene.blend';scene.write_bytes(b'preserve source')
 return module,scene,tmp_path/'output/secondary_motion/new'

def test_surface_cli_requires_explicit_activation(cli):
 module,scene,out=cli
 with pytest.raises(ValueError):module.validate_request(scene,out,'rin',False)
 assert not out.exists() and scene.read_bytes()==b'preserve source'

def test_surface_cli_accepts_only_fresh_output(cli):
 module,scene,out=cli
 assert module.validate_request(scene,out,'rin',True)==(scene,out)
 out.mkdir(parents=True);(out/'keep.txt').write_text('old results')
 with pytest.raises(ValueError):module.validate_request(scene,out,'rin',True)
 assert (out/'keep.txt').read_text()=='old results'

@pytest.mark.parametrize('profile',['../rin','Rin','rin/../other'])
def test_surface_cli_rejects_profile_path_traversal(cli,profile):
 module,scene,out=cli
 with pytest.raises(ValueError):module.validate_request(scene,out,profile,True)

def test_surface_cli_rejects_external_input_and_output(cli,tmp_path):
 module,scene,out=cli
 with pytest.raises(ValueError):module.validate_request(scene,scene.parent/'outside','rin',True)
 with pytest.raises(ValueError):module.validate_request(scene.parent.parent/'scene.blend',out,'rin',True)
