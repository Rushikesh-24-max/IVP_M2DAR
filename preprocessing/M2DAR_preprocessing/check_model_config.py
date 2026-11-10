from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'validation_tools'))
import yaml
old=yaml.safe_load((ROOT/'source'/'MVITV2_S_16x4_448_original.yaml').read_text())
path=ROOT/'MVITV2_S_16x4_448_prepared.yaml';new=yaml.safe_load(path.read_text())
def flatten(obj,prefix=''):
    out={}
    for k,v in obj.items():
        key=prefix+k
        if isinstance(v,dict):out.update(flatten(v,key+'.'))
        else:out[key]=v
    return out
a,b=flatten(old),flatten(new)
changed={k for k in set(a)|set(b) if a.get(k)!=b.get(k)}
assert changed=={'DATA.PATH_PREFIX','DATA.PATH_TO_DATA_DIR','DATA.PATH_LABEL_SEPARATOR','OUTPUT_DIR'},changed
assert new['MODEL']['NUM_CLASSES']==16 and new['DATA']['NUM_FRAMES']==16 and new['DATA']['SAMPLING_RATE']==4
assert new['TRAIN']['AUTO_RESUME'] is False and 'aicity_200ep' not in new['TRAIN']['CHECKPOINT_FILE_PATH']
result=dict(status='syntax_and_source_comparison_passed',yaml_parser='PyYAML '+yaml.__version__,
    source_commit='8fc6a3f76fc47207c4fa072dff7259f0debb5472',source_git_blob='c3e8a2cbb7b710f4568cf015240e4e8400cd74f1',
    changed_fields=sorted(changed),derived_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    model_optimizer_hyperparameters_unchanged=True,model_execution_tested=False,checkpoint_bytes_audited=False)
(ROOT/'model_config_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
