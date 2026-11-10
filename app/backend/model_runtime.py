"""Bind the service to the already-tested experiment without changing its source."""
from pathlib import Path
import sys
from protocol import CHECKPOINT,SOURCE_LOCK,DATASET,read,sha256

def config(training_root,bundle,checkpoint):
 sys.path.insert(0,str(Path(training_root)/'scripts'))
 from common import source_integrity,verify_bundle,read_cfg,seed_runtime
 if sha256(checkpoint)!=CHECKPOINT:raise ValueError('Checkpoint SHA256 does not match frozen epoch 170')
 if source_integrity()!=SOURCE_LOCK:raise ValueError('Unexpected M2DAR source lock')
 manifest=verify_bundle(bundle,check_files=False)
 if manifest['dataset_fingerprint']!=DATASET or read(Path(checkpoint).parent/'run.json')['dataset_fingerprint']!=DATASET:raise ValueError('Training bundle provenance mismatch')
 classes=read(Path(__file__).resolve().parents[1]/'dist/data/classes.json')
 expected=['Normal Forward Driving','Drinking','Phone Call(right)','Phone Call(left)','Eating','Text (Right)','Text (Left)','Reaching behind','Adjust control panel','Pick up from floor (Driver)','Pick up from floor (Passenger)','Talk to passenger at the right','Talk to passenger at backseat','yawning','Hand on head','Singing or dancing with music']
 if classes!={str(i):s for i,s in enumerate(expected)}:raise ValueError('Class mapping changed')
 seed_runtime(42);return read_cfg(Path(bundle)/'baseline.yaml'),classes

def load_model(cfg,checkpoint,cpu=False):
 from common import load_torch_checkpoint,require_shapes
 from slowfast.models import build_model
 if cpu:cfg.NUM_GPUS=0
 state=load_torch_checkpoint(checkpoint)
 if 'preparation' in state:raise ValueError('Initializer is not a trained model')
 model=build_model(cfg);require_shapes(model.state_dict(),state['model_state']);model.load_state_dict(state['model_state'],strict=True);model.eval();return model

def readiness(training_root,bundle,checkpoint):
 import torch,av,decord
 cfg,_=config(training_root,bundle,checkpoint);model=load_model(cfg,checkpoint,cpu=True);del model
 if not torch.cuda.is_available():raise RuntimeError('CUDA GPU is unavailable')
 free,total=torch.cuda.mem_get_info(0)
 if free<.20*total+1024**3:raise RuntimeError('GPU has insufficient free memory for the allocator cap and context headroom')
 return dict(verified=True,checkpoint_sha256=CHECKPOINT,source_lock_sha256=SOURCE_LOCK,dataset_fingerprint=DATASET,device=torch.cuda.get_device_name(0),free_memory_bytes=free)
