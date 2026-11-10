"""Final train/val, annotation-provenance and model-input smoke checks."""
from pathlib import Path
import json,hashlib,sys
OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(OUT));sys.path.insert(0,r'F:\A1_analysis\tools')
import numpy as np
from training_dataset import SynchronizedActionDataset

def run():
    ready=json.loads((OUT/'READY.json').read_text());audit=json.loads((OUT/'annotation_audit.json').read_text())
    changed=[]
    for path,expected in audit['source_annotation_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:changed.append(path)
    assert not changed,changed
    train=SynchronizedActionDataset(OUT,'train');val=SynchronizedActionDataset(OUT,'val')
    assert not ({g['driver'] for g in train.groups}&{g['driver'] for g in val.groups})
    assert set(g['label'] for g in train.groups)==set(range(16))
    checks=[]
    for split,ds in [('train',train),('val',val)]:
        for index in sorted({0,len(ds)-1,next((i for i,g in enumerate(ds.groups) if g.get('temporal_padding_required')),0)}):
            item=ds[index];video=item['video'];assert video.shape==(3,3,16,448,448),video.shape
            assert video.dtype==np.float32 and np.isfinite(video).all()
            assert item['frame_indices'].min()>=0 and item['frame_indices'].max()<ds.groups[index]['frame_count']
            checks.append(dict(split=split,group_id=item['group_id'],shape=list(video.shape),dtype=str(video.dtype),label=item['label'],short_clip=bool(item['temporal_padding_required']),passed=True))
        parsed=[]
        for line in (OUT/'m2dar'/f'{split}.csv').read_text().splitlines():
            path,label=line.rsplit(' ',1);assert (OUT/path).is_file();assert 0<=int(label)<16;parsed.append((path,int(label)))
        assert len(parsed)==3*len(ds)
    # Verify original media size/mtime against acquisition metadata.
    changed_media=[]
    source_features=list((OUT/'dense_motion').glob('*.npz'));assert len(source_features)==len(json.loads((OUT/'source_videos.json').read_text()))
    for p in source_features:
        z=np.load(p,allow_pickle=False);meta=json.loads(str(z['metadata']));stat=Path(meta['source_path']).stat()
        if stat.st_size!=meta['source_size_bytes'] or stat.st_mtime_ns!=meta['source_mtime_ns']:changed_media.append(meta['source_path'])
    assert not changed_media,changed_media
    result=dict(annotation_hashes_unchanged=True,media_size_and_mtime_unchanged=True,driver_leakage=False,train_csv_entries=3*len(train),val_csv_entries=3*len(val),model_input_checks=checks,model_training_executed=False)
    (OUT/'training_smoke_test.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':run()
