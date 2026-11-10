"""Reconcile completed packages against manifests and original source footprints."""
from pathlib import Path
import json,hashlib,collections,sys,csv,re
ROOT=Path(r'F:\M2DAR_preprocessed');HERE=Path(__file__).resolve().parent

def footprint(path,size):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for pos in [0,max(0,size//2-32768),max(0,size-65536)]:f.seek(pos);h.update(f.read(65536))
    return h.hexdigest()

def run():
    config_record=json.loads((ROOT/'m2dar'/'model_config_validation.json').read_text())
    assert hashlib.sha256((ROOT/'m2dar'/'MVITV2_S_16x4_448_prepared.yaml').read_bytes()).hexdigest()==config_record['derived_file_sha256']
    original=json.loads(Path(r'F:\A1_analysis\mp4_inventory.json').read_text());sources=[];reports=[]
    for r in original:
        p=Path(r['path']);assert p.stat().st_size==r['size_bytes'];assert footprint(p,r['size_bytes'])==r['sampled_fingerprint_sha256']
        sources.append(p)
    source_packages=json.loads((ROOT/'source_packages.json').read_text());source_class_coverage=[]
    for package in source_packages:
        p=Path(package['path'])/'READY.json';assert hashlib.sha256(p.read_bytes()).hexdigest()==package['ready_sha256']
        buckets=collections.defaultdict(collections.Counter)
        for r in json.loads((Path(package['path'])/'annotations_normalized.json').read_text()):
            if r['status']=='valid':buckets[(r['driver'],r['session'],r['view'])][r['label']]+=1
        for key,counts in sorted(buckets.items()):
            source_class_coverage.append(dict(package=package['package'],driver=key[0],session=key[1],view=key[2],valid_class_counts=dict(counts),
                missing_classes=sorted(set(range(16))-set(counts)),repeated_classes={k:v for k,v in counts.items() if v>1}))
    for name in ['A1_2_preprocessed','A1_3_preprocessed','A1_4_preprocessed','A2_preprocessed']:
        root=Path('F:/')/name
        inventory=json.loads((root/'source_inventory_original.json').read_text())
        for r in inventory:
            p=Path(r['path']);st=p.stat();assert (st.st_size,st.st_mtime_ns)==(r['source_size_bytes'],r['source_mtime_ns'])
            assert footprint(p,st.st_size)==r['sampled_source_fingerprint_sha256'];sources.append(p)
        file='inference_groups.jsonl' if name=='A2_preprocessed' else 'training_manifest.jsonl'
        groups=[json.loads(x) for x in (root/file).read_text().splitlines() if x.strip()]
        media=set()
        for g in groups:
            assert g['validation']['status']=='ready'
            for v in g['views']:
                p=root/g['views'][v]['relative_output_path'];assert p.is_file();media.add(p.resolve())
                st=p.stat();recorded=g['validation']['output_file_stats'][v]
                assert (st.st_size,st.st_mtime_ns)==(recorded['size_bytes'],recorded['mtime_ns'])
                check=g['validation']['video_checks'][v];assert check['pts_checked'] and check['frames']==g['frame_count'] and check['corrupt_frames']==0
            for c in g['validation']['alignment_checks'].values():assert not c.get('failed')
        assert media=={p.resolve() for p in (root/'clips').rglob('*.mp4')}
        assert not list(root.rglob('*.partial.mp4')) and not list(root.rglob('*.partial.npz'))
        reports.append(dict(package=name,source_videos=len(inventory),triplets=len(groups),active_view_clips=len(media),prepared_bytes=sum(p.stat().st_size for p in media)))
    assert len(sources)==330 and len(set(sources))==330
    manifest=[json.loads(x) for x in (ROOT/'training_manifest.jsonl').read_text().splitlines() if x.strip()]
    label_files={};expected_entries={};provenance_checks=0
    def seconds(value):
        h,m,s=map(float,value.split(':'));return 3600*h+60*m+s
    for g in manifest:
        assert g['frame_count']==g['reference_end_frame']-g['reference_start_frame']
        for v,spec in g['views'].items():
            filename=spec['source_annotation_file']
            if filename not in label_files:
                with Path(filename).open(encoding='utf-8-sig',newline='') as f:label_files[filename]=list(csv.DictReader(f))
            row=label_files[filename][spec['source_annotation_row']-2]
            rawlabel=row.get('Label (Primary)',row.get('Label','')).strip();match=re.fullmatch(r'Class\s*(\d+)',rawlabel,re.I)
            assert match and int(match[1])==g['label']
            assert spec['source_start_s']>=seconds(row['Start Time'])+.25-1e-6
            assert spec['source_end_s']<=seconds(row['End Time'])-.25+1e-6
            assert abs(spec['source_start_s']-g['reference_start_s']-spec['offset_s'])<1e-6
            resolved=str((ROOT/spec['relative_output_path']).resolve())
            assert resolved not in expected_entries;expected_entries[resolved]=(g['split'],g['label']);provenance_checks+=1
    train=[];val=[]
    for split,target in [('train',train),('val',val)]:
        for row in (ROOT/'m2dar'/f'{split}.csv').read_text().splitlines():
            p,label=row.rsplit(' ',1);p=(ROOT/p).resolve();assert p.is_file() and 0<=int(label)<16
            assert not p.is_relative_to(Path(r'F:\A2_preprocessed'));assert expected_entries[str(p)]==(split,int(label));target.append((str(p),int(label)))
    ready=json.loads((ROOT/'READY.json').read_text());assert len(train)==ready['train_csv_entries'] and len(val)==ready['val_csv_entries']
    assert not set(p for p,l in train)&set(p for p,l in val)
    assert {label for p,label in train}==set(range(16))
    a2=json.loads(Path(r'F:\A2_preprocessed\READY.json').read_text());assert a2['status']=='inference_ready_subset'
    result=dict(status='passed',source_videos_checked=330,source_original_size_and_sampled_fingerprints_unchanged=True,
        new_source_mtimes_unchanged=True,ready_package_hashes_match=True,manifest_media_and_validation_stats_match=True,
        no_partial_media=True,train_csv_rows=len(train),val_csv_rows=len(val),train_classes=sorted({x[1] for x in train}),
        val_classes=sorted({x[1] for x in val}),A2_in_training=False,packages=reports)
    result.update(retained_view_clip_labels_and_timing_checked_against_original_CSV_rows=provenance_checks)
    (ROOT/'final_verification.json').write_text(json.dumps(result,indent=2))
    metadata=ROOT/'metadata';metadata.mkdir(exist_ok=True)
    (metadata/'preprocessing_script_sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},indent=2))
    (metadata/'extension_test_results.json').write_text((HERE/'extension_test_results.json').read_text())
    (metadata/'source_class_coverage.json').write_text(json.dumps(source_class_coverage,indent=2))
    ready.update(status='training_ready_subset',final_verification_passed=True)
    (ROOT/'READY.json').write_text(json.dumps(ready,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':run()
