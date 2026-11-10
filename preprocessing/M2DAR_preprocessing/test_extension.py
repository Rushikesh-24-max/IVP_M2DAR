"""Boundary checks for annotation quarantine, ID parsing and clock restoration."""
from pathlib import Path
import sys,json,tempfile,csv,collections
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import audit_labels_generic as audit
from restore_a2_times import restore
from prepare_a2 import load_ids

def run():
    tests=[]
    assert audit.identify('Rear_view_User_id_57207_NoAudio_1.MP4')==('57207','1','rearview')
    assert audit.identify('Right_side_window_user_id_59359_7')==('59359','7','right_side_window')
    tests.append(dict(check='filename_case_camera_aliases_and_nonstandard_session',passed=True))
    with tempfile.TemporaryDirectory(dir=HERE,prefix='label_test_') as td:
        root=Path(td);labels=root/'labels'/'A1';labels.mkdir(parents=True)
        sources=[dict(driver='12345',session='5',view=v,path=str(root/(v+'.MP4')),duration_s=120) for v in audit.VIEWS]
        (root/'source_videos.json').write_text(json.dumps(sources))
        (root/'package_config.json').write_text(json.dumps(dict(raw_root='synthetic',labels_root=str(root/'labels'))))
        entries=[(0,10,0),(0,10,0),(10,100,1),(20,30,2),(40,50,3),(60,70,4),(65,75,5),(80,79,6),(115,125,7)]
        def clock(s):return f'00:{s//60:02d}:{s%60:02d}'
        with (labels/'user_id_12345.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['Filename','Start Time','End Time','Label (Primary)'])
            for a,b,label in entries:w.writerow(['Dashboard_user_id_12345_5',clock(a),clock(b),'Class '+str(label)])
            w.writerow(['Dashboard_user_id_12345_5','00:00:00','00:00:01','unknown'])
        original=audit.OUT
        try:audit.OUT=root;audit.run()
        finally:audit.OUT=original
        summary=json.loads((root/'annotation_audit.json').read_text());assert summary['valid_rows']==3 and summary['excluded_rows']==6,summary
        assert {'duplicate_annotation','spans_multiple_labeled_actions','conflicting_annotation_overlap','nonpositive_interval','outside_video_duration','unparseable_or_unmatched_annotation'}<=set(summary['issues'])
        tests.append(dict(check='invalid_duplicate_long_conflicting_and_overlapping_labels_quarantined',passed=True))
    gg=[dict(group_id='a',video_id=1,frame_count=300,reference_start_s=100),dict(group_id='b',video_id=1,frame_count=300,reference_start_s=108),dict(group_id='c',video_id=1,frame_count=300,reference_start_s=130)]
    result=restore(['a 3 7 12','b 3 0 3','c 3 0 2','a 0 0 0'],gg)
    assert len(result)==2 and result[0]['start_s']==107 and result[0]['end_s']==111 and result[1]['start_s']==130,result
    try:restore(['a 16 0 1'],gg);raise AssertionError('Invalid class accepted')
    except ValueError:pass
    tests.append(dict(check='local_to_original_times_bounds_no_candidate_and_unknown_gap',passed=True))
    sources=json.loads(Path(r'F:\A2_preprocessed\source_videos.json').read_text());ids,records=load_ids(sources)
    assert len(ids)==30 and len(records)==30
    tests.append(dict(check='official_2024_list_serialization_parsed_and_all_90_videos_joined',passed=True))
    normalized=json.loads(Path(r'F:\A1_3_preprocessed\annotations_normalized.json').read_text())
    corrected=[r for r in normalized if r.get('camera_mapping_corrected')]
    assert len(corrected)==64 and {r['view'] for r in corrected}=={'rearview','right_side_window'}
    assert all(r['driver']=='47158' and r['filename_camera_view']=='dashboard' and r['status']=='valid' for r in corrected)
    tests.append(dict(check='inspected_47158_camera_mapping_restores_64_rows_without_changing_action_times_or_labels',passed=True))
    a1parts=[json.loads(Path(r'F:\A1_analysis\mp4_inventory.json').read_text())]
    for part in ['A1_2','A1_3','A1_4']:a1parts.append(json.loads((Path('F:/')/(part+'_preprocessed')/'source_inventory_original.json').read_text()))
    drivers=[{r['driver'] for r in rr} for rr in a1parts];a2={r['driver'] for r in sources}
    assert sum(map(len,drivers))==len(set().union(*drivers))==40 and not set().union(*drivers)&a2
    candidates=collections.defaultdict(list)
    for rr in a1parts+[sources]:
        for r in rr:candidates[(r.get('size_bytes',r.get('source_size_bytes')),r.get('sampled_fingerprint_sha256',r.get('sampled_source_fingerprint_sha256')))].append(r['path'])
    duplicates=[p for p in candidates.values() if len(p)>1];assert not duplicates,duplicates
    tests.append(dict(check='40_A1_and_15_A2_drivers_disjoint_and_no_size_plus_sampled_fingerprint_duplicates',passed=True))
    (HERE/'extension_test_results.json').write_text(json.dumps(tests,indent=2));print(json.dumps(tests,indent=2))

if __name__=='__main__':run()
