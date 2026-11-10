"""Create labeled three-view action clips only inside supported common timelines."""
from pathlib import Path
import json,math,hashlib,random,collections
OUT=Path(__file__).resolve().parent
VIEWS=['dashboard','rearview','right_side_window'];FPS=30;EDGE_MARGIN=0.25;MIN_FRAMES=16
CLASSES=['Normal Forward Driving','Drinking','Phone Call(right)','Phone Call(left)','Eating','Text (Right)','Text (Left)','Reaching behind','Adjust control panel','Pick up from floor (Driver)','Pick up from floor (Passenger)','Talk to passenger at the right','Talk to passenger at backseat','yawning','Hand on head','Singing or dancing with music']

def run():
    rows=json.loads((OUT/'annotations_normalized.json').read_text());good=[r for r in rows if r['status']=='valid']
    groups=json.loads((OUT/'alignment_estimates.json').read_text())['groups']
    drivers=sorted({r['driver'] for r in rows});shuffled=drivers.copy();random.Random(42).shuffle(shuffled)
    val=set(shuffled[:2]);split={d:'val' if d in val else 'train' for d in drivers}
    (OUT/'splits.json').write_text(json.dumps(dict(seed=42,unit='driver',train=sorted(set(drivers)-val),val=sorted(val),test=[],note='A2 and B not used; both recordings and all views of a driver remain together.'),indent=2))
    (OUT/'classes.json').write_text(json.dumps({i:c for i,c in enumerate(CLASSES)},indent=2))
    plan=[];omitted=[]
    for group in groups:
        driver,session=group['driver'],group['session'];vv=group['views']
        ann={v:[r for r in good if (r['driver'],r['session'],r['view'])==(driver,session,v)] for v in VIEWS}
        for ref in ann['dashboard']:
            found=[]
            for rear in vv['rearview']['segments']:
                for right in vv['right_side_window']['segments']:
                    lo=max(ref['start_s'],rear['reference_start_s'],right['reference_start_s'])
                    hi=min(ref['end_s'],rear['reference_end_s'],right['reference_end_s'])
                    if hi-lo<(MIN_FRAMES/FPS)+2*EDGE_MARGIN:continue
                    for ra in ann['rearview']:
                        if ra['label']!=ref['label']:continue
                        for ri in ann['right_side_window']:
                            if ri['label']!=ref['label']:continue
                            offsets={'dashboard':0,'rearview':rear['offset_s'],'right_side_window':right['offset_s']}
                            originals={'dashboard':ref,'rearview':ra,'right_side_window':ri}
                            a=max(lo,ra['start_s']-offsets['rearview'],ri['start_s']-offsets['right_side_window'])+EDGE_MARGIN
                            b=min(hi,ra['end_s']-offsets['rearview'],ri['end_s']-offsets['right_side_window'])-EDGE_MARGIN
                            # Source duration and PTS boundaries must be respected independently.
                            a=max(a,*[-offsets[v] for v in VIEWS]);b=min(b,*[originals[v]['playback_duration_s']-offsets[v] for v in VIEWS])
                            begin_frame=math.ceil(a*FPS-1e-7);end_frame=math.floor(b*FPS+1e-7)
                            count=end_frame-begin_frame
                            if count<MIN_FRAMES:continue
                            identity=dict(driver=driver,session=session,label=ref['label'],reference_start_frame=begin_frame,reference_end_frame=end_frame,offset_frames={v:int(round(offsets[v]*FPS)) for v in VIEWS})
                            digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
                            gid=f'd{driver}_r{session}_c{ref["label"]:02d}_{digest}'
                            item=dict(group_id=gid,**identity,split=split[driver],fps=FPS,frame_count=count,
                                reference_start_s=begin_frame/FPS,reference_end_s=end_frame/FPS,
                                edge_margin_s=EDGE_MARGIN,temporal_padding_required=count<64,
                                short_clip_policy='retain observed frames; model sampler may repeat the valid boundary frame',views={})
                            for v in VIEWS:
                                source=originals[v]
                                evidence={'dashboard':None,'rearview':rear,'right_side_window':right}[v]
                                item['views'][v]=dict(source_path=source['source_path'],source_start_s=begin_frame/FPS+offsets[v],source_end_s=end_frame/FPS+offsets[v],offset_s=offsets[v],source_annotation_file=source['source_annotation_file'],source_annotation_row=source['source_row'],original_label_start_s=source['start_s'],original_label_end_s=source['end_s'],relative_output_path=f'clips/{driver}/{session}/{gid}_{v}.mp4',
                                    alignment_anchor_count=evidence.get('accepted_anchor_count') if evidence else None,
                                    alignment_interval_reference_s=[evidence['reference_start_s'],evidence['reference_end_s']] if evidence else [0,ref['playback_duration_s']],
                                    maximum_supported_anchor_residual_frames=evidence.get('max_anchor_residual_frames') if evidence else 0)
                            found.append(item)
            unique={x['group_id']:x for x in found}
            if not unique:omitted.append(dict(driver=driver,session=session,label=ref['label'],reference_annotation_row=ref['source_row'],reason='no_three_view_label_overlap_with_supported_alignment_and_minimum_duration'))
            plan.extend(unique.values())
    unique={p['group_id']:p for p in plan};plan=sorted(unique.values(),key=lambda p:(p['driver'],p['session'],p['reference_start_frame'],p['label']))
    assert all(p['frame_count']>=MIN_FRAMES for p in plan)
    assert not ({p['driver'] for p in plan if p['split']=='train'} & {p['driver'] for p in plan if p['split']=='val'})
    (OUT/'clip_plan.json').write_text(json.dumps(dict(schema_version=1,status='planned_not_yet_exported',reference_view='dashboard',fps=FPS,width=960,height=540,groups=plan),indent=2))
    (OUT/'clip_plan_omissions.json').write_text(json.dumps(omitted,indent=2))
    stats=dict(groups=len(plan),view_clips=len(plan)*3,recordings=len(groups),action_reference_hours=sum(x['frame_count']/FPS for x in plan)/3600,
        by_split=dict(collections.Counter(p['split'] for p in plan)),class_groups=dict(collections.Counter(p['label'] for p in plan)),omitted_reference_annotations=len(omitted))
    (OUT/'clip_plan_summary.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2))

if __name__=='__main__':run()
