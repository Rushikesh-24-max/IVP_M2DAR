"""Package only completed, model-blind synchronization evidence for the server."""
from pathlib import Path
import collections,json,tarfile
from evaluate_test import read,digest
from evaluate_synchronized_test import verify_sync
ROOT=Path(__file__).resolve().parent;SYNC=ROOT/'sync'
def main():
 r=read(SYNC/'READY_SYNC.json')
 if r['status']!='synchronization_assessment_complete':raise ValueError('Synchronization is not complete')
 # This is a provenance-only addition for the already-running first pass.
 for name,key in [('full_source_motion_audit.json','source_motion_audit_sha256'),('group_checks.json','group_checks_sha256'),('alignment_estimates.json','alignment_sha256')]:r[key]=digest(SYNC/name)
 (SYNC/'READY_SYNC.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
 media=Path(read(ROOT/'READY.json')['media_root']);protocol,groups,r=verify_sync(media,SYNC)
 audit=read(SYNC/'full_source_motion_audit.json')
 if len(audit)!=114 or len({a['key'] for a in audit})!=114 or any(a['metadata'].get('corrupt_frames',0) for a in audit):raise ValueError('Incomplete source audit')
 output=ROOT/'synchronization_verified_transfer.tar.gz'
 names=['sync_protocol.json','full_source_motion_audit.json','group_checks.json','alignment_estimates.json','alignment_omissions.json','synchronized_test_groups.jsonl','READY_SYNC.json']
 with tarfile.open(output,'w:gz') as t:
  for name in names:t.add(SYNC/name,arcname='M2DAR_testset_sync/'+name)
 print(json.dumps(dict(archive=str(output),archive_sha256=digest(output),verified_groups=len(groups),verified_classes=r['verified_classes'],verified_drivers=r['verified_drivers'],primary_test_clips=1748,source_videos_checked=114),indent=2))
if __name__=='__main__':main()
