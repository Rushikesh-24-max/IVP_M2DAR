(function(root){
 'use strict';
 const views=['dashboard','rearview','right_side_window'];
 function validate(r){
  if(!r||r.schema_version!==1||!r.views||!Number.isFinite(r.duration_s)||r.duration_s<=0||r.duration_s>86400)throw Error('Unsupported result. Import a Driver Action Lab result JSON.');
  const keys=Object.keys(r.views);
  if(!keys.length||keys.some(k=>!views.includes(k)))throw Error('Unsupported camera views.');
  for(const k of keys){const v=r.views[k];if(!Number.isFinite(v.duration_s)||v.duration_s<=0||!Number.isFinite(v.offset_s)||!Array.isArray(v.windows)||!v.windows.length||v.windows.length>10000)throw Error('Invalid video timing or windows.');
   for(const w of v.windows){if(!Number.isFinite(w.start_s)||!Number.isFinite(w.end_s)||w.start_s<0||w.end_s<=w.start_s||w.end_s>v.duration_s+.01||!Array.isArray(w.probabilities)||w.probabilities.length!==16||w.probabilities.some(x=>!Number.isFinite(x)||x<0||x>1.001)||Math.abs(w.probabilities.reduce((a,b)=>a+b,0)-1)>.003)throw Error('Invalid probability vector or window boundary.');}
  }return r;
 }
 function atView(v,t){const time=t+v.offset_s,windows=v.windows.filter(w=>w.start_s<=time&&time<w.end_s);if(!windows.length)return null;return Array.from({length:16},(_,i)=>windows.reduce((s,w)=>s+w.probabilities[i],0)/windows.length);}
 function at(r,t,view='fused'){
  if(view!=='fused')return atView(r.views[view],t);
  if(t<(r.reference_start_s||0)||t>=r.duration_s)return null;
  const ps=Object.values(r.views).map(v=>atView(v,t));if(ps.some(p=>!p))return null;
  return Array.from({length:16},(_,i)=>ps.reduce((s,p)=>s+p[i],0)/ps.length);
 }
 function rank(p){return p?p.map((score,id)=>({id,score})).sort((a,b)=>b.score-a.score||a.id-b.id):[];}
 function intervals(r,view='fused',threshold=0,hideNormal=false){
  const vs=view==='fused'?Object.values(r.views):[r.views[view]],edges=new Set([0,r.duration_s,r.reference_start_s||0]);
  for(const v of vs)for(const w of v.windows){for(const x of [w.start_s-v.offset_s,w.end_s-v.offset_s])if(x>0&&x<r.duration_s)edges.add(x);}
  const points=[...edges].sort((a,b)=>a-b),out=[];
  for(let i=0;i<points.length-1;i++){const start=points[i],end=points[i+1];if(end-start<1e-8)continue;const top=rank(at(r,(start+end)/2,view))[0];if(!top||top.score<threshold||hideNormal&&top.id===0)continue;
   const last=out.at(-1),weight=(end-start)*top.score;if(last&&last.id===top.id&&Math.abs(last.end-start)<1e-7){last.end=end;last.weight+=weight;last.score=last.weight/(end-last.start);}else out.push({id:top.id,start,end,score:top.score,weight});
  }return out;
 }
 const engine={validate,at,atView,rank,intervals};if(typeof module!=='undefined'&&module.exports)module.exports=engine;else root.M2DAR=engine;
})(typeof window!=='undefined'?window:globalThis);
