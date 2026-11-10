// Exercise actual player handlers and score-panel updates without claiming browser QA.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),P=require('../dist/prediction.js');
const elements=new Map();
function element(id){if(!elements.has(id)){const attrs={},classes=new Set();elements.set(id,{id,value:'0',textContent:'',innerHTML:'',style:{},dataset:{},hidden:false,paused:true,currentTime:0,duration:6,readyState:1,events:{},setAttribute(k,v){attrs[k]=v;},getAttribute(k){return attrs[k];},addEventListener(k,f){this.events[k]=f;},pause(){this.paused=true;},async play(){this.paused=false;},classList:{toggle(k,v){if(v)classes.add(k);else classes.delete(k);},remove(k){classes.delete(k);}},removeAttribute(){},load(){}});}return elements.get(id);}
const sandbox={window:{M2DARPrediction:P,addEventListener(){}},document:{getElementById:element,querySelectorAll:()=>[]},history:{replaceState(){}},location:{hash:'',hostname:'localhost',protocol:'http:',origin:'http://localhost:8765'},console,setTimeout:()=>0,clearTimeout(){},URL,Blob,AbortSignal,crypto:require('node:crypto').webcrypto};
vm.createContext(sandbox);vm.runInContext(fs.readFileSync('dist/app.js','utf8').replace(/\binit\(\);\s*$/,''),sandbox);
const one=c=>Array.from({length:16},(_,i)=>Number(i===c));
const r={schema_version:1,checkpoint_sha256:P.CHECKPOINT,inference_mode:'sliding_window',duration_s:6,views:{dashboard:{offset_s:0,duration_s:6,windows:[{start_s:0,end_s:2,probabilities:one(1)},{start_s:1,end_s:3,probabilities:one(2)},{start_s:4,end_s:6,probabilities:one(3)}]}}};
sandbox.fixture={result:r,file:{},exportData:r,classes:Object.fromEntries(Array.from({length:16},(_,i)=>[i,'Action '+i]))};
vm.runInContext('Object.assign(state,fixture);setPlayerDuration(6);',sandbox);
const slider=element('predictionSlider'),video=element('videoPlayer');
slider.value='1.5';slider.oninput({target:slider});assert.equal(video.currentTime,1.5);assert.equal(element('timelineCurrent').textContent,'00:01.5');assert.equal(element('probabilityTime').textContent,'00:01.5');assert.equal(element('currentAction').textContent,'Action 1');assert.equal((element('topPredictions').innerHTML.match(/class="rank-row"/g)||[]).length,5);assert.ok(element('topPredictions').innerHTML.includes('50.0%'));
slider.value='3.5';slider.oninput({target:slider});assert.equal(element('currentAction').textContent,'Uncovered interval');assert.equal(element('topPredictions').innerHTML,'');assert.ok(element('timeSlotInfo').textContent.includes('unknown'));
element('nextSlot').onclick();assert.equal(video.currentTime,4);assert.equal(element('currentAction').textContent,'Action 3');
video.currentTime=4.75;video.events.timeupdate();assert.equal(Number(slider.value),4.75);assert.equal(element('timelineCurrent').textContent,'00:04.7');
vm.runInContext('seekToTime(100);',sandbox);assert.ok(video.currentTime<6);assert.equal(element('currentAction').textContent,'Action 3');
sandbox.saved={probabilities:one(0),ground_truth:0};vm.runInContext('state.result=null;state.selected=saved;setPlayerDuration(6);seekToTime(5);',sandbox);assert.equal(element('currentAction').textContent,'Action 0');assert.ok(element('timeSlotInfo').textContent.includes('stay constant'));
vm.runInContext('state.selected=null;state.result=null;seekToTime(3);',sandbox);assert.equal(element('currentAction').textContent,'Awaiting prediction');assert.equal(element('topPredictions').innerHTML,'');
assert.equal(P.slotTime(r,1.5,-1,6),1);assert.equal(P.slotTime(r,1.5,1,6),4);assert.equal(P.clampTime(-1,6),0);
console.log('Actual slider handlers: video seeking, top-five window averages, gaps, playback sync, endpoints, slot navigation and unchanged clip scores passed.');
const requested=[];sandbox.fetch=async url=>{requested.push(url);return {ok:true,json:async()=>r};};
vm.runInContext("state.api='http://localhost:8765';state.busy=false;",sandbox);
vm.runInContext("inspectStoredJob('existing-job')",sandbox).then(()=>{assert.equal(requested.length,1);assert.ok(requested[0].endsWith('/api/jobs/existing-job/result'));assert.ok(element('videoMeta').textContent.includes('not loaded'));assert.equal(element('timelinePlay').disabled,true);slider.value='4.5';slider.oninput({target:slider});assert.equal(element('currentAction').textContent,'Action 3');console.log('Stored-job timeline reopens and scrubs without a new upload or GPU run.');}).catch(e=>{console.error(e);process.exitCode=1;});
