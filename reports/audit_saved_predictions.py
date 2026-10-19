from pathlib import Path
import json,csv,hashlib,re,collections
import numpy as np
import pyarrow.parquet as pq
B=Path('/root/autodl-tmp/openpi_hrc/outputs/dissertation_final_audit')
O=Path('/root/dissertation_release_staging/reports')
M=json.loads((B/'anchor_manifest.json').read_text()); anchors=M['anchors']
R={'scope':'Independent archive consistency and saved-prediction rescoring only; training, policy inference and simulation: Not independently verified','manifest_sha256':hashlib.sha256((B/'anchor_manifest.json').read_bytes()).hexdigest(),'models':{},'conditioning':{},'logs':{},'checks':{}}
data={}
for key,root in M['dataset_roots'].items():
 cols=['index','observation.state','action','hrc.valid_anchor','hrc.context_id','hrc.source_episode_id','hrc.raw_global_frame_index','episode_index']
 files=sorted(Path(root).glob('data/**/*.parquet')); rows=[]
 for f in files:rows.extend(pq.read_table(f,columns=cols).to_pylist())
 rows.sort(key=lambda x:x['index']);data[key]=rows
R['checks']['matched_physical_rows']=all(np.array_equal(data['A'][a['row_a']]['action'],data['B_C'][a['row_bc']]['action']) and np.array_equal(data['A'][a['row_a']]['observation.state'],data['B_C'][a['row_bc']]['observation.state']) for a in anchors)
R['checks']['context_counts']=dict(collections.Counter(a['context'] for a in anchors))
arms=[i for i in range(23) if i not in (0,1,2,14,22)]; groups={'arms':arms,'base':[0,1,2],'grippers':[14,22]}
def events(x,prev):
 s=np.sign(x[:,[14,22]]);p=np.vstack([np.sign(prev[[14,22]])[None,:],s[:-1]])
 return [(int(t),int(g),s[t,g]) for t,g in np.argwhere(s!=p)]
for model in ('A20','B20','C20'):
 rows=data['A' if model=='A20' else 'B_C'];rf='row_a' if model=='A20' else 'row_bc';preds=[];allmetrics=[];md={'seeds':{},'max_absolute_csv_error':0.,'max_absolute_seed_summary_error':0.}
 for seed in range(5):
  root=B/model/'per_seed'/f'seed_{seed}'; z=np.load(root/'predictions.npz',allow_pickle=False);pred=z['predictions'];preds.append(pred)
  assert list(z['anchor_ids'])==[a['anchor_id'] for a in anchors];assert pred.shape==(652,32,23);assert np.isfinite(pred).all()
  saved=list(csv.DictReader((root/'per_anchor.csv').open()));ss=json.loads((root/'summary.json').read_text());acc=collections.defaultdict(list)
  for i,a in enumerate(anchors):
   t=a[rf];row=rows[t];state=np.asarray(row['observation.state'],np.float32);truth=np.asarray([r['action'] for r in rows[t:t+32]],np.float32);prev=np.asarray(rows[t-1]['action'] if t and rows[t-1]['hrc.source_episode_id']==a['source_episode_id'] else row['action'],np.float32)
   assert row['hrc.valid_anchor'];assert row['hrc.source_episode_id']==a['source_episode_id'];assert row['hrc.raw_global_frame_index']==a['raw_global_frame_index']
   assert all(r['hrc.context_id']==a['context_id'] and r['hrc.source_episode_id']==a['source_episode_id'] and r['episode_index']==row['episode_index'] for r in rows[t:t+33])
   null=np.zeros((32,23),np.float32);null[:,arms]=state[arms];null[:,[14,22]]=prev[[14,22]]
   met={}
   for g,ix in groups.items():
    met[g+'_raw_mae']=float(np.abs(pred[i][:,ix]-truth[:,ix]).mean());met[g+'_null_mae']=float(np.abs(null[:,ix]-truth[:,ix]).mean())
   pe=events(pred[i],prev);ee=events(truth,prev);unmatched=ee.copy();tp=0
   for t1,g,s in pe:
    matches=[j for j,(t2,g2,s2) in enumerate(unmatched) if g==g2 and s==s2 and abs(t1-t2)<=8]
    if matches:unmatched.pop(matches[0]);tp+=1
   met.update(gripper_expert_events=len(ee),gripper_predicted_events=len(pe),gripper_true_positives=tp)
   for k,v in met.items():md['max_absolute_csv_error']=max(md['max_absolute_csv_error'],abs(v-float(saved[i][k])))
   acc[a['context']].append(met)
  ctx={}
  for c,rs in acc.items():
   vals={k:float(np.mean([r[k] for r in rs])) for k in rs[0] if not k.startswith('gripper_')};e=sum(r['gripper_expert_events'] for r in rs);p=sum(r['gripper_predicted_events'] for r in rs);tp=sum(r['gripper_true_positives'] for r in rs)
   vals.update(gripper_expert_events=e,gripper_predicted_events=p,gripper_true_positives=tp,gripper_f1=2*tp/(e+p) if e else None,gripper_precision=tp/p if p else None,gripper_recall=tp/e if e else None)
   for g in groups:
    n=vals[g+'_null_mae'];vals[g+'_skill']=1-vals[g+'_raw_mae']/n if n else None
    for k in ('raw_mae','null_mae','skill'):
     v=vals[g+'_'+k];sv=ss['by_context'][c]['groups'][g][k]
     if v is not None:md['max_absolute_seed_summary_error']=max(md['max_absolute_seed_summary_error'],abs(v-sv))
   for k in ('expert_events','predicted_events','true_positives','f1','precision','recall'):
    assert vals['gripper_'+k]==ss['by_context'][c]['gripper'][k]
   ctx[c]=vals
  md['seeds'][str(seed)]={'num_predictions':652,'finite':True,'anchor_order_exact':True,'by_context':ctx};allmetrics.append(ctx)
 summary=json.loads((B/model/'summary.json').read_text());md['aggregate_pickup']={}
 for k in ('arms_raw_mae','arms_skill','gripper_f1','gripper_precision','gripper_recall'):
  vals=np.array([s['pickup'][k] for s in allmetrics]);md['aggregate_pickup'][k]={'mean':float(vals.mean()),'sample_std':float(vals.std(ddof=1))}
 R['models'][model]=md
for model in ('B20','C20'):
 p=B.parent/'dissertation_conditioning_probe'/model/'per_anchor.csv';rs=list(csv.DictReader(p.open()));out={}
 for sp in ('val','train_coverage'):
  sub=[r for r in rs if r['split']==sp];out[sp]={'num_anchors':len(sub),'sources':sorted({int(r['source_episode_id']) for r in sub})}
  for c in ('pickup','radio'):
   cs=[r for r in sub if r['context']==c];out[sp][c]={'n':len(cs),'arms_image_over_noise_mean':float(np.mean([float(r['arms_D_image'])/float(r['arms_D_noise']) for r in cs]))}
 R['conditioning'][model]=out
for model,name in [('A20','train_e6_rerun_20k.log'),('B20','train_b_subinstr_20k.log'),('C20','train_c_qnorm_20k.log')]:
 p=B.parents[1]/'logs/hrc'/name;s=p.read_text();tr=re.findall(r'Step (\d+):[^\n]*?loss=([\d.eE+-]+)',s);va=re.findall(r'Validation loss at step (\d+): ([\d.eE+-]+)',s)
 R['logs'][model]={'path':str(p),'training_points':len(tr),'validation_points':len(va),'last_training':tr[-1],'last_validation':va[-1]}
 with (O/(model+'_training_curve.csv')).open('w') as f:
  w=csv.writer(f);w.writerow(['model','phase','optimizer_step','normalized_flow_loss','source_log']);w.writerows((model,phase,step,loss,str(p)) for phase,vals in [('train',tr),('val',va)] for step,loss in vals)
(O/'independent_archive_checks.json').write_text(json.dumps(R,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'models':{k:{kk:vv for kk,vv in v.items() if kk!='seeds'} for k,v in R['models'].items()},'conditioning':R['conditioning'],'checks':R['checks'],'output':str(O/'independent_archive_checks.json')},indent=2))
