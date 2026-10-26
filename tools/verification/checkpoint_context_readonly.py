#!/usr/bin/env python3
"""Collect focused launch/config/frozen-archive context; writes only delivery JSON."""
import datetime,hashlib,json,os,pathlib,re,subprocess,sys
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import numpy as np
import yaml
B=pathlib.Path('/root/autodl-tmp/openpi_hrc');O=pathlib.Path('/root/dissertation_evidence_verification')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 return h.hexdigest()
def file(p):
 p=pathlib.Path(p);return {'path':str(p),'exists':p.exists(),**({'size_bytes':p.stat().st_size,'sha256':sha(p)} if p.is_file() else {})}
d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':sys.argv,'launch_records':{},'norms':{},'frozen_archives':{},'symlink_aliases':[],'resource_checks':{}}
for lab,pp in [('A20','/root/autodl-tmp/wandb/wandb/offline-run-20260725_230155-jzt3tr1g/files'),('C20',str(B/'wandb/run-20260810_010736-rk0qfy4t/files'))]:
 p=pathlib.Path(pp);cfg=yaml.safe_load((p/'config.yaml').read_text());meta=json.loads((p/'wandb-metadata.json').read_text())
 # Select research-relevant fields; do not copy email/host/GPU UUID into delivery.
 d['launch_records'][lab]={'config_file':file(p/'config.yaml'),'metadata_file':file(p/'wandb-metadata.json'),'seed':cfg.get('seed'),'model':cfg.get('model'),'data':cfg.get('data'),'num_train_steps':cfg.get('num_train_steps'),'exp_name':cfg.get('exp_name'),'metadata_selected':{k:meta.get(k) for k in ['args','program','codePath','code_path','git','root','executable','startedAt','started_at']}}
 req=p/'requirements.txt'
 if req.exists():d['launch_records'][lab]['requirements']=file(req);d['launch_records'][lab]['selected_requirements']=[s for s in req.read_text().splitlines() if s.split('==')[0] in ['jax','jaxlib','flax','orbax-checkpoint','numpy','torch','transformers']]
# B W&B files/config.yaml is absent, but debug.log preserves the fully expanded config.
bpath=pathlib.Path('/root/autodl-tmp/wandb/wandb/offline-run-20260729_204114-f5eqx95o/logs/debug.log')
bline=next((n,line) for n,line in enumerate(bpath.read_text().splitlines(),1) if line.startswith('config: ') and "'name': 'pi05_hrc_b_subinstr_20k'" in line)
breq=bpath.parent.parent/'files/requirements.txt'
d['launch_records']['B20']={'config_log':file(bpath),'line':bline[0],'recorded_config_line':bline[1],'seed':{'value':int(re.search(r"'seed': (\d+)",bline[1]).group(1))},'num_train_steps':{'value':int(re.search(r"'num_train_steps': (\d+)",bline[1]).group(1))},'exp_name':{'value':re.search(r"'exp_name': '([^']+)'",bline[1]).group(1)},'requirements':file(breq),'selected_requirements':[z for z in breq.read_text().splitlines() if z.split('==')[0] in ['jax','jaxlib','flax','orbax-checkpoint','numpy','torch','transformers']],'exact_training_argv':'Not Verified: full expanded run config is retained in debug.log; raw B training argv is not independently located.'}
for lab,asset in [('A20','main_continuous_rerun_v1_natural'),('B20','subinstr_rerun_v1'),('C20','subinstr_rerun_v1_qnorm')]:
 p=B/'assets/hrc'/asset/'norm_stats.json';obj=json.loads(p.read_text())['norm_stats'];d['norms'][lab]={'file':file(p),'fields':{key:{k:(len(v) if isinstance(v,list) else v) for k,v in val.items()} for key,val in obj.items()}}
r=B/'outputs/dissertation_final_audit';manifest=json.loads((r/'anchor_manifest.json').read_text());manifestsha=sha(r/'anchor_manifest.json');d['frozen_manifest']={'file':file(r/'anchor_manifest.json'),'num_anchors':manifest['num_anchors'],'flow_seeds':manifest['flow_seeds'],'protocol_seed':manifest['protocol_seed'],'source_ids':manifest['source_ids'],'noise_scheme':manifest['noise_scheme']}
for lab in ['A20','B20','C20']:
 d['frozen_archives'][lab]=[]
 for seed in range(5):
  p=r/lab/'per_seed'/f'seed_{seed}'/'predictions.npz';sm=json.loads((p.parent/'summary.json').read_text());rec={'file':file(p),'summary_file':file(p.parent/'summary.json'),'manifest_sha_matches':sm['manifest_sha256']==manifestsha,'summary_flow_seed':sm['flow_seed'],'summary_checkpoint':sm['checkpoint']}
  with np.load(p,allow_pickle=False) as z:
   rec['array_keys']=list(z.keys());rec['shapes']={k:list(z[k].shape) for k in z};rec['dtypes']={k:str(z[k].dtype) for k in z};rec['predictions_finite']=bool(np.isfinite(z['predictions']).all());rec['anchor_ids_match']=bool(np.array_equal(z['anchor_ids'],np.array([a['anchor_id'] for a in manifest['anchors']])))
  d['frozen_archives'][lab].append(rec)
for p in ['/root/.cache','/root/.cache/openpi','/root/autodl-tmp/root_cache/openpi','/root/autodl-tmp/.cache/openpi']:
 pp=pathlib.Path(p);d['symlink_aliases'].append({'path':p,'lexists':os.path.lexists(pp),'is_symlink':pp.is_symlink(),'link_target':os.readlink(pp) if pp.is_symlink() else None,'realpath':str(pp.resolve()),'exists':pp.exists()})
genv=dict(os.environ,GIT_OPTIONAL_LOCKS='0');
def git(args):return subprocess.run(['git','-C',str(B),*args],env=genv,capture_output=True,check=False)
d['git_current_head']=git(['rev-parse','HEAD']).stdout.decode().strip();d['git_status_porcelain']=git(['status','--porcelain']).stdout.decode().splitlines()
d['archived_commit_code_comparison']=[]
for rel in ['src/openpi/training/config.py','src/openpi/models/model.py','src/openpi/models/pi0.py','src/openpi/models/pi0_config.py','src/openpi/models/gemma.py','src/openpi/models/tokenizer.py','src/openpi/transforms.py','src/openpi/policies/policy.py','src/openpi/policies/policy_config.py','scripts/hrc/run_dissertation_final_audit.py','scripts/hrc/dissertation_protocol.py','scripts/hrc/evaluate_hrc_pi05_continuous.py']:
 rr=git(['show','cfd097d6ca0f957b68d05616a3b71b86dabdf2cf:'+rel]);rec={'path':str(B/rel),'current_sha256':sha(B/rel),'archive_commit':'cfd097d6ca0f957b68d05616a3b71b86dabdf2cf','git_show_returncode':rr.returncode}
 if rr.returncode==0:rec['archived_blob_sha256']=hashlib.sha256(rr.stdout).hexdigest();rec['current_matches_archived_commit']=rec['current_sha256']==rec['archived_blob_sha256']
 d['archived_commit_code_comparison'].append(rec)
for p in ['/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current']:d['resource_checks'][p]=pathlib.Path(p).read_text().strip()
d['resource_checks']['gpu']=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.free','--format=csv,noheader'],capture_output=True,check=False,text=True).stdout.strip()
out=O/'evidence/checkpoint_context.json';out.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'output':str(out),'archive_checks':{k:len(v) for k,v in d['frozen_archives'].items()},'git_head':d['git_current_head'],'current_code_matches_archive':all(r.get('current_matches_archived_commit',False) for r in d['archived_commit_code_comparison'])}))
