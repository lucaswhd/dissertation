#!/usr/bin/env python3
"""Read-only bounded HRC checkpoint locator / metadata and hash audit. Stdlib only.
Writes only --output under the verification delivery directory. No checkpoint restore.
"""
import argparse,datetime,hashlib,json,os,pathlib,re,sys,tarfile,zipfile
BASE=pathlib.Path('/root/autodl-tmp/openpi_hrc')
OUT=pathlib.Path('/root/dissertation_evidence_verification')
SPECS={'A20':('pi05_hrc_e6_rerun_20k','hrc_e6_rerun_seed42','main_continuous_rerun_v1_natural','train_e6_rerun_20k.log'),'B20':('pi05_hrc_b_subinstr_20k','b_subinstr_seed42','subinstr_rerun_v1','train_b_subinstr_20k.log'),'C20':('pi05_hrc_c_qnorm_20k','c_qnorm_seed42','subinstr_rerun_v1_qnorm','train_c_qnorm_20k.log')}
ROOTS=[(BASE,9),('/root/autodl-tmp/openpi',9),('/root/autodl-tmp/outputs',9),('/root/autodl-tmp/hrc_snapshots',9),('/root/autodl-tmp/root_cache/openpi',9),('/root/autodl-tmp/.cache/openpi',9),('/root/.local/share/Trash',9),('/root/.Trash',9),('/root/.Trash-0',9),('/root/autodl-tmp/root_cache/huggingface',7),('/root/autodl-tmp/root_cache/wandb',7),('/root/autodl-tmp/wandb',9),('/root/autodl-tmp/hf_cache',7),('/root/autodl-tmp/tmp',5),('/root/autodl-tmp/.Trash-0',9),('/root/autodl-tmp/hrc_pipeline_traintest',9),('/root/hrc_pipeline',9),('/root/dissertation_release_staging',8),('/root/dissertation_local_audit',8),('/root/dissertation_v8_delivery',8)]
SKIP={'.venv','venv','node_modules','.git','__pycache__','.pytest_cache','.ruff_cache','site-packages','datasets','raw','raw_data','videos','media','lib','include','bin','share'}
PAT=re.compile(r'b_subinstr|c_qnorm|e6_rerun|backup|checkpoint|archive|snapshot|20000',re.I)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def stat(p,hash_it=True):
 p=pathlib.Path(p);d={'path':str(p),'lexists':os.path.lexists(p),'exists':p.exists(),'is_symlink':p.is_symlink(),'realpath':str(p.resolve())}
 if p.is_symlink():d['link_target']=os.readlink(p)
 if p.is_file():
  s=p.stat();d.update(size_bytes=s.st_size,device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns)
  if hash_it:d['sha256']=sha(p)
 return d

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',default=str(OUT/'evidence/checkpoint_inventory.json'));ap.add_argument('--hash-payload',action='store_true');a=ap.parse_args()
 op=pathlib.Path(a.output).resolve();assert op.is_relative_to(OUT)
 d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':sys.argv,'scope':{'roots':[(str(r),m) for r,m in ROOTS],'skip_directory_names':sorted(SKIP),'followlinks':False,'archive_limit_bytes':2*1024**3},'search':[],'models':{},'evidence_files':[],'archive_indexes':[]}
 archives=[]
 for rr,maxdepth in ROOTS:
  root=pathlib.Path(rr);rec={'root':str(root),'exists':root.exists(),'max_depth':maxdepth,'errors':[],'directories_visited':0,'files_seen':0,'matching_paths':[],'symlinks':[],'pruned_directory_paths':[],'depth_limited_paths':[]}
  if not root.exists():d['search'].append(rec);continue
  def err(e):rec['errors'].append(str(e))
  for cur,dirs,files in os.walk(root,followlinks=False,onerror=err):
   cp=pathlib.Path(cur);depth=len(cp.relative_to(root).parts);rec['directories_visited']+=1;rec['files_seen']+=len(files)
   for n in dirs+files:
    p=cp/n
    if p.is_symlink():rec['symlinks'].append(stat(p,False))
    if PAT.search(n) or n in {'_METADATA','manifest.ocdbt','model.safetensors','_CHECKPOINT_METADATA'}:rec['matching_paths'].append(stat(p,False))
    if p.is_file() and (n.endswith(('.tar.gz','.tgz','.tar','.tar.xz','.tar.bz2','.zip'))):archives.append(p)
   rec['pruned_directory_paths'].extend(str(cp/x) for x in dirs if x in SKIP)
   if depth>=maxdepth:rec['depth_limited_paths'].extend(str(cp/x) for x in dirs if x not in SKIP)
   dirs[:]=[x for x in dirs if x not in SKIP] if depth<maxdepth else []
  d['search'].append(rec)
 for p in sorted(set(archives)):
  rec=stat(p,True);rec.update(members=0,checkpoint_payload_members=[],relevant_members=[],error=None)
  if p.stat().st_size>2*1024**3:rec['error']='not indexed: over 2GiB limit';d['archive_indexes'].append(rec);continue
  try:
   if p.name.endswith('.zip'):
    with zipfile.ZipFile(p) as z:items=[(i.filename,i.file_size) for i in z.infolist()]
   else:
    with tarfile.open(p,'r:*') as t:items=[(i.name,i.size) for i in t]
   rec['members']=len(items)
   for n,s in items:
    if PAT.search(n):rec['relevant_members'].append({'name':n,'size':s})
    if n.endswith(('_CHECKPOINT_METADATA','manifest.ocdbt','model.safetensors')) or '/params/ocdbt.' in n or '/params/d/' in n:rec['checkpoint_payload_members'].append({'name':n,'size':s})
  except Exception as e:rec['error']=repr(e)
  d['archive_indexes'].append(rec)
 for label,(config,exp,asset,log) in SPECS.items():
  p=BASE/'checkpoints'/config/exp/'20000';rec={'config':config,'experiment':exp,'path_check':stat(p,False),'ancestry':[],'files':[]}
  for parent in [BASE,BASE/'checkpoints',p.parent.parent,p.parent,p]:rec['ancestry'].append(stat(parent,False))
  if p.is_dir():
   rec['top_level_entries']=sorted(x.name for x in p.iterdir())
   for f in sorted(p.rglob('*')):
    if f.is_file():rec['files'].append(stat(f,a.hash_payload or f.stat().st_size<1024*1024))
   rec['total_file_bytes']=sum(f.get('size_bytes',0) for f in rec['files']);rec['file_count']=len(rec['files'])
   rec['parsed_metadata']={}
   for f in [p/'_CHECKPOINT_METADATA',p/'params/_METADATA',p/'params/array_metadatas/process_0',p/'assets'/asset/'norm_stats.json']:
    if f.is_file():
     try:rec['parsed_metadata'][str(f)]=json.loads(f.read_text())
     except Exception as e:rec['parsed_metadata'][str(f)]={'parse_error':repr(e)}
  logpath=BASE/'logs/hrc'/log
  rec['training_log']=stat(logpath);rec['training_log_matches']=[{'line':i,'text':line} for i,line in enumerate(logpath.read_text().splitlines(),1) if '20000' in line and ('checkpoint' in line.lower() or 'Step ' in line)]
  sm=BASE/'outputs/dissertation_final_audit'/label/'summary.json';j=json.loads(sm.read_text());rec['frozen_summary_file']=stat(sm);rec['frozen_summary']={k:v for k,v in j.items() if k!='by_context'}
  norm=BASE/'assets/hrc'/asset/'norm_stats.json';rec['repository_norm']=stat(norm)
  if p.is_dir():rec['checkpoint_norm_matches_repository']=sha(norm)==sha(p/'assets'/asset/'norm_stats.json')
  d['models'][label]=rec
 for rel in ['src/openpi/training/config.py','src/openpi/models/model.py','src/openpi/models/pi0_config.py','src/openpi/policies/policy_config.py','scripts/hrc/run_dissertation_final_audit.py','scripts/hrc/dissertation_protocol.py','scripts/hrc/evaluate_hrc_pi05_continuous.py','scripts/b1k/train_b1k.py','outputs/dissertation_final_audit/anchor_manifest.json','outputs/dissertation_final_audit/README.md','logs/hrc/run_c_qnorm.sh','wandb/run-20260810_010736-rk0qfy4t/files/config.yaml','wandb/run-20260810_010736-rk0qfy4t/files/wandb-metadata.json']:
  d['evidence_files'].append(stat(BASE/rel))
 op.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'output':str(op),'models':{m:{'exists':r['path_check']['exists'],'bytes':r.get('total_file_bytes'),'files':r.get('file_count')} for m,r in d['models'].items()},'search_directories':sum(x['directories_visited'] for x in d['search']),'archives':len(archives)}))
if __name__=='__main__':main()
