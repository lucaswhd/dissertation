#!/usr/bin/env python3
"""Restore existing A20 params and bind them to the current pi0.5 architecture.
No inference: exact historical source/environment identity is not established.
Read-only checkpoint, no CheckpointManager, saving, optimizer restore or downloads.
"""
import datetime,hashlib,json,os,pathlib,sys,time,traceback
ROOT=pathlib.Path('/root/dissertation_evidence_verification')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['JAX_PLATFORMS']='cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['JAX_ENABLE_COMPILATION_CACHE']='false'
os.environ['JAX_COMPILATION_CACHE_DIR']=str(ROOT/'checkpoint_tmp/jax_cache')
os.environ['TMPDIR']=str(ROOT/'checkpoint_tmp')
os.environ['HF_HOME']=str(ROOT/'checkpoint_tmp/hf')
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['HF_HUB_OFFLINE']='1'
sys.path.insert(0,'/root/autodl-tmp/openpi_hrc/src')
sys.path.insert(0,'/root/autodl-tmp/openpi_hrc')
result={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':sys.argv,'checkpoint':'/root/autodl-tmp/openpi_hrc/checkpoints/pi05_hrc_e6_rerun_20k/hrc_e6_rerun_seed42/20000','config':'pi05_hrc_e6_rerun_20k','environment':{k:os.environ[k] for k in ['JAX_PLATFORMS','XLA_PYTHON_CLIENT_PREALLOCATE','JAX_ENABLE_COMPILATION_CACHE','JAX_COMPILATION_CACHE_DIR','TMPDIR','HF_HOME','TRANSFORMERS_OFFLINE','HF_HUB_OFFLINE']},'params_restore':'Not Verified','architecture_bind':'Not Verified','inference':'Not Verified','inference_reason':'Exact historical execution source/environment hash lock is absent; no non-comparable inference was run.'}
start=time.monotonic()
try:
 import importlib.metadata as md
 import jax,jax.numpy as jnp,numpy as np
 from flax import traverse_util
 from openpi.models.model import restore_params
 from openpi.training.config import get_config
 result['versions']={k:md.version(k) for k in ['jax','jaxlib','flax','orbax-checkpoint','numpy','torch','transformers']}
 result['devices']=[str(d) for d in jax.devices()]
 c=get_config(result['config']);result['runtime_config']={'seed':c.seed,'num_train_steps':c.num_train_steps,'model':repr(c.model),'data':repr(c.data),'checkpoint_base_dir':c.checkpoint_base_dir}
 print('BEGIN params restore CPU bfloat16',flush=True)
 params=restore_params(pathlib.Path(result['checkpoint'])/'params',dtype=jnp.bfloat16)
 jax.block_until_ready(params)
 result['params_restore']='Verified';result['params_restore_elapsed_secs']=time.monotonic()-start
 flat=traverse_util.flatten_dict(params)
 result['restored_leaves']=[{'path':'/'.join(k),'shape':list(v.shape),'dtype':str(v.dtype),'size':int(v.size),'nbytes':int(v.nbytes)} for k,v in flat.items()]
 result['leaf_count']=len(flat);result['parameter_count']=sum(int(v.size) for v in flat.values());result['logical_bytes']=sum(int(v.nbytes) for v in flat.values())
 print('RESTORED',result['leaf_count'],'leaves',result['logical_bytes'],'logical bytes',flush=True)
 bad=[]
 for k,v in flat.items():
  arr=np.asarray(v).reshape(-1)
  for lo in range(0,len(arr),4*1024*1024):
   if not np.isfinite(arr[lo:lo+4*1024*1024].astype(np.float32)).all():bad.append('/'.join(k));break
 result['nonfinite_leaves']=bad
 t=time.monotonic();print('BEGIN architecture binding (NNX abstract shape + full supplied tree)',flush=True)
 # Avoid default remove_extra_params silently discarding any unexpected checkpoint leaf.
 model=c.model.load(params,remove_extra_params=False)
 result['architecture_bind']='Verified';result['architecture_bind_elapsed_secs']=time.monotonic()-t;result['model_type']=type(model).__module__+'.'+type(model).__name__
 result['model_attributes']={k:getattr(model,k) for k in ['action_dim','action_horizon','max_token_len']}
 result['overall_level_c']='Partially Verified: full params restore + model architecture bind verified; controlled inference not run.'
except Exception as e:
 result['exception_type']=type(e).__name__;result['exception']=str(e);result['traceback']=traceback.format_exc();result['overall_level_c']='Not Verified; runtime failure is not evidence of checkpoint corruption.'
 print(result['traceback'],flush=True)
finally:
 import resource
 result['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 result['elapsed_secs']=time.monotonic()-start;result['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 (ROOT/'evidence/checkpoint_restore_A20.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ['restored_leaves','runtime_config','traceback']},ensure_ascii=False),flush=True)
