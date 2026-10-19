#!/usr/bin/env python3
"""Read-only metadata/chunk audit. Never opens research files in write mode."""
import argparse,csv,hashlib,json,subprocess
from collections import Counter
from pathlib import Path
import h5py
import numpy as np

def dump_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def audit(data_root,out):
    batch=data_root/'hrc_pipeline_traintest/outputs/hrc_batch_rerun_v1'
    export=data_root/'lerobot_datasets/hrc_pi05_continuous_rerun_v1'
    manifest=json.loads((export/'manifest.json').read_text())
    splits=manifest['source_splits'];ids=[i for v in splits.values() for i in v]
    assert len(ids)==len(set(ids))==45
    qc=json.loads((batch/'rescan_qc.json').read_text())
    clean=sorted(x['idx'] for x in qc['episodes'] if x['verdict']=='CLEAN')
    assert clean==sorted(ids)
    stats=[];trans=Counter();freq=Counter();issues=[];checked=0;video_counts=[]
    for src in manifest['sources']:
        sid=src['source_episode_id'];ep=batch/f'ep_{sid:04d}'
        total=0;seq=[]
        with h5py.File(ep/'hrc_dataset.hdf5','r') as f:
            states=json.loads(f['data'].attrs['fsm_states'])
            for dn in ('demo_0','demo_1'):
                d=f['data/'+dn];n=len(d['action']);total+=n
                assert d['action'].shape==(n,23)
                assert d['obs/joint_positions'].shape==(n,28)
                fsm=d['obs/fsm_state_id'][:];freq.update(map(int,fsm));seq.extend(map(int,fsm))
                for name,ds in d.items():
                    if isinstance(ds,h5py.Dataset):
                        if len(ds)!=n:issues.append(f'source{sid}/{dn}/{name}:length mismatch')
                def visit(name,ds):
                    nonlocal checked
                    if not isinstance(ds,h5py.Dataset) or ds.dtype.kind not in 'fiu':return
                    if ds.shape and ds.shape[0]!=n:issues.append(f'source{sid}/{dn}/{name}:length mismatch')
                    for st in range(0,n,2048):
                        block=ds[st:st+2048];checked+=block.size
                        if not np.isfinite(block).all():issues.append(f'source{sid}/{dn}/{name}:nonfinite')
                d.visititems(visit)
        changes=[seq[i] for i in range(len(seq)) if i==0 or seq[i]!=seq[i-1]]
        trans.update(zip(changes[:-1],changes[1:]))
        assert total==src['num_frames']
        video=ep/'videos/head_rgb.mp4'
        rr=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,r_frame_rate,width,height','-of','json',str(video)],capture_output=True,text=True,check=True)
        vi=json.loads(rr.stdout)['streams'][0];nv=int(vi['nb_frames']);assert nv==total
        video_counts.append(nv)
        stats.append({'source_id':sid,'split':src['split'],'frames':total,'seconds':round(total/30,4),'round0_frames':src['demo0_length'],'round1_frames':src['demo1_length'],'video_frames':nv,'video_width':vi['width'],'video_height':vi['height'],'video_fps':vi['r_frame_rate'],'valid_anchors':src['num_valid_anchors']})
    out.mkdir(parents=True,exist_ok=True)
    dump_csv(out/'source_statistics.csv',stats)
    dump_csv(out/'fsm_frequency.csv',[{'state_id':i,'state':states[i],'frames':freq[i]} for i in range(len(states))])
    dump_csv(out/'fsm_transition_counts.csv',[{'from':states[a],'to':states[b],'count':n} for (a,b),n in sorted(trans.items())])
    split_stats=[{'split':sp,'sources':len(v),'frames':sum(r['frames'] for r in stats if r['split']==sp)} for sp,v in splits.items()]
    dump_csv(out/'split_summary.csv',split_stats)
    result={'sources':len(stats),'frames':sum(x['frames'] for x in stats),'seconds':sum(x['frames'] for x in stats)/30,'split':split_stats,'qc':dict(Counter(x['verdict'] for x in qc['episodes'])),'numeric_values_checked':checked,'issues':issues,'video_metadata_matches':len(video_counts),'method':'All 45 retained HDF5 files; numeric datasets scanned in chunks of 2048 rows; ffprobe container metadata (not full per-frame decoding).'}
    (out/'dataset_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();audit(a.data_root,a.out)
