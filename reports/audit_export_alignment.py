#!/usr/bin/env python3
"""Independent local HRC HDF5 -> A/B Parquet alignment audit; no project imports.

Read one source at a time, numeric columns only. No models, training, simulation,
video decoding or writes to source files. Only --output is written.
"""
from pathlib import Path
import argparse, collections, datetime, hashlib, json
import h5py
import numpy as np
import pyarrow.dataset as pads


def sha256(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset-base', type=Path, default=Path('/root/autodl-tmp/lerobot_datasets'))
    ap.add_argument('--output', type=Path, default=Path('/root/dissertation_release_staging/reports/export_alignment_checks.json'))
    args = ap.parse_args()
    roots = {'A': args.dataset_base / 'hrc_pi05_continuous_rerun_v1',
             'B_C': args.dataset_base / 'hrc_subinstr_rerun_v1'}
    manifests = {k: json.loads((p / 'manifest.json').read_text()) for k, p in roots.items()}
    plan_path = Path(manifests['B_C']['subinstruction_plan'])
    plan = json.loads(plan_path.read_text())
    r = {'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'method': 'Independent NumPy/PyArrow/h5py reconstruction, one HDF5 source at a time; no project alignment functions imported.',
         'scope': 'All 45 retained sources and both complete A/B_C exports across train/val/test; numeric columns only.',
         'experiment_execution': 'Training, model inference and simulation: Not independently verified',
         'horizon': 32, 'action_shift': 1, 'exact_float32_comparison': True,
         'input_manifests': {k: {'path': str(p / 'manifest.json'), 'sha256': sha256(p / 'manifest.json')} for k,p in roots.items()},
         'subinstruction_plan': {'path': str(plan_path), 'sha256': sha256(plan_path)},
         'totals': {'sources': 0, 'unique_raw_rows': 0, 'exported_rows_A_plus_B': 0,
                    'state_action_numeric_values_checked': 0, 'nonfinite_state_action_values': 0,
                    'state_raw_max_abs_diff': 0.0, 'action_raw_shift1_max_abs_diff': 0.0,
                    'A_B_state_max_abs_diff': 0.0, 'A_B_action_max_abs_diff': 0.0,
                    'source_sentinels_checked_per_export': 0, 'valid_anchors_A': 0, 'valid_anchors_B_C': 0,
                    'horizon_violations_A': 0, 'horizon_violations_B_C': 0,
                    'decision_anchors_B_C': 0, 'demo_boundaries_checked_per_export': 0},
         'splits': {}, 'sources': [], 'failures': [], 'metadata_warnings': [],
         'limitations': ['No decoded image/frame equality, exposure-time or simulator-clock synchronization check.',
                        'Parquet timestamps are nominal frame_index/30; agreement cannot establish physical sensor/action synchronization.',
                        'B idle-tail and decision-flag audit checks adherence to stored annotation plan; semantic correctness of that plan is not independently re-annotated.',
                        'Source HDF5 command/action semantics are taken from the recorded schema; no simulator execution is reproduced.']}
    def check(ok, name, source=None, export=None, detail=None):
        if not bool(ok):
            r['failures'].append({'check': name, 'source': source, 'export': export, 'detail': detail})
    def arr(table, key, dtype=None):
        col = table[key].combine_chunks()
        if key in ('observation.state','action'):
            return np.asarray(col.values.to_numpy(zero_copy_only=False), dtype=dtype).reshape(len(table),23)
        return np.asarray(col.to_numpy(zero_copy_only=False), dtype=dtype)
    check(manifests['A']['source_splits']==manifests['B_C']['source_splits'], 'source split manifest equality')
    source_specs = {int(s['source_episode_id']): s for s in manifests['A']['sources']}
    cols = ['observation.state','action','hrc.valid_anchor','hrc.task_id','hrc.context_id',
            'hrc.source_episode_id','hrc.fsm_state_id','hrc.raw_global_frame_index','hrc.raw_demo_id',
            'hrc.raw_demo_frame_index','timestamp','frame_index','episode_index','index']
    for split in ('train','val','test'):
        sources = sorted(manifests['A']['source_splits'][split])
        datasets = {k: pads.dataset(str(p/split/'data'),format='parquet') for k,p in roots.items()}
        sd = {'source_ids': sources, 'rows_per_export': 0, 'valid_anchors_A': 0,
              'valid_anchors_B_C': 0, 'decision_anchors_B_C': 0,
              'parquet_inputs': {k: [{'path': str(f), 'bytes': f.stat().st_size, 'sha256': sha256(f)}
                                    for f in sorted((p/split/'data').rglob('*.parquet'))] for k,p in roots.items()}}
        for k, ds in datasets.items():
            check(ds.count_rows()==sum(source_specs[s]['num_frames'] for s in sources), 'split total row count',export=k,detail=split)
        offset = 0
        for episode_no, sid in enumerate(sources):
            spec = source_specs[sid];hp=Path(spec['hdf5_path'])
            with h5py.File(hp,'r') as hf:
                ds0,ds1=hf['data/demo_0'],hf['data/demo_1'];n0,n1=len(ds0['action']),len(ds1['action']);n=n0+n1
                raw=np.concatenate([ds0['action'][:],ds1['action'][:]]).astype(np.float32)
                joints=np.concatenate([ds0['obs/joint_positions'][:],ds1['obs/joint_positions'][:]]).astype(np.float32)
                grip=np.concatenate([ds0['obs/gripper_state'][:],ds1['obs/gripper_state'][:]]).astype(np.float32)
                fsm=np.concatenate([ds0['obs/fsm_state_id'][:],ds1['obs/fsm_state_id'][:]]).astype(np.int64)
            commands={int(c['round_id']):c for c in json.loads(Path(spec['commands_path']).read_text())}
            c0,c1=int(commands[0]['issued_frame']),int(commands[1]['issued_frame'])
            check((n0,n1,n)==(spec['demo0_length'],spec['demo1_length'],spec['num_frames']),'manifest source lengths',sid)
            check((c0,c1)==(spec['round0_issued_global'],spec['round1_issued_global']),'manifest command boundaries',sid)
            base=raw[:,:3].copy();base[0]=0
            state=np.column_stack([base,joints[:,6:10],joints[:,[10,12,14,16,18,20,22]],
                                   grip[:,:2].sum(axis=1),joints[:,[11,13,15,17,19,21,23]],grip[:,2:].sum(axis=1)]).astype(np.float32)
            action=np.vstack([raw[1:],raw[-1:]])
            context=np.select([np.arange(n)<c0,np.arange(n)<n0,np.arange(n)<c1],[0,1,2],default=3).astype(np.int64)
            expected_valid=np.zeros(n,bool)
            # Four contiguous coarse-context intervals: t+32 must exist in same interval.
            for lo,hi in [(0,c0),(c0,n0),(n0,c1),(c1,n)]:
                if hi-lo>32:expected_valid[lo:hi-32]=True
            bvalid=expected_valid.copy();bdecision=np.zeros(n,bool)
            for rp in plan['sources'][str(sid)].values():
                bvalid[int(rp['idle_tail_from']):int(rp['execute_stop'])]=False
                for flip in rp['flip_frames']:
                    bdecision[max(int(rp['execute_start']),int(flip)-int(plan['flip_radius'])):
                              min(int(rp['execute_stop']),int(flip)+int(plan['flip_radius'])+1)]=True
            expected_demo=np.r_[np.zeros(n0,np.int64),np.ones(n1,np.int64)]
            expected_demo_frame=np.r_[np.arange(n0),np.arange(n1)]
            exported={};source_result={'source_id':sid,'split':split,'raw_hdf5_path':str(hp),'rows':n,
                                      'round0_frames':n0,'round1_frames':n1,'exports':{}}
            for key,ds in datasets.items():
                tb=ds.to_table(columns=cols,filter=pads.field('hrc.source_episode_id')==sid)
                order=np.argsort(arr(tb,'index',np.int64));tb=tb.take(order)
                a={c:arr(tb,c,np.float32 if c in ('observation.state','action','timestamp') else None) for c in cols}
                exported[key]=a;check(len(tb)==n,'source row count',sid,key)
                if len(tb)!=n:continue
                for col,expected in [('hrc.source_episode_id',np.full(n,sid)),('hrc.raw_global_frame_index',np.arange(n)),
                                     ('frame_index',np.arange(n)),('index',np.arange(offset,offset+n)),
                                     ('episode_index',np.full(n,episode_no)),('hrc.raw_demo_id',expected_demo),
                                     ('hrc.raw_demo_frame_index',expected_demo_frame),('hrc.context_id',context),('hrc.fsm_state_id',fsm)]:
                    check(np.array_equal(a[col],expected),col+' mapping',sid,key)
                st,ac=a['observation.state'],a['action'];finite=int((~np.isfinite(st)).sum()+(~np.isfinite(ac)).sum())
                sm=float(np.max(np.abs(st-state)));am=float(np.max(np.abs(ac-action)))
                check(sm==0,'state23 raw reconstruction',sid,key,sm);check(am==0,'raw action shift1',sid,key,am)
                check(finite==0,'finite state/action',sid,key,finite)
                ev=expected_valid if key=='A' else bvalid;valid=a['hrc.valid_anchor'].astype(bool)
                check(np.array_equal(valid,ev),'complete valid mask reconstruction',sid,key,int(np.count_nonzero(valid!=ev)))
                check(not valid[-1] and np.array_equal(ac[-1],raw[-1]),'last-row invalid repeat-action sentinel',sid,key)
                t=np.flatnonzero(valid);safe=t[t+32<n];bad=int(np.count_nonzero(t+32>=n))
                if len(safe):
                    # Explicitly inspect all 33 positions, not just endpoints.
                    for delta in range(33):bad+=int(np.count_nonzero(context[safe+delta]!=context[safe]))
                check(bad==0,'horizon32 source/coarse-context safety',sid,key,bad)
                ts_expected=(np.arange(n,dtype=np.float64)/30).astype(np.float32)
                td=float(np.max(np.abs(a['timestamp'].astype(np.float64)-np.arange(n)/30)))
                check(np.array_equal(a['timestamp'],ts_expected),'nominal timestamp frame/30',sid,key,td)
                check(np.all(np.diff(a['timestamp'])>0),'monotonic per-source timestamps',sid,key)
                if key=='B_C':check(np.array_equal(a['hrc.task_id'],bdecision.astype(np.int64)),'decision flags vs annotation plan',sid,key)
                sr={'rows':n,'valid_anchors':int(valid.sum()),'state_raw_max_abs_diff':sm,'action_raw_shift1_max_abs_diff':am,
                    'nonfinite_state_action_values':finite,'horizon_violations':bad,'sentinel_valid':bool(valid[-1]),
                    'timestamp_max_abs_diff_from_nominal_seconds':td,'demo_boundary_state_base_matches_current_raw_action':bool(np.array_equal(st[n0,:3],raw[n0,:3]))}
                source_result['exports'][key]=sr
                r['totals']['state_action_numeric_values_checked']+=int(st.size+ac.size)
                r['totals']['nonfinite_state_action_values']+=finite
                r['totals']['state_raw_max_abs_diff']=max(r['totals']['state_raw_max_abs_diff'],sm)
                r['totals']['action_raw_shift1_max_abs_diff']=max(r['totals']['action_raw_shift1_max_abs_diff'],am)
                r['totals']['horizon_violations_'+key]+=bad
                r['totals']['valid_anchors_'+key]+=int(valid.sum());sd['valid_anchors_'+key]+=int(valid.sum())
            for field,outkey in [('observation.state','A_B_state_max_abs_diff'),('action','A_B_action_max_abs_diff')]:
                diff=float(np.max(np.abs(exported['A'][field]-exported['B_C'][field])))
                check(diff==0,'A/B '+field+' equality',sid,detail=diff);r['totals'][outkey]=max(r['totals'][outkey],diff)
            dc=int((bdecision & exported['B_C']['hrc.valid_anchor']).sum());source_result['decision_anchors_B_C']=dc
            sd['decision_anchors_B_C']+=dc;r['totals']['decision_anchors_B_C']+=dc
            actualb=source_result['exports']['B_C']['valid_anchors'];stored=int(spec['num_valid_anchors'])
            bspec=next(s for s in manifests['B_C']['sources'] if int(s['source_episode_id'])==sid)
            if actualb!=int(bspec['num_valid_anchors']):
                r['metadata_warnings'].append({'kind':'B manifest source num_valid_anchors inherited from A',
                                              'source_id':sid,'manifest_count':int(bspec['num_valid_anchors']),'parquet_count':actualb})
            r['sources'].append(source_result);sd['rows_per_export']+=n;offset+=n
            r['totals']['sources']+=1;r['totals']['unique_raw_rows']+=n;r['totals']['exported_rows_A_plus_B']+=2*n
            r['totals']['source_sentinels_checked_per_export']+=1;r['totals']['demo_boundaries_checked_per_export']+=1
            print(f'{split} source {sid:02d}: rows={n} Avalid={int(expected_valid.sum())} Bvalid={actualb}',flush=True)
        r['splits'][split]=sd
    # Metadata only; do not read historical canonical imagery or full numeric arrays.
    cp=Path('/root/autodl-tmp/hrc_canonical/hrc_multitask_v01_step10/canonical.hdf5')
    if cp.exists():
        with h5py.File(cp,'r') as hf:
            inventory=[]
            hf.visititems(lambda name,obj: inventory.append({'key':name,'shape':list(obj.shape),'dtype':str(obj.dtype)}) if isinstance(obj,h5py.Dataset) else None)
            r['historical_canonical']={'path':str(cp),'bytes':cp.stat().st_size,'dataset_inventory':inventory,
                                       'scope':'Metadata only; historical ACT branch, not a mandatory intermediate for final pi0.5 continuous export.'}
    r['status']='passed_with_metadata_warnings' if not r['failures'] and r['metadata_warnings'] else ('passed' if not r['failures'] else 'failed')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':r['status'],'totals':r['totals'],'failures':r['failures'],'metadata_warning_count':len(r['metadata_warnings']),'output':str(args.output)},indent=2))
    return 1 if r['failures'] else 0

if __name__=='__main__':raise SystemExit(main())
