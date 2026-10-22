#!/usr/bin/env python3
"""Generate only v8 Figures 3.6/5.4 from measured records; no training/audit rerun.

Extraction reads saved segment annotations and numeric columns of the final B/C
Parquet export. Model metrics read fifteen saved per-seed summaries. --from-tables
regenerates the figures from the v8 numerical CSVs without the large local inputs.
No existing v7 figure, evidence or audit file is overwritten.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter, PercentFormatter

ROOT = Path(__file__).resolve().parents[2]
CONTEXTS = ('wait', 'pickup', 'follow', 'radio')
MODELS = ('A20', 'B20', 'C20')
BLUE, ORANGE, GREEN = '#2c5368', '#bc7044', '#718470'
SEGMENTS = [
    ('Walk to the trash can and pick it up.', 'Acquire trash can'),
    ('Pick up the soda can.', 'Pick up soda can'),
    ('Drop the can into the trash can you are holding.', 'Deposit soda can'),
    ('Put the trash can down.', 'Put trash can down'),
    ('Turn on the radio receiver.', 'Radio interaction'),
]
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8.8,
    'axes.labelsize': 8.6, 'xtick.labelsize': 8, 'ytick.labelsize': 8.1,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titleweight': 'bold', 'axes.titlesize': 9.1,
    'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'savefig.facecolor': 'white', 'axes.linewidth': .7,
})


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        out = csv.DictWriter(stream, fieldnames=list(rows[0]))
        out.writeheader(); out.writerows(rows)


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def quantiles(values):
    x = np.asarray(values, float)
    return dict(n=int(len(x)), minimum=float(x.min()), p10=float(np.quantile(x,.1)),
                q25=float(np.quantile(x,.25)), median=float(np.median(x)),
                q75=float(np.quantile(x,.75)), p90=float(np.quantile(x,.9)), maximum=float(x.max()))


def extract(data_root, table_root):
    import pyarrow.parquet as pq
    base = data_root / 'lerobot_datasets'
    dsroot = base / 'hrc_subinstr_rerun_v1'
    manifest_path = dsroot / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    plan_path = Path(manifest['subinstruction_plan'])
    plan = json.loads(plan_path.read_text())
    # Resolve physical units from the original asset without copying that asset.
    import xml.etree.ElementTree as ET
    urdf_path = data_root / 'datasets/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf'
    joint_types = {j.attrib['name']: j.attrib['type'] for j in ET.parse(urdf_path).getroot().findall('joint')}
    joint_names = ([f'torso_joint{i}' for i in range(1,5)] +
                   [f'left_arm_joint{i}' for i in range(1,8)] +
                   [f'right_arm_joint{i}' for i in range(1,8)])
    units_evidence = [{'joint': name, 'type': joint_types[name], 'position_unit': 'rad'} for name in joint_names]
    assert all(joint_types[name] == 'revolute' for name in joint_names)
    sources = {int(s['source_episode_id']): s for s in manifest['sources']}
    assert len(sources) == 45
    provenance = [{'role':'source_identity_only_not_valid_counts', 'path':str(manifest_path), 'sha256':digest(manifest_path)},
                  {'role':'measured_segment_boundaries','path':str(plan_path),'sha256':digest(plan_path)},
                  {'role':'local_joint_type_unit_verification_only_not_redistributed','path':str(urdf_path),'sha256':digest(urdf_path)}]
    # True segment observations, including repeated can cycles; no pseudo-observations
    # are reconstructed from aggregate quantiles.
    segments = []
    for sid, spec in sorted(sources.items()):
        for rnd, rp in sorted(plan['sources'][str(sid)].items()):
            for i, seg in enumerate(rp['segments']):
                if seg['label'] not in dict(SEGMENTS):
                    assert 'Stand by.' in seg['label']
                    continue
                length = int(seg['stop']) - int(seg['start'])
                assert length > 0
                segments.append(dict(source_id=sid, split=spec['split'], round_id=int(rnd),
                                     segment_index=i, label=seg['label'], start_frame=int(seg['start']),
                                     stop_frame_exclusive=int(seg['stop']), duration_frames=length,
                                     duration_seconds=length/30.0, fps=30))
    assert len(segments) == 495
    write_csv(table_root / 'v8_segment_durations.csv', segments)
    rows = []
    for split in ('train','val','test'):
        paths = sorted((dsroot / split / 'data').rglob('*.parquet'))
        assert paths
        # Projected numeric-only read: no RGB/video access and no whole-source audit.
        cols = ['observation.state','action','hrc.source_episode_id','hrc.context_id',
                'hrc.valid_anchor','hrc.task_id','hrc.raw_global_frame_index']
        arrays = []
        for path in paths:
            arrays.append(pq.read_table(path, columns=cols))
            provenance.append({'role':'numeric_characteristic_extraction','path':str(path),'sha256':digest(path)})
        import pyarrow as pa
        tb = pa.concat_tables(arrays)
        def col(name):
            c=tb[name].combine_chunks()
            if name in ('observation.state','action'):
                return c.values.to_numpy(zero_copy_only=False).reshape(-1,23)
            return c.to_numpy(zero_copy_only=False)
        source = col('hrc.source_episode_id'); context=col('hrc.context_id')
        valid=col('hrc.valid_anchor').astype(bool); decision=col('hrc.task_id')==1
        frames=col('hrc.raw_global_frame_index'); state=col('observation.state'); action=col('action')
        arm_dims=np.array([i for i in range(23) if i not in (0,1,2,14,22)])
        magnitude=np.mean(np.abs(action[:,arm_dims]-state[:,arm_dims]),axis=1)
        assert np.isfinite(magnitude).all()
        for sid in sorted(manifest['source_splits'][split]):
            ix=np.flatnonzero(source==sid)
            ix=ix[np.argsort(frames[ix])]
            assert np.array_equal(frames[ix],np.arange(len(ix)))
            # Only rows whose next raw frame remains within this source/context;
            # remove the last raw row of each context, including the sentinel.
            next_same=np.zeros(len(ix),bool)
            next_same[:-1]=context[ix[:-1]]==context[ix[1:]]
            for cid,name in enumerate(CONTEXTS):
                mask=context[ix]==cid; kept=ix[mask & next_same]
                assert len(kept)>0
                rows.append(dict(source_id=sid, split=split, context=name,
                                 raw_frames=int(mask.sum()), bc_valid_anchors=int(valid[ix[mask]].sum()),
                                 bc_decision_valid_anchors=int((valid[ix[mask]] & decision[ix[mask]]).sum()),
                                 within_context_next_action_rows=len(kept),
                                 source_mean_absolute_arm_delta_rad=float(magnitude[kept].mean()),
                                 arm_dimensions=18))
        del tb,arrays,state,action,magnitude
    assert sum(r['raw_frames'] for r in rows)==409670
    assert sum(r['bc_valid_anchors'] for r in rows)==387544
    # Count actual mask values; the B manifest carries stale A anchor statistics.
    old=json.loads((table_root/'decision_anchor_summary.json').read_text())
    for split in ('train','val','test'):
        sr=[r for r in rows if r['split']==split]
        assert sum(r['bc_valid_anchors'] for r in sr)==old[split]['num_valid_anchors']
        assert sum(r['bc_decision_valid_anchors'] for r in sr)==old[split]['num_decision_anchors']
    write_csv(table_root/'v8_source_context_characteristics.csv',rows)
    context_summary=[]
    for name in CONTEXTS:
        sr=[r for r in rows if r['context']==name]
        q=quantiles([r['source_mean_absolute_arm_delta_rad'] for r in sr])
        context_summary.append(dict(context=name, sources=len(sr),raw_frames=sum(r['raw_frames'] for r in sr),
                                    bc_valid_anchors=sum(r['bc_valid_anchors'] for r in sr),
                                    bc_decision_valid_anchors=sum(r['bc_decision_valid_anchors'] for r in sr),
                                    within_context_next_action_rows=sum(r['within_context_next_action_rows'] for r in sr),
                                    **{'arm_delta_'+k:v for k,v in q.items()}))
    write_csv(table_root/'v8_context_characteristic_summary.csv',context_summary)
    duration_summary=[]
    for label,short in SEGMENTS:
        sr=[s for s in segments if s['label']==label]
        duration_summary.append(dict(label=label,display_label=short,unit='seconds',
                                     source_count=len({s['source_id'] for s in sr}),
                                     **quantiles([s['duration_seconds'] for s in sr])))
    write_csv(table_root/'v8_segment_duration_summary.csv',duration_summary)
    # Saved scores are pooled within context at each inference seed; recompute the
    # five-seed mean and sample SD instead of copying formatted thesis numbers.
    seeds=[];model_metrics=[]
    manifest_file=ROOT/'evidence/frozen_audit/anchor_manifest.json'
    expected_manifest_sha=digest(manifest_file)
    frozen_manifest=json.loads(manifest_file.read_text())
    assert frozen_manifest['num_anchors']==652 and frozen_manifest['source_ids']==[0,2,15,18]
    provenance.append({'role':'matched_validation_anchor_manifest','path':str(manifest_file),'sha256':expected_manifest_sha})
    frozen=data_root/'openpi_hrc/outputs/dissertation_final_audit'
    for model in MODELS:
        for seed in range(5):
            path=frozen/model/'per_seed'/f'seed_{seed}'/'summary.json'
            d=json.loads(path.read_text())
            assert d['flow_seed']==seed and d['num_anchors']==652
            assert d['manifest_sha256']==expected_manifest_sha
            provenance.append({'role':'frozen_inference_seed_summary','path':str(path),'sha256':digest(path)})
            for context in ('pickup','radio'):
                g=d['by_context'][context]['gripper']
                assert d['by_context'][context]['num_anchors']==200
                for metric in ('f1','precision','recall'):
                    seeds.append(dict(model=model,context=context,metric=metric,flow_seed=seed,value=g[metric],
                                      matched_context_anchors=200,training_seed=42,training_runs=1,
                                      expert_events=g['expert_events'],predicted_events=g['predicted_events'],
                                      true_positives=g['true_positives']))
        archived_path=ROOT/'evidence/frozen_audit'/model/'summary.json'
        archived=json.loads(archived_path.read_text())
        provenance.append({'role':'included_frozen_summary_cross_check','path':str(archived_path),'sha256':digest(archived_path)})
        for context in ('pickup','radio'):
            for metric in ('f1','precision','recall'):
                vals=[float(r['value']) for r in seeds if r['model']==model and r['context']==context and r['metric']==metric]
                assert len(vals)==5
                mean=float(np.mean(vals));sd=float(np.std(vals,ddof=1))
                old=archived['by_context'][context]['metrics']['grippers.'+metric]
                assert abs(mean-old['mean'])<1e-12 and abs(sd-old['std'])<1e-12
                model_metrics.append(dict(model=model,context=context,metric=metric,mean=mean,sample_sd=sd,
                                          inference_flow_seeds=5,training_runs=1,training_seed=42,
                                          matched_context_anchors=200,validation_sources=4))
    write_csv(table_root/'v8_gripper_event_per_seed.csv',seeds)
    write_csv(table_root/'v8_gripper_event_summary.csv',model_metrics)
    sanity={('pickup','A20'):.1120,('pickup','B20'):.3358,('pickup','C20'):.3825,
            ('radio','A20'):.0235,('radio','B20'):.1768,('radio','C20'):.2181}
    for r in model_metrics:
        if r['metric']=='f1':assert abs(r['mean']-sanity[(r['context'],r['model'])])<.00005
    result={'scope':'Descriptive dataset characteristic extraction and saved-seed score aggregation only; no v7 audit rerun, inference, training or simulation.',
            'dataset_population':{'sources':45,'train_val_test_sources':[37,4,4],'raw_frames':409670,
                                  'nonstandby_segments':495,'source_context_statistics':len(rows),
                                  'within_context_next_action_rows':sum(r['within_context_next_action_rows'] for r in rows)},
            'definitions':{
                'segment_duration_seconds':'(exclusive stop raw frame - start raw frame) / 30; actual measured subinstruction instances; five nonstandby labels.',
                'arm_delta':'Mean over 18 torso/arm dimensions of abs(exported physical shift1 target[t] - current physical state[t]); aggregate mean within each source/context, excluding last raw row of every context. Joint position units rad; no normalization or angle wrapping.',
                'context_counts':'Raw frames versus eligible B/C anchors counted from actual Parquet mask; decision subset is valid_anchor & task_id==1. All 45 retained sources, all splits descriptive.',
                'boxplots':'Median, 25th/75th percentiles; whiskers at measured-sample 10th/90th percentiles; outliers outside whiskers not drawn; no confidence intervals.',
                'gripper_metrics':'At each seed, context-pooled event counts greedily matched by gripper/direction within +/-8 frames; then mean and sample SD (ddof=1) across five inference seeds. One training run per model, seed42.',
                'uncertainty_limit':'No training-seed uncertainty or significance/causal claim; SD bars are not confidence intervals and may extend below zero.'},
            'generator':{'path':'tools/plotting/generate_v8_evidence.py','sha256':digest(Path(__file__))},
            'joint_unit_evidence':units_evidence,
            'input_provenance':provenance,'checks':{'archived_summary_max_difference_tolerance':1e-12,'F1_four_decimal_sanity_check':'passed',
                                                   'actual_B_valid_mask_counts_match_existing_v7_summary':True},
            'figure_paths':{'3.6':'figures/dataset/fig_3_6_temporal_action_characteristics',
                            '5.4':'figures/results/fig_5_4_gripper_event_performance'}}
    (table_root/'v8_evidence_figure_statistics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')


def save(fig, root, subdir, name):
    target=root/'figures'/subdir;target.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw()
    # Fixed 5.7-inch canvas, so text sizes match the intended A4 embedding width.
    renderer=fig.canvas.get_renderer();fb=fig.bbox
    outside=[]
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():continue
        bb=text.get_window_extent(renderer)
        if bb.x0<-.5 or bb.y0<-.5 or bb.x1>fb.x1+.5 or bb.y1>fb.y1+.5:
            outside.append(text.get_text())
    if outside:raise AssertionError('Text outside canvas: '+repr(outside))
    layout_elements=list(fig.texts)+list(fig.legends)
    for ax in fig.axes:
        layout_elements.extend([ax.title,ax.xaxis.label])
        if ax.get_legend() is not None:layout_elements.append(ax.get_legend())
    for i,left in enumerate(layout_elements):
        for right in layout_elements[i+1:]:
            if left.get_window_extent(renderer).overlaps(right.get_window_extent(renderer)):
                raise AssertionError('Title/legend/axis-label/figure-note overlap detected')
    for ext in ('png','pdf','svg'):
        fig.savefig(target/f'{name}.{ext}',dpi=400,facecolor='white')
    plt.close(fig)


def draw_dataset(root):
    tables=root/'tables'
    segments=read_csv(tables/'v8_segment_durations.csv')
    characteristics=read_csv(tables/'v8_source_context_characteristics.csv')
    counts=read_csv(tables/'v8_context_characteristic_summary.csv')
    fig=plt.figure(figsize=(5.7,6.35))
    gs=fig.add_gridspec(3,1,height_ratios=[1.25,1,1.2],left=.29,right=.97,top=.90,bottom=.13,hspace=.90)
    ax=fig.add_subplot(gs[0]);values=[];labels=[]
    for long,short in SEGMENTS:
        values.append([float(r['duration_seconds']) for r in segments if r['label']==long])
        labels.append(f'{short}  (n={len(values[-1])})')
    ax.boxplot(values,orientation='horizontal',whis=(10,90),showfliers=False,patch_artist=True,widths=.54,
               boxprops={'facecolor':'#e6eef1','edgecolor':BLUE,'linewidth':1},
               medianprops={'color':ORANGE,'linewidth':1.5},whiskerprops={'color':BLUE},capprops={'color':BLUE})
    ax.set_yticks(range(1,6),labels);ax.invert_yaxis();ax.set_xscale('log')
    ax.set_xlim(.6,120);ax.set_xticks([1,3,10,30,100],['1','3','10','30','100'])
    ax.xaxis.set_minor_formatter(NullFormatter());ax.set_xlabel('Segment duration (s; log scale)')
    ax.set_title('a  Non-standby segment durations',loc='left',pad=9)
    ax.grid(axis='x',which='major',alpha=.18)
    ax=fig.add_subplot(gs[1]);vals=[]
    for context in CONTEXTS:
        vals.append([float(r['source_mean_absolute_arm_delta_rad']) for r in characteristics if r['context']==context])
    ax.boxplot(vals,orientation='horizontal',whis=(10,90),showfliers=False,patch_artist=True,widths=.54,
               boxprops={'facecolor':'#e8eee7','edgecolor':GREEN,'linewidth':1},
               medianprops={'color':ORANGE,'linewidth':1.5},whiskerprops={'color':GREEN},capprops={'color':GREEN})
    ax.set_yticks(range(1,5),[c.capitalize() for c in CONTEXTS]);ax.invert_yaxis();ax.set_xscale('log')
    ax.set_xlim(3e-6,.06);ax.set_xticks([1e-5,1e-4,1e-3,1e-2],['10⁻⁵','10⁻⁴','10⁻³','10⁻²'])
    ax.xaxis.set_minor_formatter(NullFormatter());ax.set_xlabel('Mean |target − state| over 18 torso/arm joints (rad)')
    ax.set_title('b  Per-source physical next-action magnitude',loc='left',pad=9)
    ax.grid(axis='x',which='major',alpha=.18)
    ax=fig.add_subplot(gs[2]);y=np.arange(4);h=.32
    raw=np.array([int(r['raw_frames']) for r in counts]);valid=np.array([int(r['bc_valid_anchors']) for r in counts])
    ax.barh(y-h/2,raw/1000,height=h,color='#c5ced2',label='Raw frames')
    ax.barh(y+h/2,valid/1000,height=h,color=BLUE,label='B/C valid anchors')
    ax.set_yticks(y,[c.capitalize() for c in CONTEXTS]);ax.invert_yaxis();ax.set_xlabel('Frame / anchor counts (thousands)')
    ax.set_xlim(0,max(raw)/1000*1.12)
    ax.set_xticks(np.arange(0, max(raw)/1000*1.12, 50));ax.grid(axis='x',alpha=.18)
    ax.set_title('c  Raw frames and eligible B/C anchors',loc='left',pad=27)
    ax.legend(frameon=False,ncol=2,fontsize=7.4,loc='lower right',bbox_to_anchor=(1.01,1.005),borderaxespad=0)
    # The ratio column is outside the bars but inside the fixed figure canvas.
    for i,(ra,va) in enumerate(zip(raw,valid)):
        ax.text(max(ra,va)/1000+2,i,f'{va/ra:.0%}',ha='left',va='center',fontsize=7.5,color='#3c4950')
    fig.text(.03,.989,'45 retained sources · all splits · descriptive dataset statistics',ha='left',va='top',fontsize=8.4,color='#45535a')
    fig.text(.03,.020,'Boxes: median and IQR; whiskers: 10th–90th percentiles.\nPanel c labels: valid/raw frame ratio.',fontsize=7.3,color='#45535a')
    save(fig,root,'dataset','fig_3_6_temporal_action_characteristics')


def draw_events(root):
    stats=read_csv(root/'tables/v8_gripper_event_summary.csv')
    fig,axs=plt.subplots(1,2,figsize=(5.7,2.95),sharey=True)
    fig.subplots_adjust(left=.115,right=.98,top=.77,bottom=.22,wspace=.20)
    styles={'f1':dict(color=BLUE,marker='o',mfc=BLUE,ls='-',label='F1'),
            'precision':dict(color=ORANGE,marker='D',mfc='white',ls='--',label='Precision')}
    for ax,context,letter in zip(axs,('pickup','radio'),('a','b')):
        for metric,offset in [('f1',-.055),('precision',.055)]:
            sr=[next(r for r in stats if r['model']==model and r['context']==context and r['metric']==metric) for model in MODELS]
            style=styles[metric]
            ax.errorbar(np.arange(3)+offset,[float(r['mean']) for r in sr],
                        yerr=[float(r['sample_sd']) for r in sr],capsize=3,elinewidth=1,
                        linewidth=1.15,markersize=5,markeredgewidth=1,**style)
        ax.set_xticks(range(3),MODELS);ax.set_xlim(-.35,2.35);ax.set_ylim(-.05,.50)
        ax.set_yticks([0,.1,.2,.3,.4,.5]);ax.grid(axis='y',alpha=.18)
        ax.axhline(0,color='#7d898e',lw=.6);ax.set_title(f'{letter}  {context.capitalize()}',loc='left',pad=9)
        ax.set_xlabel('20k-step checkpoint')
    axs[0].set_ylabel('Gripper-event score')
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.54,.965),ncol=2,frameon=False,fontsize=8.4)
    fig.text(.5,.043,'Mean ± sample SD over 5 inference seeds; one training run per checkpoint.',ha='center',fontsize=7.5,color='#45535a')
    save(fig,root,'results','fig_5_4_gripper_event_performance')


def write_figure_metadata(root):
    metadata = {
        '3.6': {
            'png': 'figures/dataset/fig_3_6_temporal_action_characteristics.png',
            'lead': 'Figure 3.6 shows why total frame count alone does not describe the supervision available to the learner. Acquisition and soda-can pickup segments last longer than deposit or radio-interaction segments, while next-action targets change much less in wait and follow than in the two manipulation contexts. Eligibility for a complete action horizon further changes the available anchor population. These descriptive differences motivate RQ1: chunk construction and sampling determine which recorded rows contribute to learning. Repeated segments within each source are correlated.',
            'caption': 'Figure 3.6: Temporal and action characteristics of the 45 retained HRC sources (all splits, descriptive). (a) Durations of 495 measured non-standby sub-instruction segments; n counts actual segment instances, including repeated cycles. (b) Across-source distributions of the mean absolute difference between next-action targets and current states over 18 revolute torso/arm joints, with 45 sources per context. Boxes show medians and interquartile ranges; whiskers use the 10th–90th percentiles, with more extreme values omitted. (c) Recorded frames and eligible B/C anchors from actual masks; labels show valid/raw ratios. Panels (a,b) use logarithmic axes.',
            'population': {'sources':45, 'source_split':[37,4,4], 'measured_nonstandby_segments':495, 'raw_frames':409670, 'eligible_B_C_anchors':387544},
            'width_inches':5.7, 'height_inches':6.35,
            'variables': {'a':'Actual segment duration, seconds', 'b':'Per-source mean absolute next-action target minus state across 18 revolute torso/arm joints, radians', 'c':'Raw frame and eligible anchor counts; all source splits descriptive'},
            'evidence': ['tables/v8_segment_durations.csv', 'tables/v8_source_context_characteristics.csv', 'tables/v8_context_characteristic_summary.csv']
        },
        '5.4': {
            'png': 'figures/results/fig_5_4_gripper_event_performance.png',
            'lead': 'Figure 5.4 compares the event-detection profiles in the two manipulation contexts. B20 and C20 have higher mean F1 than A20 on the frozen validation anchors, but precision remains limited: the highest pickup precision is 0.3163. Radio scores also vary substantially with inference noise. These differences do not establish successful grasping, a single-factor causal effect, or statistical significance. A20 to B20 changes prompts and sampling together; B20 to C20 changes state and action normalisation together.',
            'caption': 'Figure 5.4: Gripper-event prediction performance across A20, B20 and C20 for (a) pickup and (b) radio. Points show mean F1 and precision; bars show sample standard deviations over five inference flow-noise seeds. Each seed pools event counts over 200 matched validation anchors per context. Each checkpoint comes from one training run; the bars are not training-run uncertainty or confidence intervals. Lines guide comparison between configurations. SD bars are untruncated and may extend below zero. Higher event scores do not establish successful grasping.',
            'population': {'full_matched_validation_manifest':652, 'matched_anchors_per_context':200, 'validation_sources':4, 'inference_seeds':5, 'training_runs_per_model':1, 'training_seed':42},
            'width_inches':5.7, 'height_inches':2.95,
            'variables': {'x':'A20/B20/C20 categorical checkpoints at 20000 optimizer updates', 'y':'Dimensionless context-pooled event F1 or precision', 'error_bar':'Sample standard deviation across five inference seeds, ddof=1'},
            'evidence': ['tables/v8_gripper_event_per_seed.csv', 'tables/v8_gripper_event_summary.csv', 'evidence/frozen_audit/anchor_manifest.json']
        }
    }
    (root/'tables/v8_evidence_figure_metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,default=Path('/root/autodl-tmp'))
    ap.add_argument('--output-root',type=Path,default=ROOT)
    ap.add_argument('--from-tables',action='store_true')
    args=ap.parse_args();tables=args.output_root/'tables';tables.mkdir(parents=True,exist_ok=True)
    if not args.from_tables:extract(args.data_root,tables)
    draw_dataset(args.output_root);draw_events(args.output_root)
    write_figure_metadata(args.output_root)
    print('Generated only v8 Figures 3.6/5.4 and v8_ evidence tables.')

if __name__=='__main__':main()
