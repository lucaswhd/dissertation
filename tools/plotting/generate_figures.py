#!/usr/bin/env python3
"""Reproduce dissertation additions from actual videos, CSVs and source manifests."""
import argparse,csv,json,subprocess
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.ticker import PercentFormatter
import numpy as np
from PIL import Image

C=['#2c5368','#bc7044','#718470'];ROOT=Path(__file__).resolve().parents[2]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
def read(p):return list(csv.DictReader(p.open()))
def save(fig,sub,stem):
 d=ROOT/'figures'/sub;d.mkdir(parents=True,exist_ok=True)
 for ext in ('png','pdf','svg'):fig.savefig(d/f'{stem}.{ext}',dpi=300,bbox_inches='tight',pad_inches=.08)
 plt.close(fig)
def box(ax,xy,w,h,text,fill='#edf3f5',fs=9):
 p=FancyBboxPatch(xy,w,h,boxstyle='round,pad=0.012,rounding_size=0.014',facecolor=fill,edgecolor='#819099',lw=.8);ax.add_patch(p);ax.text(xy[0]+w/2,xy[1]+h/2,text,ha='center',va='center',fontsize=fs);return p
def arrow(ax,a,b,**kw):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,lw=.95,color='#53606a',**kw))
def storyboard(data):
 m=json.loads((data/'lerobot_datasets/hrc_subinstr_rerun_v1/manifest.json').read_text());s=next(x for x in m['sources'] if x['source_episode_id']==0)
 a=json.loads((data/'lerobot_datasets/hrc_subinstructions_rerun_v1.json').read_text())['sources']['0']
 b=s['demo0_length'];e=s['round1_issued_global']
 items=[('a','Wait for instruction',20,'Scripted FSM',0),('b','Acquire trash can',a['0']['segments'][0]['stop']-10,'Recorded replay',0),('c','Pick up soda can',a['0']['segments'][1]['stop']-10,'Recorded replay',0),('d','Deposit soda can',a['0']['segments'][2]['stop']-10,'Recorded replay',0),('e','Follow human',(b+e)//2,'Scripted FSM',1),('f','Radio interaction',a['1']['segments'][1]['stop']-8,'Recorded replay',1)]
 local=ROOT.parent/'dissertation_figures_delivery/frames';local.mkdir(parents=True,exist_ok=True)
 fig,axs=plt.subplots(2,3,figsize=(7.0,4.35));records=[]
 for ax,(letter,label,fr,control,rnd) in zip(axs.flat,items):
  p=local/f'source0000_frame{fr:06d}.png'
  subprocess.run(['ffmpeg','-v','error','-i',str(s['head_video_path']),'-vf',f'select=eq(n\\,{fr})','-frames:v','1','-y',str(p)],check=True)
  ax.imshow(Image.open(p));ax.set_axis_off();ax.set_title(f'{letter}  {label}',loc='left',fontsize=8.2,pad=5)
  ax.text(0,-.05,f'S0 · round {rnd} · frame {fr} · {fr/30:.2f} s\n{control}',transform=ax.transAxes,fontsize=7.3,va='top')
  records.append({'panel':letter,'source':0,'split':'val','round':rnd,'raw_global_frame':fr,'seconds':fr/30,'camera':'head RGB','control':control,'video':'hrc_batch_rerun_v1/ep_0000/videos/head_rgb.mp4','frame_image':p.name})
 fig.subplots_adjust(wspace=.06,hspace=.45,top=.94,bottom=.12)
 save(fig,'methodology','fig_3_3_task_storyboard')
 (ROOT/'tables/storyboard_frames.json').write_text(json.dumps(records,indent=2))
def architecture():
 fig,ax=plt.subplots(figsize=(7.0,5.0));ax.set_xlim(-.01,1.01);ax.set_ylim(0,1);ax.axis('off')
 ax.text(0,1,'a  Implemented finite-state scheduling',weight='bold',fontsize=10)
 names=[['WaitNearHuman','ReceiveCommand','GoToTask'],['ReturnToHuman','TaskSegmentDone','ExecuteBehaviorTask'],['FaceHuman','WaitNextCommand','Relocate']]
 xs=[.02,.36,.70];ys=[.83,.68,.53]
 for row in range(3):
  for col in range(3):
   n=names[row][col];box(ax,(xs[col],ys[row]),.28,.09,n,fill='#f5eadf' if n=='ExecuteBehaviorTask' else '#edf3f5',fs=7.6)
 for row in (0,2):
  for col in range(2):arrow(ax,(xs[col]+.294,ys[row]+.045),(xs[col+1]-.014,ys[row]+.045))
 for col in (2,1):arrow(ax,(xs[col]-.014,ys[1]+.045),(xs[col-1]+.294,ys[1]+.045))
 arrow(ax,(.84,.816),(.84,.784))
 arrow(ax,(.16,.666),(.16,.634))
 ax.text(.5,.494,'Non-final round: Relocate → WaitNearHuman. Final round: FaceHuman → AllTasksDone.',ha='center',fontsize=7.6)
 ax.text(0,.445,'b  ExecuteBehaviorTask: two controller modes',weight='bold',fontsize=10)
 box(ax,(.015,.325),.46,.09,'Dataset construction\nBEHAVIOR simulator-state / action replay',fill='#f5eadf',fs=8.5)
 box(ax,(.525,.325),.46,.09,'Policy rollout\nπ0.5 client → action clamp → env.step',fill='#e9f0e6',fs=8.5)
 ax.text(.5,.295,'Surrounding human-facing and transition phases remain scripted.',ha='center',fontsize=8)
 ax.text(0,.24,'c  Recorded data to the diagnostic benchmark',weight='bold',fontsize=10)
 texts=['HDF5 states/actions\nRGB video + commands','Source QC and split\n45 CLEAN; 37 / 4 / 4','LeRobot export\nshift-1; 32-step chunks','Training + frozen audit\n652 validation anchors']
 for i,t in enumerate(texts):
  x=.015+i*.25;box(ax,(x,.075),.22,.115,t,fs=6.8)
  if i<3:arrow(ax,(x+.232,.133),(x+.25,.133))
 ax.text(.5,.018,'Replayed demonstrations and policy rollouts are separate evidence sources.',ha='center',fontsize=8,color='#555555')
 save(fig,'methodology','fig_3_4_controller_data_flow')
def construction(data=None):
 checks=json.loads((ROOT/'tables/dataset_checks.json').read_text());rows=read(ROOT/'tables/source_statistics.csv')
 export=json.loads((ROOT/'tables/decision_anchor_summary.json').read_text())
 natural=export['train']['decision_fraction']
 fig=plt.figure(figsize=(7,5.7));gs=fig.add_gridspec(2,2,height_ratios=[1,1.2],hspace=.55,wspace=.5)
 ax=fig.add_subplot(gs[0,:]);labels=['CLEAN (retained)','REJECT (excluded)','TRIM (excluded)'];values=[45,19,1]
 ax.barh(labels,values,color=[C[0],C[1],'#a6a6a6'],height=.6);ax.invert_yaxis();ax.set_xlim(0,70);ax.set_xlabel('Number of source executions');ax.set_title('a  Source-level quality control (n = 65)',loc='left')
 for i,v in enumerate(values):ax.text(v+.6,i,str(v),va='center')
 ax.text(.72,.5,'45 retained sources\n37 train · 4 validation · 4 test\n409,670 frames at 30 Hz',transform=ax.transAxes,va='center',fontsize=9,bbox={'facecolor':'white','edgecolor':'#ccd3d5','pad':8})
 ax=fig.add_subplot(gs[1,0]);dur=[float(r['frames'])/1800 for r in rows];bins=np.arange(0,9.01,1)
 ax.hist([[float(r['frames'])/1800 for r in rows if r['split']==s] for s in ('train','val','test')],bins=bins,stacked=True,color=C,label=['Train','Validation','Test'],edgecolor='white')
 ax.set_xlabel('Complete source duration (min)');ax.set_ylabel('Number of sources');ax.set_xticks(range(0,10,2));ax.set_title('b  Source durations (n = 45)',loc='left',fontsize=9);ax.legend(frameon=False,fontsize=7)
 ax=fig.add_subplot(gs[1,1]);vals=[natural,.2];ax.bar(['Natural exported\nvalid training anchors','Configured B/C\nsampler target'],vals,color=[C[0],C[1]],width=.58)
 for i,v in enumerate(vals):ax.text(i,v+.008,f'{v:.3%}' if i==0 else '20%',ha='center',fontsize=9)
 ax.set_ylim(0,.25);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_ylabel('Decision-frame share');ax.set_title('c  Decision-frame supervision',loc='left',fontsize=8.7);ax.tick_params(axis='x',labelsize=8)
 save(fig,'dataset','fig_3_5_dataset_construction')
def training(data=None):
 lognames={'A20':'train_e6_rerun_20k.log','B20':'train_b_subinstr_20k.log','C20':'train_c_qnorm_20k.log'}
 import re
 fig,axs=plt.subplots(1,3,figsize=(7,2.6),sharey=True)
 allrows=[]
 for ax,(model,fn) in zip(axs,lognames.items()):
  txt=(data/'openpi_hrc/logs/hrc'/fn).read_text(errors='replace') if data is not None else '';curves={}
  for phase,pat in [('Train',r'Step (\d+):[^\n]*?loss=([\d.eE+-]+)'),('Monitored validation',r'Validation loss at step (\d+): ([\d.eE+-]+)')]:
   pairs=[(int(x),float(y)) for x,y in re.findall(pat,txt)] if data is not None else [(int(r['optimizer_step']),float(r['normalized_flow_loss'])) for r in read(ROOT/'tables/training_curves.csv') if r['model']==model and r['phase']==phase];assert len(pairs)==(200 if phase=='Train' else 20)
   ax.plot([x for x,y in pairs],[y for x,y in pairs],lw=1.0,color=C[0] if phase=='Train' else C[1],label=phase)
   allrows.extend({'model':model,'phase':phase,'optimizer_step':x,'normalized_flow_loss':y,'source_log':fn} for x,y in pairs)
  ax.set_title(model);ax.set_xlabel('Optimiser updates');ax.set_xlim(0,20000);ax.set_ylim(0,1);ax.set_xticks([0,10000,20000],['0','10k','20k']);ax.grid(axis='y',alpha=.2)
 axs[0].set_ylabel('Normalised flow objective');axs[0].legend(frameon=False,fontsize=7);fig.tight_layout();save(fig,'experiments','fig_A_1_training_diagnostics')
 with (ROOT/'tables/training_curves.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
def extras():
 rows=read(ROOT/'tables/fsm_frequency.csv');fig,ax=plt.subplots(figsize=(7,3.4));total=sum(int(r['frames']) for r in rows)
 ax.barh([r['state'] for r in rows],[int(r['frames'])/total for r in rows],color=C[0]);ax.invert_yaxis();ax.set_xlabel('Fraction of recorded frames (45 retained sources)');ax.xaxis.set_major_formatter(PercentFormatter(1));fig.tight_layout();save(fig,'dataset','supp_fsm_state_frequency')
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path);p.add_argument('--from-release-tables',action='store_true');a=p.parse_args()
 if a.from_release_tables:
  architecture();construction();training();extras();print('Regenerated four non-video figure families from included tables.')
 else:
  if a.data_root is None:p.error('--data-root is required unless --from-release-tables is used')
  storyboard(a.data_root);architecture();construction(a.data_root);training(a.data_root);extras();print('Generated five figure families (PNG/PDF/SVG).')
if __name__=='__main__':main()
