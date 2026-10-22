#!/usr/bin/env python3
"""Re-layout the same six verified v7 video frames; local-only image output."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--frames',type=Path,required=True);a=p.parse_args()
 rows=json.loads((ROOT/'tables/storyboard_frames.json').read_text());assert len(rows)==6
 labels=['Wait for instruction','Acquire trash can','Pick up soda can','Deposit soda can','Follow human','Radio interaction']
 fig,axs=plt.subplots(2,3,figsize=(5.7,4.25));fig.subplots_adjust(left=.015,right=.99,top=.94,bottom=.12,wspace=.07,hspace=.51)
 for ax,r,label in zip(axs.flat,rows,labels):
  im=Image.open(a.frames/r['frame_image']);ax.imshow(im);ax.axis('off');ax.set_title(r['panel']+'  '+label,loc='left',fontsize=8,fontweight='bold',pad=5)
  ax.text(0,-.055,f"S{r['source']} · R{r['round']} · f{r['raw_global_frame']} · {r['seconds']:.2f} s\n{r['control']}",transform=ax.transAxes,va='top',fontsize=7.1)
 d=ROOT/'figures/methodology';d.mkdir(exist_ok=True)
 for ext in ['png','pdf','svg']:fig.savefig(d/f'v8_fig_3_3_task_storyboard.{ext}',dpi=400)
 plt.close(fig);print('Reused six unchanged recorded frames; local-only storyboard labels laid out at 5.7 inches.')
if __name__=='__main__':main()
