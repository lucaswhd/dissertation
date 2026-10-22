#!/usr/bin/env python3
"""Redraw Figure 5.1 from archived validation pickup counts; no inference."""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]

def main():
    data=json.loads((ROOT/'tables/native_tables.json').read_text())['5.1']
    archived=list(csv.DictReader((ROOT/'evidence/visibility/summary.csv').open()))
    segments=['Walk to the trash can and pick it up','Pick up the soda can',
              'Drop the can into the trash can you are holding','Put the trash can down']
    rows=[]
    for native,segment in zip(data['rows'],segments):
        matches=[r for r in archived if r['split']=='val' and r['context']=='pickup'
                 and r['population']=='valid_anchors' and r['segment']==segment]
        assert len(matches)==3
        total=sum(int(r['total']) for r in matches)
        inside=sum(int(r['target_center_in_frustum']) for r in matches)
        assert [int(v.replace(',','')) for v in native[1:3]]==[total,inside]
        assert f'{inside/total:.4f}'==native[3]
        rows.append({'segment':native[0],'valid_anchors':total,'in_frustum':inside,'ratio':inside/total})
    with (ROOT/'tables/v8_frustum_coverage.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'text.color':'#23313a',
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(5.7,2.85))
    fig.subplots_adjust(left=.16,right=.975,bottom=.23,top=.93)
    bars=ax.bar(range(4),[r['ratio'] for r in rows],width=.56,
                color=['#2c5368','#bc7044','#718470','#7d929d'])
    ax.set(ylim=(0,1.13),ylabel='Target centre in frustum (fraction)')
    ax.set_xticks(range(4),['Walk / acquire\nbin','Pick up\nsoda can','Deposit\ncan','Put bin\ndown'])
    ax.set_yticks([0,.25,.5,.75,1.0]);ax.yaxis.grid(True,color='#dce3e6',lw=.5)
    ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    for bar,row in zip(bars,rows):
        ax.text(bar.get_x()+bar.get_width()/2,row['ratio']+.018,
                f"{100*row['ratio']:.1f}%\nn = {row['valid_anchors']:,}",ha='center',va='bottom',fontsize=7.5)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=fig.bbox
    for artist in [ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels(),*ax.texts]:
        box=artist.get_window_extent(renderer)
        assert bounds.contains(box.x0,box.y0) and bounds.contains(box.x1,box.y1),artist.get_text()
    dest=ROOT/'figures/results';dest.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf','svg'):fig.savefig(dest/f'v8_fig_5_1_frustum.{ext}',dpi=400)
    plt.close(fig)
    (ROOT/'reports/V8_FRUSTUM_FIGURE.md').write_text('''# V8 Figure 5.1 repair

Pass: redrawn from evidence/visibility/summary.csv, filtered to val / pickup / valid_anchors. Early/middle/late counts are summed before division and checked exactly against native Table5.1. Four validation sources; this is an anchor-weighted descriptive geometric metric, not an independent-source estimate. No error bars or new inference.

Variables: n = valid anchors in segment; numerator = anchors with target centre in the camera frustum; ratio = numerator/n. Counts:2916/3569,6054/6792,1560/1560,381/381. The metric does not model occlusion or rendered-pixel visibility. Segment labels are shortened only in the axis.

Caption: Validation pickup target-centre geometric frustum coverage by task segment. Counts are aggregated over early/middle/late positions before division; n denotes valid anchors. Geometric coverage does not establish rendered visibility or policy success.

The legacy raster clipped the left edge of its y-axis label. This fixed-width5.7in redraw checks every label against canvas bounds and preserves the counts. PNG400dpi, editable text SVG and vector PDF are saved under figures/results/v8_fig_5_1_frustum.*; derived CSV:tables/v8_frustum_coverage.csv. Reproduce: python tools/plotting/generate_v8_frustum.py.
''')
    print('Figure5.1: archived counts match Table5.1; all labels within canvas.')

if __name__=='__main__':main()
