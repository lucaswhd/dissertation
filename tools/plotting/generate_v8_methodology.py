#!/usr/bin/env python3
"""Draw v8 methods schematics from the established protocol; no raw data needed.

Only writes the three v8 figure families and V8_METHOD_FIGURES.md.
All diagrams are conceptual, with no invented measured durations or outcomes.
"""
from pathlib import Path
import argparse
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'figures' / 'methodology'
NAVY = '#2c5368'
ORANGE = '#bc7044'
GREEN = '#718470'
INK = '#23313a'
MUTED = '#536471'
BLUE_FILL = '#edf3f5'
ORANGE_FILL = '#f6ece3'
GREEN_FILL = '#edf2ea'
GREY = '#f5f6f7'
EDGE = '#a6b2b9'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9, 'text.color': INK,
    'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'savefig.facecolor': 'white', 'axes.facecolor': 'white',
    'lines.solid_capstyle': 'round',
})


def canvas(height):
    fig = plt.figure(figsize=(5.7, height))
    ax = fig.add_axes([.025, .025, .95, .95])
    ax.set(xlim=(-3, 103), ylim=(0, 102))
    ax.axis('off')
    fig._method_boxes = []
    return fig, ax


def txt(ax, x, y, label, size=9, weight='normal', ha='left', va='center', color=INK, **kw):
    return ax.text(x, y, label, fontsize=size, weight=weight, ha=ha, va=va,
                   color=color, linespacing=1.2, **kw)


def box(ax, x, y, w, h, label, fill=BLUE_FILL, edge=EDGE, size=8.5, weight='normal', dashed=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08,rounding_size=1.0',
                      facecolor=fill, edgecolor=edge, lw=.8,
                      linestyle=(0, (4, 3)) if dashed else '-')
    ax.add_patch(p)
    t = txt(ax, x+w/2, y+h/2, label, size=size, weight=weight, ha='center')
    ax.figure._method_boxes.append((p, t))
    return p


def arrow(ax, a, b, color=NAVY, lw=.9, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=8,
                                lw=lw, color=color, shrinkA=1.3, shrinkB=1.3, **kw))


def polyarrow(ax, points, color=NAVY, lw=.9):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw)
    arrow(ax, points[-2], points[-1], color=color, lw=lw)


def panel(ax, y, label):
    txt(ax, 0, y, label, size=10, weight='bold')


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    issues = []
    for p, t in fig._method_boxes:
        pb = p.get_window_extent(renderer)
        tb = t.get_window_extent(renderer)
        if tb.x0 < pb.x0+2 or tb.x1 > pb.x1-2 or tb.y0 < pb.y0+1 or tb.y1 > pb.y1-1:
            issues.append('Text outside box: '+t.get_text().replace('\n', ' / '))
    texts = [t for ax in fig.axes for t in ax.texts if t.get_text().strip()]
    for a, b in itertools.combinations(texts, 2):
        aa, bb = a.get_window_extent(renderer), b.get_window_extent(renderer)
        overlap_x = min(aa.x1,bb.x1)-max(aa.x0,bb.x0)
        overlap_y = min(aa.y1,bb.y1)-max(aa.y0,bb.y0)
        if overlap_x > 1 and overlap_y > 1:
            issues.append('Text overlap: '+a.get_text().replace('\n',' / ')+' <> '+b.get_text().replace('\n',' / '))
    if issues:
        raise RuntimeError(stem+'\n'+'\n'.join(issues))
    for ext in ('png', 'pdf', 'svg'):
        # Fixed physical width; avoid tight-bbox rescaling at manuscript insertion.
        fig.savefig(OUT / f'{stem}.{ext}', dpi=300)
    plt.close(fig)


def hierarchy():
    fig, ax = canvas(4.0)
    box(ax, 0, 88, 100, 12, 'Long-horizon HRC demonstrations', size=10, weight='bold', edge=NAVY)
    arrow(ax, (50,88), (50,83.5))
    group = FancyBboxPatch((0,29),100,54,boxstyle='round,pad=.1,rounding_size=1.2',
                           facecolor='white',edgecolor=NAVY,lw=.9)
    ax.add_patch(group)
    ax.add_patch(Rectangle((.2,73),99.6,9.5,facecolor=BLUE_FILL,edgecolor='none'))
    txt(ax, 3, 77.8, 'Controlled final audit', size=10, weight='bold',color=NAVY)
    txt(ax, 97,77.8,'Levels 1–4',size=8.5,ha='right',color=NAVY)
    levels=[('1','Supervision','Alignment · chunks · valid anchors'),
            ('2','Behavioural baselines','No-motion · lookup · persistence'),
            ('3','Stochastic inference','Matched noise · five inference seeds'),
            ('4','Observation conditioning','Controlled input perturbations')]
    for i,(num,title,detail) in enumerate(levels):
        y=67.2-i*10.5
        if i:
            ax.plot([3,97],[y+5.2,y+5.2],color='#d8e0e4',lw=.6)
        txt(ax, 5,y,num,size=10,weight='bold',color=NAVY)
        txt(ax, 11,y,title,size=9,weight='bold')
        txt(ax, 97,y,detail,size=8.1,ha='right',color=MUTED)
    arrow(ax,(50,29),(50,24))
    box(ax,0,2.5,100,21,'',fill=GREY,edge=MUTED,dashed=True)
    txt(ax,5,15,'5',size=10,weight='bold',color=MUTED)
    txt(ax,11,16.5,'Historical behavioural evidence',size=9.3,weight='bold')
    txt(ax,11,9,'Closed-loop diagnostics · separate supporting protocol',size=8.5,color=MUTED)
    save(fig,'v8_fig_3_1_diagnostic_hierarchy')


def task_supervision():
    fig, ax = canvas(5.45)
    panel(ax,99,'a  Fixed two-round task order')
    txt(ax,0,93,'Round 0 · Cleanup',size=9,weight='bold',color=NAVY)
    xs=[0,25,65,89];ws=[20,35,19,11]
    r0=['Walk / acquire\ntrash can','Pick up → deposit\nthree soda cans','Put down\ntrash can','Stand\nby']
    r1=['Walk / acquire\ntrash can','Turn on radio','Put down\ntrash can','Stand\nby']
    for y,labels in [(80,r0),(62,r1)]:
        for i,(x,w,label) in enumerate(zip(xs,ws,labels)):
            box(ax,x,y,w,10,label,fill=BLUE_FILL if i==1 else GREY,size=8.4,edge=NAVY if i==1 else EDGE)
            if i<3:arrow(ax,(x+w,y+5),(xs[i+1],y+5))
    txt(ax,0,75,'Round 1 · Radio interaction',size=9,weight='bold',color=NAVY)
    panel(ax,55,'b  Event annotation and future-action supervision')
    def tx(frame):return 22+(frame+24)/64*78
    # Separate event f=0 from decision anchor t=-8; frame offsets are schematic.
    f,t,boundary=0,-8,32
    bdy=tx(boundary);local=tx(8)
    txt(ax,0,46.3,'Coarse\ncontext',size=8.4)
    box(ax,tx(-24),43.2,bdy-tx(-24),6,'Same context over [t, t+32]',fill=GREEN_FILL,size=8.5,edge=GREEN)
    box(ax,bdy,43.2,tx(40)-bdy,6,'Next',fill=GREY,size=7.5)
    txt(ax,0,37.5,'B/C local\nprompt',size=8.4)
    box(ax,tx(-24),34.5,local-tx(-24),6,'Local prompt 1',fill=GREY,size=8.5)
    box(ax,local,34.5,bdy-local,6,'Local prompt 2',fill=GREY,size=8.5)
    ax.plot([local,local],[11.2,34.5],color=ORANGE,lw=.9,ls=(0,(3,3)),zorder=.4)
    txt(ax,0,28.8,'Expert\ngripper',size=8.4)
    ax.plot([tx(-24),tx(f),tx(f),bdy],[30.8,30.8,26.8,26.8],color=NAVY,lw=1.3)
    ax.scatter([tx(f)],[26.8],s=12,color=NAVY,zorder=4)
    txt(ax,tx(f)-1,25.1,'event f',size=8.2,ha='right',color=NAVY)
    txt(ax,0,20.1,'Decision\nwindow',size=8.4)
    box(ax,tx(f-16),18.3,tx(f+16)-tx(f-16),4.4,'±16 frames',fill=BLUE_FILL,size=8.4,edge=NAVY)
    txt(ax,tx(f-16),16.2,'f − 16',size=8,ha='center')
    txt(ax,tx(f+16),16.2,'f + 16',size=8,ha='center')
    txt(ax,0,9.2,'Shift-1\ntarget',size=8.4)
    box(ax,tx(t+1),7.0,tx(t+32)-tx(t+1),4.5,'32 future actions',fill=ORANGE_FILL,size=8.5,edge=ORANGE)
    ax.plot([tx(t),tx(t)],[6.6,13.2],color=INK,lw=1.0)
    txt(ax,tx(t)-1.7,9.2,'anchor t',size=8.3,ha='right',weight='bold')
    txt(ax,tx(t+1),4.5,'t + 1',size=8,ha='left')
    txt(ax,tx(t+32),4.5,'t + 32',size=8,ha='right')
    save(fig,'v8_fig_3_2_task_supervision')


def controller_flow():
    fig,ax=canvas(6.15)
    panel(ax,100,'a  Scripted HRC finite-state scheduler')
    xs=[1,26,51,76];w=22;h=7
    top=['WaitNearHuman','ReceiveCommand','GoToTask','ExecuteBehavior\nTask']
    for i,(x,label) in enumerate(zip(xs,top)):
        box(ax,x,89,w,h,label,fill=ORANGE_FILL if i==3 else BLUE_FILL,size=8.0)
        if i<3:arrow(ax,(x+w,92.5),(xs[i+1],92.5))
    lower=[('AllTasksDone',xs[0]),('FaceHuman',xs[1]),('ReturnToHuman',xs[2]),('TaskSegmentDone',xs[3])]
    for label,x in lower:
        box(ax,x,77,w,h,label,fill=GREEN_FILL if label=='AllTasksDone' else BLUE_FILL,size=8.0)
    arrow(ax,(87,89),(87,84))
    arrow(ax,(76,80.5),(73,80.5));arrow(ax,(51,80.5),(48,80.5))
    arrow(ax,(26,80.5),(23,80.5))
    txt(ax,12,86,'last round',size=7.8,ha='center',color=GREEN)
    arrow(ax,(37,77),(37,72))
    txt(ax,48,75,'more rounds',size=7.8,color=MUTED)
    box(ax,26,65,22,7,'WaitNextCommand',size=8.0)
    box(ax,51,65,22,7,'Relocate',size=8.0)
    arrow(ax,(48,68.5),(51,68.5))
    polyarrow(ax,[(62,65),(62,61),(-1.5,61),(-1.5,92.5),(1,92.5)])
    txt(ax,29,62.6,'next round',size=7.7,ha='center',color=NAVY)
    panel(ax,56.8,'b  Controller scope inside ExecuteBehaviorTask')
    box(ax,0,43.4,43,9,'Dataset construction\nBEHAVIOR state–action replay',fill=ORANGE_FILL,size=8.5,edge=ORANGE)
    box(ax,57,43.4,43,9,'Policy diagnostic\nπ0.5 predicted actions',fill=GREEN_FILL,size=8.5,edge=GREEN)
    box(ax,32.5,32.9,35,7.1,'ExecuteBehaviorTask',size=8.5,weight='bold',edge=NAVY)
    polyarrow(ax,[(21.5,43.4),(21.5,36.5),(32.5,36.5)],color=ORANGE)
    polyarrow(ax,[(78.5,43.4),(78.5,36.5),(67.5,36.5)],color=GREEN)
    txt(ax,50,29.1,'Scheduling and surrounding navigation remain scripted.',size=8.4,ha='center',color=MUTED)
    panel(ax,23.6,'c  From recorded sources to the benchmark')
    flow=['Recorded\nsources','Source-level\nQC and split','Supervision\nexport','Diagnostic\nevaluation']
    for i,label in enumerate(flow):
        x=i*26
        box(ax,x,10.5,22,9.2,label,size=8.4,fill=BLUE_FILL)
        if i<3:arrow(ax,(x+22,15.1),(x+26,15.1))
    txt(ax,50,5.2,'Source identity is retained through export and evaluation.',size=8.4,ha='center',color=MUTED)
    save(fig,'v8_fig_3_4_controller_data_flow')


def report():
    content='''# V8 methodology figures

三幅方法图均为概念示意，由已审计协议、v7文字和现有源码引用绘制，不重新审计源数据、不引入新的统计或运行结果。PNG按300dpi输出，PDF/SVG保留矢量；画布固定5.7inch宽，图中文字主要为8–10pt。旧图和旧脚本未覆盖。

## Figure 3.1

**Files:** `figures/methodology/v8_fig_3_1_diagnostic_hierarchy.{png,pdf,svg}`

**Lead:** Figure 3.1 separates the controlled final policy audit from supporting historical behavioural evidence.

**Caption:** Hierarchical diagnostic evaluation framework. Levels 1–4 form the controlled final audit of supervision, behavioural baselines, stochastic inference and observation conditioning. Level 5 summarises historical closed-loop diagnostics under a separate supporting protocol; it is not a matched A20/B20/C20 rollout comparison.

**Variables/source:** The level numbers denote evidential layers, not quantitative scores. Five inference seeds refer to inference flow noise, not independent training runs. Source: v7 Figure 3.1 and reports/THESIS_REVIEW.md (Figure 3.1 section); no numerical data are plotted.

## Figure 3.2

**Files:** `figures/methodology/v8_fig_3_2_task_supervision.{png,pdf,svg}`

**Lead:** Figure 3.2 distinguishes the fixed task order from the frame-level rules used to construct action supervision.

**Caption:** Task structure and supervision annotation. (a) Fixed task order within each retained source: cleanup in round 0, followed by radio interaction in round 1. Cleanup contains three soda-can pickup–deposit cycles. Box widths do not encode recorded duration. (b) An illustrative expert gripper transition at frame f marks decision frames within ±16 frames. At anchor t, shift-1 alignment predicts the 32 future actions from t+1 to t+32. A valid anchor preserves source, episode and coarse context over [t,t+32]; the B/C chunk may cross a local-prompt boundary while that coarse context remains unchanged. The dashed orange line marks such an allowed local boundary, whereas “Next” lies beyond the valid coarse-context interval.

**Variables/source:** f = illustrative gripper-transition frame; t = observation/state anchor frame; future targets = a_(t+1),…,a_(t+32); decision annotation = |t−f|≤16; “Next” = another coarse context. The relative positions of f, t and the local boundary are schematic, not measured source timestamps. Task order follows v7 Figure 3.2. Supervision rules and stable [t,t+32] boundaries: reports/EXPERIMENT_AUDIT.md (“样本与随机性”) and reports/THESIS_REVIEW.md (Figure 3.2 section and population notes). No raw demonstrations were loaded.

## Figure 3.4

**Files:** `figures/methodology/v8_fig_3_4_controller_data_flow.{png,pdf,svg}`

**Lead:** Figure 3.4 locates the policy diagnostic within the scripted scheduler and links the recorded sources to the benchmark.

**Caption:** Controller scope and data flow for the evaluated HRC protocol. (a) The scripted finite-state scheduler returns through WaitNextCommand and Relocate to WaitNearHuman for a non-final round; after the final FaceHuman phase it enters AllTasksDone. (b) Dataset construction uses upstream BEHAVIOR state–action replay inside ExecuteBehaviorTask; the policy diagnostic substitutes π0.5 action prediction only in that execution scope. Scheduling and surrounding navigation remain scripted. Replay is not autonomous learned manipulation. (c) Source-level quality control and splitting precede supervision export and diagnostic evaluation, with source identity retained throughout.

**Variables/source:** FSM labels are categorical scheduler states, not success predicates. ExecuteBehaviorTask is the control-substitution boundary, not a claim that every policy rollout succeeds. State names/order: source_snapshots/hrc_pipeline_traintest/scripts/collection/hrc_utils.py:52–63 and hrc_multitask_demo.py:5206–5222,5573–5625. Replay/policy interface evidence: same hrc_multitask_demo.py:812–868,5359–5398,5510–5552, plus reports/FIGURE_TRACEABILITY.md and reports/THESIS_REVIEW.md (Figure 3.4). The current branch includes broader optional rollout hooks; the drawing follows the dissertation's narrower evaluated policy scope, as required by the v8 brief. Source-level QC/export population is documented in reports/DATASET_AUDIT.md. No old core branch is used as final-implementation evidence.

## Reproduce and QA

Run from the repository root:

```bash
python3 tools/plotting/generate_v8_methodology.py
```

The script does not use network access, git, source-data reads, simulation or training. It writes only these three v8 figure families and this report. Before export it checks text bounding boxes for text–text overlap and text escaping node boxes. The initial v8 exports were additionally inspected with view_image; all three standalone figures passed visual inspection with no label overlap or cropping. Visual checks concern the standalone figures; final DOCX pagination is the main document workflow's responsibility.
'''
    content += '\n\n| Figure | PNG pixels | Physical size | Standalone visual QA | PDF / SVG |\n|---|---|---|---|---|\n| 3.1 | 1710 × 1200 | 5.7 × 4.00 in | Pass | Single page / editable text |\n| 3.2 | 1710 × 1635 | 5.7 × 5.45 in | Pass | Single page / editable text |\n| 3.4 | 1710 × 1845 | 5.7 × 6.15 in | Pass | Single page / editable text |\n\nAll PNGs carry 300 dpi metadata (rounded from 299.9994). PDF page widths are 410.4 pt = 5.7 in. Geometry checks passed for every node label and all text pairs. The figures contain no scene screenshots, third-party meshes, or raw data. The main agent should still inspect the inserted figures and captions in the final rendered document.\n'
    p=ROOT/'reports'/'V8_METHOD_FIGURES.md'
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(content,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    hierarchy();task_supervision();controller_flow();report()
    print('Generated three v8 methodology figures: PNG 300dpi, PDF, SVG; report written.')


if __name__=='__main__':
    main()
