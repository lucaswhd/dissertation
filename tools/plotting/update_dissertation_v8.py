#!/usr/bin/env python3
"""Apply bounded v8 changes to the actual v7 manuscript, preserving its source."""
import argparse,copy,hashlib,json,re
from pathlib import Path
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches,Pt
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.text.paragraph import Paragraph
from update_dissertation import tc,field,para_after
ROOT=Path(__file__).resolve().parents[2]

def after(p,tx='',style=None):return para_after(p,tx,style)
def set_text(p,tx):p.text=tx

def delete_tc_fields(doc):
 # Remove TC fields only; preserve all visible research prose and other fields.
 for p in doc.paragraphs:
  nodes=list(p._p);remove=[];active=[];is_tc=False
  for node in nodes:
   if node.tag!=qn('w:r'):continue
   ch=node.find(qn('w:fldChar'));ins=node.find(qn('w:instrText'))
   if ch is not None and ch.get(qn('w:fldCharType'))=='begin':active=[node];is_tc=False;continue
   if active:
    active.append(node)
    if ins is not None and (ins.text or '').lstrip().startswith('TC '):is_tc=True
    if ch is not None and ch.get(qn('w:fldCharType'))=='end':
     if is_tc:remove.extend(active)
     active=[];is_tc=False
  for node in remove:p._p.remove(node)

def build(inp,out,event_no):
 assert inp.resolve()!=out.resolve();original_hash=hashlib.sha256(inp.read_bytes()).hexdigest();doc=Document(inp);assert len(doc.tables)==18
 original=list(doc.paragraphs);all_before='\n'.join(p.text for p in original)
 body_width=min((s.page_width-s.left_margin-s.right_margin)/914400 for s in doc.sections)
 if 'FigureBlock' not in doc.styles:
  st=doc.styles.add_style('FigureBlock',WD_STYLE_TYPE.PARAGRAPH);st.base_style=doc.styles['Normal']
 st=doc.styles['FigureBlock'];st.paragraph_format.line_spacing=1.0;st.paragraph_format.space_before=Pt(7);st.paragraph_format.space_after=Pt(4);st.paragraph_format.keep_with_next=True;st.paragraph_format.keep_together=True
 def find(prefix):
  exact=[p for p in doc.paragraphs if p.text==prefix]
  matches=exact or [p for p in doc.paragraphs if p.text.startswith(prefix)];assert len(matches)==1,(prefix,len(matches));return matches[0]
 def image_before(cap):
  n=cap._p.getprevious()
  while n is not None:
   if n.tag==qn('w:p') and n.xpath('.//w:drawing'):return Paragraph(n,cap._parent)
   n=n.getprevious()
  raise ValueError(cap.text)
 def set_image(p,path,maxheight=6.4):
  from PIL import Image
  im=Image.open(path);ww=min(body_width,maxheight*im.width/im.height)
  p.clear();p.style='FigureBlock';p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.line_spacing=1.0;p.paragraph_format.keep_with_next=True;p.paragraph_format.keep_together=True
  p.add_run().add_picture(str(path),width=Inches(ww))
 def figure_after(anchor,no,path,lead,caption):
  le=after(anchor,lead);le.paragraph_format.keep_with_next=False
  im=after(le);set_image(im,path)
  ca=after(im,f'Figure {no}: {caption}','PDFCaption');ca.paragraph_format.line_spacing=1.05;ca.paragraph_format.keep_together=True;ca.paragraph_format.keep_with_next=False
  return ca
 changes=[]
 replacements={
 '3.1':('methodology/v8_fig_3_1_diagnostic_hierarchy.png','Hierarchical diagnostic evaluation framework. Levels 1–4 form the controlled final audit of supervision, behavioural baselines, stochastic inference and observation conditioning. Level 5 summarises historical closed-loop diagnostics under a separate supporting protocol; it is not a matched A20/B20/C20 rollout comparison.'),
 '3.2':('methodology/v8_fig_3_2_task_supervision.png','Task structure and supervision annotation. (a) Fixed task order: cleanup in round 0, followed by radio interaction in round 1, with three soda-can pickup–deposit cycles. Box widths do not encode duration. (b) An illustrative transition at f marks decision frames within ±16 frames. Anchor t predicts 32 future actions from t+1 to t+32 under shift-1 alignment. A valid anchor preserves source and coarse context over [t,t+32]; B/C targets may cross a local-prompt boundary within that unchanged context. Positions are schematic, not measured timestamps.'),
 '3.4':('methodology/v8_fig_3_4_controller_data_flow.png','Implemented controller scope and data flow. (a) The scripted scheduler returns through WaitNextCommand and Relocate to WaitNearHuman for a non-final round; the final FaceHuman phase exits to AllTasksDone. (b) Dataset construction uses upstream BEHAVIOR state–action replay inside ExecuteBehaviorTask; policy rollout substitutes π0.5 action prediction in that execution scope. Scheduling and surrounding navigation remain scripted. (c) Source-level QC and splitting precede supervision export and diagnostic evaluation, with source identity retained.')}
 for no,(asset,caption) in replacements.items():
  cap=find(f'Figure {no}:');set_image(image_before(cap),ROOT/'figures'/asset);set_text(cap,f'Figure {no}: {caption}');changes.append(f'Redraw Figure {no}')
 cap=find('Figure 3.3:');set_image(image_before(cap),ROOT/'figures/methodology/v8_fig_3_3_task_storyboard.png')
 # All drawing paragraphs explicitly override the exact-height Normal style.
 for p in doc.paragraphs:
  if p._p.xpath('.//w:drawing'):
   p.paragraph_format.line_spacing=1.0;p.paragraph_format.keep_with_next=True;p.paragraph_format.keep_together=True
   p.paragraph_format.space_before=Pt(7);p.paragraph_format.space_after=Pt(4)
 # Preserve the original six actual video frames; no AI-created replacement.
 # Metadata supplied by the evidence-figure script determines exact statistics/captions.
 evidence=json.loads((ROOT/'tables/v8_evidence_figure_metadata.json').read_text())
 for no,key,anchor in [('3.6','3.6','The distinction is central to RQ1.'),(event_no,'5.4','of 0.0810 across the five flow seeds.')]:
  d=evidence[key];figure_after(find(anchor),no,ROOT/d['png'],d['lead'].replace('Figure 5.4',f'Figure {event_no}'),re.sub(r'^Figure [^:]+:\s*','',d['caption']));changes.append(f'Insert Figure {no}')
 # Repair the original paragraph's mismatched all-source versus validation reference.
 for p in doc.paragraphs:
  if 'Figure 5.1 summarises these geometric coverage ratios by task segment.' in p.text:
   set_text(p,p.text.replace('Figure 5.1 summarises these geometric coverage ratios by task segment.','Figure 5.1 gives the segment-level geometric coverage for validation pickup anchors.'))
 cap=find('Figure 5.1:');set_text(cap,cap.text.replace('The early walk/acquire','The walk/acquire'))
 set_image(image_before(cap),ROOT/'figures/results/v8_fig_5_1_frustum.png')
 changes.append('Redraw Figure 5.1 from archived counts to repair source-axis clipping')
 conditioning_no='5.4' if event_no=='5.3' else '5.3'
 if event_no=='5.3':
  for p in doc.paragraphs:
   if p.text.startswith('Figure 5.3: Validation observation-conditioning') or 'Figure 5.3 visualises the perturbation/noise ratios' in p.text:
    set_text(p,p.text.replace('Figure 5.3','Figure 5.4'))
 # Keep all equation/references content intact; add narrow links to RQ1 and RQ2.
 p=find('The distinction is central to RQ1.');p.add_run(' Figure 3.6 relates this distinction to the recorded temporal and action structure.')
 # Update the factual AI disclosure for v8 graphics while leaving declaration wording/signatures intact.
 for p in doc.paragraphs:
  if p.text.startswith('Figures 3.1, 3.2 and 3.4 are schematic diagrams'):
   set_text(p,'Figures 3.1, 3.2 and 3.4 are schematic diagrams drawn using AI-assisted code. Figure 3.3 contains extracted simulation-video frames; Figures 3.5 and 3.6 summarise recorded dataset characteristics. The statistical')
 # Add front/body sections so page numbers can use Roman/Arabic independently.
 # The input has no footer/page-number fields; chapter text and heading order remain unchanged.
 s0=copy.deepcopy(doc.sections[0]._sectPr)
 for child in list(s0):
  if child.tag in (qn('w:headerReference'),qn('w:footerReference')):s0.remove(child)
 def section_pr(fmt=None,start=None):
  ss=copy.deepcopy(s0);typ=ss.find(qn('w:type'))
  if typ is None:typ=OxmlElement('w:type');ss.insert(0,typ)
  typ.set(qn('w:val'),'nextPage')
  if fmt:
   pn=ss.find(qn('w:pgNumType'))
   if pn is None:pn=OxmlElement('w:pgNumType');ss.append(pn)
   pn.set(qn('w:fmt'),fmt);pn.set(qn('w:start'),str(start))
  return ss
 # Title-page section ends at the final 2026 immediately before the first declaration.
 declaration=find('Statement of Originality and AI Use Disclosure');prev=declaration._p.getprevious();prev.find(qn('w:pPr'))
 pr=prev.find(qn('w:pPr'))
 if pr is None:pr=OxmlElement('w:pPr');prev.insert(0,pr)
 pr.append(section_pr())
 chapter1=find('Chapter 1');prev=chapter1._p.getprevious();pr=prev.find(qn('w:pPr'))
 if pr is None:pr=OxmlElement('w:pPr');prev.insert(0,pr)
 pr.append(section_pr('lowerRoman',1))
 final=doc.sections[-1]._sectPr;pn=final.find(qn('w:pgNumType'))
 if pn is None:pn=OxmlElement('w:pgNumType');final.append(pn)
 pn.set(qn('w:fmt'),'decimal');pn.set(qn('w:start'),'1')
 for i,section in enumerate(doc.sections):
  section.footer.is_linked_to_previous=False
  fp=section.footer.paragraphs[0];fp.clear();fp.alignment=WD_ALIGN_PARAGRAPH.CENTER;fp.paragraph_format.line_spacing=1.0
  if i>0:field(fp,'PAGE','1' if i==2 else 'i')
 # Restore the inherited run-on acronym text as separate editable tabbed paragraphs.
 acronym_heading=find('Acronyms');hp=acronym_heading._p.getnext();header=Paragraph(hp,acronym_heading._parent);header.text='Acronym\tMeaning'
 dp=hp.getnext();definition=Paragraph(dp,acronym_heading._parent)
 terms=[('ACT','Action Chunking with Transformers'),('FOV','Field of View'),('FSM','Finite-State Machine'),('HRC','Human–Robot Collaboration'),('LfD','Learning from Demonstration'),('LoRA','Low-Rank Adaptation'),('MAE','Mean Absolute Error'),('QA','Quality Assurance'),('RGB','Red–Green–Blue image'),('VLA','Vision-Language-Action model'),('VLM','Vision-Language Model')]
 for j,(term,meaning) in enumerate(terms):
  pp=definition if j==0 else after(pp)
  pp.text=term+'\t'+meaning;pp.paragraph_format.tab_stops.add_tab_stop(Inches(.65));pp.paragraph_format.line_spacing=1.2;pp.paragraph_format.space_after=Pt(4)
 # Logical chapter/front-matter page starts, with linked heading lines and widow control.
 for p in doc.paragraphs:
  p.paragraph_format.widow_control=True
  if p.style.name in ['ChapterTitle','SectionTitle','SubsectionTitle']:p.paragraph_format.keep_with_next=True
  if p.text=='References' or p.text=='List of Figures':p.paragraph_format.page_break_before=True
 declaration.paragraph_format.page_break_before=False;chapter1.paragraph_format.page_break_before=False
 # Replace private TC indexes with standard outline and SEQ-caption fields.
 delete_tc_fields(doc)
 for name,base,level in [('V8ChapterHeading','ChapterTitle',0),('V8FrontHeading','ChapterTitle',0),('V8SectionHeading','SectionTitle',1),('V8SubsectionHeading','SubsectionTitle',2)]:
  if name not in doc.styles:
   style=doc.styles.add_style(name,WD_STYLE_TYPE.PARAGRAPH);style.base_style=doc.styles[base]
  pp=doc.styles[name].element.get_or_add_pPr();lv=pp.find(qn('w:outlineLvl'))
  if lv is None:lv=OxmlElement('w:outlineLvl');pp.append(lv)
  lv.set(qn('w:val'),str(level))
 # Merge each chapter's two heading lines into one paragraph; the visible two-line title is preserved.
 for p in list(doc.paragraphs):
  if re.match(r'^Chapter (?:[1-7]|A)$',p.text):
   nxt=p._p.getnext();following=Paragraph(nxt,p._parent);p.add_run().add_break();p.add_run(following.text);nxt.getparent().remove(nxt);p.style='V8ChapterHeading'
  elif p.style.name in ['SectionTitle','SubsectionTitle'] and re.match(r'^(?:[1-7]|A)\.',p.text):p.style='V8SectionHeading' if p.style.name=='SectionTitle' else 'V8SubsectionHeading'
  elif p.text in ['Abstract','Acknowledgements','Acronyms','References']:p.style='V8FrontHeading'
 figcaps=[];tablecaps=[];last={'Figure':None,'Table':None}
 for p in doc.paragraphs:
  match=re.match(r'^(Figure|Table) ([345A])\.(\d+):\s*(.*)',p.text,re.S)
  if match:
   label,ch,num,caption=match.groups();parts=caption.split('. ',1);short=parts[0]+('.' if len(parts)>1 else '');detail=parts[1] if len(parts)>1 else '';p.clear();p.add_run(f'{label} {ch}.')
   code=f'SEQ {label} \\r {num}'
   field(p,code,num);p.add_run(': '+short);last[label]=ch
   if detail:
    note=after(p,detail,'PDFCaption');note.paragraph_format.line_spacing=1.05;note.paragraph_format.keep_together=True;note.paragraph_format.keep_with_next=(label=='Table');note.paragraph_format.space_before=Pt(0);note.paragraph_format.space_after=Pt(6)
   if label=='Figure':figcaps.append((ch+'.'+num,caption.split('. ')[0]));p.paragraph_format.keep_with_next=bool(detail);p.paragraph_format.keep_together=True
   else:tablecaps.append((ch+'.'+num,caption));p.paragraph_format.keep_with_next=True
   p.paragraph_format.line_spacing=1.05
 for heading,code,entries in [('Table of Contents','TOC \\o "1-3" \\h \\z \\u',[p.text.replace('\n',' ') for p in doc.paragraphs if p.style.name=='V8ChapterHeading']),('List of Figures','TOC \\c "Figure" \\h \\z',[n+' '+t for n,t in figcaps]),('List of Tables','TOC \\c "Table" \\h \\z',[n+' '+t for n,t in tablecaps])]:
  h=find(heading);n=h._p.getnext()
  while n is not None and n.tag!=qn('w:p'):n=n.getnext()
  assert n is not None
  pp=Paragraph(n,h._parent);pp.clear();field(pp,code,'\n'.join(entries));pp.paragraph_format.line_spacing=1.05;pp.paragraph_format.keep_with_next=False
 for t in doc.tables:
  margins=OxmlElement('w:tblCellMar')
  for side in ('left','right'):
   mm=OxmlElement('w:'+side);mm.set(qn('w:w'),'60');mm.set(qn('w:type'),'dxa');margins.append(mm)
  t._tbl.tblPr.append(margins)
  for ri,row in enumerate(t.rows):
   for h in row._tr.xpath('./w:trPr/w:trHeight[@w:hRule="exact"]'):h.getparent().remove(h)
   if len(t.rows)<=9:
    for cell in row.cells:
     for pp in cell.paragraphs:pp.paragraph_format.keep_with_next=ri<len(t.rows)-1
 # Native OMML repair is a layout correction only, using original equation definitions.
 from v8_math_repair import repair_math
 math_changes=repair_math(doc)
 assert not any(c.get('status')=='Pending' for c in math_changes),math_changes
 changes.extend(math_changes)
 # These tables are carried over byte-for-byte in values; only paragraph geometry changes.
 assert len(doc.tables)==18
 out.parent.mkdir(parents=True,exist_ok=True);doc.save(out)
 assert original_hash==hashlib.sha256(inp.read_bytes()).hexdigest()
 report={'input_v7_sha256':original_hash,'output_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'event_figure_number':event_no,'conditioning_figure_number':conditioning_no,'native_tables':18,'numbered_figures':len(figcaps),'changes':changes,'fixed_cause':'New v7 pictures inherited Normal exact line height; all v8 image paragraphs use explicit automatic single spacing.','v7_preserved':True}
 (ROOT/'reports/V8_BUILD_METADATA.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--event-number',choices=['5.3','5.4'],default='5.3');a=p.parse_args();build(a.input,a.output,a.event_number)
