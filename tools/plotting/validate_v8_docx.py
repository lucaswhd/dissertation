#!/usr/bin/env python3
"""Validate actual DOCX drawing layout inputs, numbering and package integrity."""
import argparse,hashlib,json,re,posixpath,zipfile
from pathlib import Path
from xml.etree import ElementTree as E
N={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing','a':'http://schemas.openxmlformats.org/drawingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships','pr':'http://schemas.openxmlformats.org/package/2006/relationships'}
def val(e,k):return e.get('{'+N['w']+'}'+k) if e is not None else None
def validate(path):
 with zipfile.ZipFile(path) as z:
  bad=z.testzip();assert bad is None,bad
  for name in z.namelist():
   if name.endswith(('.xml','.rels')):E.fromstring(z.read(name))
  doc=E.fromstring(z.read('word/document.xml'));styles=E.fromstring(z.read('word/styles.xml'));rels=E.fromstring(z.read('word/_rels/document.xml.rels'))
  sm={val(s,'styleId'):s for s in styles.findall('w:style',N)};targets={r.get('Id'):r.get('Target') for r in rels}
  issues=[];images=[];captions=[];text=[]
  def effective_spacing(p):
   e=p.find('w:pPr/w:spacing',N)
   if e is not None and val(e,'lineRule') is not None:return val(e,'lineRule'),val(e,'line')
   sid=val(p.find('w:pPr/w:pStyle',N),'val') or 'Normal';seen=set()
   while sid in sm and sid not in seen:
    seen.add(sid);e=sm[sid].find('w:pPr/w:spacing',N)
    if e is not None and val(e,'lineRule') is not None:return val(e,'lineRule'),val(e,'line')
    sid=val(sm[sid].find('w:basedOn',N),'val')
   return 'auto',None
  for i,p in enumerate(doc.findall('w:body/w:p',N)):
   tx=''.join(t.text or '' for t in p.findall('.//w:t',N));text.append(tx)
   sid=val(p.find('w:pPr/w:pStyle',N),'val') or ''
   if re.match(r'^(?:Figure|Table) (?:[3-5]|A)\.\d+:',tx) and not any(k in sid.lower() for k in ['index','toc']):captions.append(tx)
   for d in p.findall('.//wp:inline',N)+p.findall('.//wp:anchor',N):
    extent=d.find('wp:extent',N);cx=int(extent.get('cx'));cy=int(extent.get('cy'));blip=d.find('.//a:blip',N);rid=blip.get('{'+N['r']+'}embed')
    target=targets.get(rid);name=posixpath.normpath('word/'+target) if target else ''
    entry={'paragraph':i,'image':name,'cx':cx,'cy':cy,'line_spacing':effective_spacing(p),'inline':d.tag.endswith('inline'),'caption_next':text[-1] if False else ''}
    if entry['line_spacing'][0]=='exact':issues.append({'kind':'exact_line_spacing_on_picture','paragraph':i,'image':name})
    if name not in z.namelist():issues.append({'kind':'missing_image_relationship','paragraph':i,'target':name})
    for cr in d.findall('.//a:srcRect',N):
     if any(int(v)!=0 for v in cr.attrib.values()):issues.append({'kind':'drawing_crop','paragraph':i})
    images.append(entry)
  fixed_rows=[E.tostring(e,encoding='unicode') for e in doc.findall('.//w:trPr/w:trHeight',N) if val(e,'hRule')=='exact']
  if fixed_rows:issues.append({'kind':'fixed_exact_table_rows','count':len(fixed_rows)})
  figcaps=[c for c in captions if c.startswith('Figure ')];tabcaps=[c for c in captions if c.startswith('Table ')]
  nums=[re.match(r'Figure ([^:]+):',x).group(1) for x in figcaps]
  if len(nums)!=len(set(nums)):issues.append({'kind':'duplicate_figure_caption'})
  sects=doc.findall('.//w:sectPr',N)
  return {'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'package_xml':'Pass','images':images,'figure_numbers':nums,'figure_captions':len(figcaps),'table_captions':len(tabcaps),'native_tables':len(doc.findall('.//w:tbl',N)),'section_count':len(sects),'issues':issues,'passed':not issues}
def main():
 p=argparse.ArgumentParser();p.add_argument('docx',type=Path);p.add_argument('--output',type=Path);p.add_argument('--expect-pass',action='store_true');a=p.parse_args();d=validate(a.docx)
 if a.output:a.output.write_text(json.dumps(d,indent=2))
 print(json.dumps({k:v for k,v in d.items() if k!='images'},indent=2))
 if a.expect_pass:assert d['passed'],d['issues']
if __name__=='__main__':main()
