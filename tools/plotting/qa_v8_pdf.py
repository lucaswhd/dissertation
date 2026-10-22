#!/usr/bin/python3
"""Scan every rendered page and export contact sheets for human/agent visual QA."""
import argparse,json,re,math
from pathlib import Path
import fitz
from PIL import Image,ImageDraw

def inspect(pdf,out,event_number='5.3'):
 out.mkdir(parents=True,exist_ok=True);d=fitz.open(pdf);pages=[];figures={};flags=[];thumbs=[];in_body=False;table_numbers=[]
 for idx,p in enumerate(d):
  tx=p.get_text()
  if re.match(r'^Chapter 1\s+Introduction',tx):in_body=True
  words=p.get_text('words');n=len(words);spans=[s for b in p.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
  image_infos=p.get_image_info(xrefs=True);ims=[i for i in image_infos if (i['bbox'][2]-i['bbox'][0])*(i['bbox'][3]-i['bbox'][1])>1000]
  bad=[s for s in spans if s['text'].strip() and (s['bbox'][0]<-1 or s['bbox'][1]<-1 or s['bbox'][2]>p.rect.width+1 or s['bbox'][3]>p.rect.height+1)]
  edge=[s for s in spans if s['text'].strip() and (s['bbox'][0]<30 or s['bbox'][2]>p.rect.width-30)]
  img_bad=[i for i in ims if i['bbox'][0]<-1 or i['bbox'][1]<-1 or i['bbox'][2]>p.rect.width+1 or i['bbox'][3]>p.rect.height+1]
  blank=n<4 and not ims
  if bad:flags.append({'page':idx+1,'kind':'text_outside_page','count':len(bad)})
  if edge:flags.append({'page':idx+1,'kind':'text_near_page_edge','count':len(edge)})
  if img_bad:flags.append({'page':idx+1,'kind':'image_outside_page','count':len(img_bad)})
  if blank:flags.append({'page':idx+1,'kind':'blank_page'})
  if '\ufffd' in tx:flags.append({'page':idx+1,'kind':'replacement_character'})
  if in_body:table_numbers.extend(re.findall(r'^Table\s+([345A]\.\d+)\s*:',tx,re.M))
  for no in (re.findall(r'^Figure\s+([35A]\.\d+)\s*:',tx,re.M) if in_body else []):
   rects=p.search_for('Figure '+no+':');figures[no]={'page':idx+1,'caption_boxes':[list(x) for x in rects],'large_image_boxes':[list(i['bbox']) for i in ims]}
   if not ims:flags.append({'page':idx+1,'kind':'caption_without_image_on_page','figure':no})
  rec={'page':idx+1,'words':n,'images':len(ims),'size_pt':[p.rect.width,p.rect.height],'text_outside_page':len(bad),'near_edge':len(edge),'blank':blank}
  pages.append(rec)
  pix=p.get_pixmap(matrix=fitz.Matrix(700/p.rect.width,700/p.rect.width),alpha=False);path=out/f'page_{idx+1:03d}.png';pix.save(path);thumbs.append(path)
 # Contact sheets cover every PDF page; individual full-page renders remain available.
 sheets=[];cellw=300;cellh=447
 for start in range(0,len(thumbs),12):
  batch=thumbs[start:start+12];sheet=Image.new('RGB',(4*cellw,3*cellh),'#d7d7d7');draw=ImageDraw.Draw(sheet)
  for j,path in enumerate(batch):
   im=Image.open(path);im.thumbnail((290,417));x=(j%4)*cellw;y=(j//4)*cellh;sheet.paste(im,(x+(cellw-im.width)//2,y+24));draw.text((x+10,y+6),f'PDF page {start+j+1}',fill='black')
  path=out/f'contact_{start+1:03d}_{start+len(batch):03d}.jpg';sheet.save(path,quality=92);sheets.append(str(path))
 expected_figures=[f'3.{i}' for i in range(1,7)]+[f'5.{i}' for i in range(1,5)]+['A.1']
 expected_tables=['3.1','3.2','4.1','4.2']+[f'5.{i}' for i in range(1,6)]+[f'A.{i}' for i in range(1,10)]
 if list(figures)!=expected_figures:flags.append({'kind':'wrong_figure_number_sequence','actual':list(figures),'expected':expected_figures})
 if table_numbers!=expected_tables:flags.append({'kind':'wrong_table_number_sequence','actual':table_numbers,'expected':expected_tables})
 result={'pdf':pdf.name,'table_numbers':table_numbers,'pages':len(d),'all_pages_scanned':len(pages),'figures':figures,'flags':flags,'page_records':pages,'contact_sheets':sheets,'limitations':'Geometry/text checks do not prove absence of all visual overlap; every contact-sheet page and each figure page must also be visually inspected. LibreOffice render is not a Microsoft Word rendering guarantee.'}
 (out/'pdf_qa.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['page_records','contact_sheets']},indent=2));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--pdf',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--event-number',default='5.3');a=p.parse_args();inspect(a.pdf,a.out,a.event_number)
