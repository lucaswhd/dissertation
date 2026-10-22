#!/usr/bin/python3
"""Render DOCX using one isolated LibreOffice instance and update its indexes.
Requires system python3-uno; input is never overwritten. PDF is a review render.
"""
import argparse,json,subprocess,time,uuid
from pathlib import Path
import uno
from com.sun.star.beans import PropertyValue

def prop(name,value):
 p=PropertyValue();p.Name=name;p.Value=value;return p

def render(inp,out,persist_docx=None):
 profile=out.parent/'.lo_v8_profile';profile.mkdir(parents=True,exist_ok=True)
 pipe='v8_'+uuid.uuid4().hex
 proc=subprocess.Popen(['libreoffice',f'-env:UserInstallation={profile.as_uri()}','--headless','--invisible','--nodefault','--norestore','--nofirststartwizard',f'--accept=pipe,name={pipe};urp;StarOffice.ComponentContext'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 doc=None;desktop=None
 try:
  ctx=uno.getComponentContext();resolver=ctx.ServiceManager.createInstanceWithContext('com.sun.star.bridge.UnoUrlResolver',ctx)
  for _ in range(120):
   try:remote=resolver.resolve(f'uno:pipe,name={pipe};urp;StarOffice.ComponentContext');break
   except Exception:time.sleep(.25)
  else:raise RuntimeError('LibreOffice connection timed out')
  desktop=remote.ServiceManager.createInstanceWithContext('com.sun.star.frame.Desktop',remote)
  doc=desktop.loadComponentFromURL(inp.resolve().as_uri(),'_blank',0,(prop('Hidden',True),prop('ReadOnly',True),prop('UpdateDocMode',3),prop('MacroExecutionMode',4)))
  if doc is None:raise RuntimeError('LibreOffice failed to load DOCX')
  doc.refresh();doc.TextFields.refresh();indexes=doc.getDocumentIndexes();counts=[]
  # Repeating handles page-number shifts caused by the newly expanded indexes.
  for _ in range(2):
   for i in range(indexes.Count):indexes.getByIndex(i).update()
   doc.TextFields.refresh();doc.refresh()
  for i in range(indexes.Count):
   ix=indexes.getByIndex(i);counts.append({'name':ix.Name,'type':ix.ImplementationName,'text_length':len(ix.Anchor.String)})
  out.parent.mkdir(parents=True,exist_ok=True)
  doc.storeToURL(out.resolve().as_uri(),(prop('FilterName','writer_pdf_Export'),prop('Overwrite',True),prop('FilterData',(prop('SelectPdfVersion',2),prop('ExportBookmarks',True),prop('UseTaggedPDF',True)))))
  if persist_docx:
   doc.storeToURL(persist_docx.resolve().as_uri(),(prop('FilterName','Office Open XML Text'),prop('Overwrite',True)))
  return {'renderer':'LibreOffice','input':inp.name,'pdf':out.name,'indexes':counts,'page_count':__import__('fitz').open(out).page_count,'pdf_bytes':out.stat().st_size,'saved_docx':persist_docx.name if persist_docx else None}
 finally:
  if doc:doc.close(True)
  if desktop:desktop.terminate()
  try:proc.wait(timeout=10)
  except subprocess.TimeoutExpired:proc.terminate()

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--pdf',type=Path,required=True);p.add_argument('--updated-docx',type=Path);p.add_argument('--report',type=Path);a=p.parse_args();d=render(a.input,a.pdf,a.updated_docx)
 if a.report:a.report.write_text(json.dumps(d,indent=2))
 print(json.dumps(d,indent=2))
if __name__=='__main__':main()
