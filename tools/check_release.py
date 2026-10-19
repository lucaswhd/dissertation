#!/usr/bin/env python3
"""Check the exact staged file set; never print possible secret values."""
from pathlib import Path
import ast,json,re,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1]
r=subprocess.run(['git','-C',str(ROOT),'ls-files','-z'],capture_output=True,check=True)
paths=[ROOT/p.decode() for p in r.stdout.split(b'\0') if p]
rules={
 'private_key':re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----'),
 'github_token':re.compile(r'\bgh[pousr]_[A-Za-z0-9]{25,}|\bgithub_pat_[A-Za-z0-9_]{40,}'),
 'hf_token':re.compile(r'\bhf_[A-Za-z0-9]{25,}'),
 'aws_access_id':re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
 'credential_literal':re.compile(r'''(?i)(?:api[_-]?key|access[_-]?token|secret[_-]?key|password)\s*[:=]\s*["']([A-Za-z0-9/+_=-]{20,})["']'''),
 'credential_url':re.compile(r'https?://[^\s/:]+:[^\s/@]+@'),
 'historical_private_path':re.compile(r'/(?:media/mldadmin|home/s124mdg39)[^\s"\']*')}
findings=[];syntax=[];files=[];total=0;maxsize=0;prohibited=[]
for p in paths:
 rel=str(p.relative_to(ROOT));s=p.lstat().st_size;total+=s;maxsize=max(maxsize,s)
 if p.is_symlink():findings.append({'file':rel,'rule':'symlink'})
 if s>=25*1024*1024:findings.append({'file':rel,'rule':'file_over_25MiB','bytes':s})
 if p.suffix.lower() in {'.hdf5','.h5','.npz','.parquet','.mp4','.pt','.pth','.ckpt','.safetensors','.docx'}:prohibited.append(rel)
 if 'fig_3_3_task_storyboard' in rel:prohibited.append(rel)
 blob=p.read_bytes()
 if rel not in ('reports/RELEASE_FILE_MANIFEST.json','reports/SECURITY_CHECKS.json'):
  files.append({'path':rel,'bytes':s,'sha256':hashlib.sha256(blob).hexdigest()})
 if b'\0' not in blob[:8192]:
  tx=blob.decode('utf-8',errors='replace')
  for name,pat in rules.items():
   for match in pat.finditer(tx):findings.append({'file':rel,'line':tx.count('\n',0,match.start())+1,'rule':name})
  if p.suffix=='.py':
   try:ast.parse(tx)
   except Exception as e:syntax.append({'file':rel,'error_type':type(e).__name__})
# Confirm actual exclusion behavior, not only presence of .gitignore text.
ignores=['.env','checkpoints/test/weights.pt','local.hdf5','local.mp4','manuscript.docx','figures/methodology/fig_3_3_task_storyboard.png']
checks={x:subprocess.run(['git','-C',str(ROOT),'check-ignore','-q',x]).returncode==0 for x in ignores}
result={'staged_files':len(paths),'staged_bytes':total,'max_file_bytes':maxsize,'secrets_or_private_path_findings':findings,'prohibited_files':prohibited,'syntax_errors':syntax,'ignore_checks':checks,'scope':'Regex scan of exact git-index files plus size/symlink/prohibited-type/syntax checks; not a proof against all unknown secrets.','passed':not findings and not prohibited and not syntax and all(checks.values())}
print(json.dumps(result,indent=2));(ROOT/'reports/SECURITY_CHECKS.json').write_text(json.dumps(result,indent=2));(ROOT/'reports/RELEASE_FILE_MANIFEST.json').write_text(json.dumps({'excludes_self_and_security_receipt_to_avoid_circular_hashes':True,'files':files},indent=2))
assert result['passed'],'Release blocked; inspect paths/rules without revealing values.'
