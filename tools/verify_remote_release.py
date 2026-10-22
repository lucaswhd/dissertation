#!/usr/bin/env python3
"""Verify a private GitHub branch against HEAD and hash every fetched file.

Uses the existing gh/git credentials without displaying credential material.
Writes only the requested verification receipt; does not push or change history.
"""
import argparse,hashlib,io,json,os,subprocess,tarfile
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ENV=dict(os.environ);ENV.pop('GH_DEBUG',None)

def run(args):
    result=subprocess.run(args,cwd=ROOT,env=ENV,capture_output=True,check=False)
    if result.returncode:raise RuntimeError(f'{args[0]} command failed with exit {result.returncode}; credential-bearing output suppressed')
    return result.stdout

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',default='lucaswhd/dissertation')
    ap.add_argument('--branch',default='main')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    run(['git','fetch','origin'])
    metadata=json.loads(run(['gh','api',f'repos/{args.repo}']))
    ref=json.loads(run(['gh','api',f'repos/{args.repo}/git/ref/heads/{args.branch}']))
    remote=ref['object']['sha'];head=run(['git','rev-parse','HEAD']).decode().strip()
    fetched=run(['git','rev-parse',f'origin/{args.branch}']).decode().strip()
    commit=json.loads(run(['gh','api',f'repos/{args.repo}/git/commits/{remote}']))
    tree_sha=commit['tree']['sha']
    tree=json.loads(run(['gh','api',f'repos/{args.repo}/git/trees/{tree_sha}?recursive=1']))
    assert metadata['private'] and remote==head==fetched and not tree['truncated']
    assert tree['sha']==run(['git','rev-parse','HEAD^{tree}']).decode().strip()
    remote_blobs={x['path']:x['sha'] for x in tree['tree'] if x['type']=='blob'}
    hashes=[]
    archive=run(['git','archive',remote])
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:') as tar:
        for member in tar:
            if member.isdir():continue
            assert member.isfile(),member.name
            data=tar.extractfile(member).read();path=ROOT/member.name
            local=path.read_bytes()
            assert local==data,member.name
            blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            assert remote_blobs[member.name]==blob,member.name
            hashes.append({'path':member.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'github_blob_sha':blob})
    assert {x['path'] for x in hashes}==set(remote_blobs)
    prohibited=[x['path'] for x in hashes if Path(x['path']).suffix.lower() in {'.docx','.h5','.hdf5','.npz','.parquet','.mp4','.pt','.pth','.ckpt','.safetensors'} or 'fig_3_3_task_storyboard' in x['path'] or (x['path'].startswith('manuscript/') and x['path'].endswith('.pdf'))]
    assert not prohibited,prohibited
    result={'verified_utc':datetime.now(timezone.utc).isoformat(),'repository':f'https://github.com/{args.repo}',
            'branch':args.branch,'private':metadata['private'],'remote_commit_sha':remote,
            'local_head_sha':head,'remote_tree_sha':tree['sha'],'tracked_files_verified':len(hashes),
            'all_local_remote_sha256_match':True,'prohibited_files':prohibited,
            'method':'GitHub API branch/tree identity plus freshly fetched git archive; every file compared to worktree and GitHub blob identity, SHA256 recorded.',
            'files':hashes,'passed':True}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))

if __name__=='__main__':main()
