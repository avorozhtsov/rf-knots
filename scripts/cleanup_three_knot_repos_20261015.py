"""Authorized old checkpoint retention; explicit regular-file targets only."""
import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path('/Users/artemvorozhtsov/projects/rf-knots')
REPOS = [ROOT, ROOT.parent/'unknotdb', ROOT.parent/'pgx-mcts-bench']
RUN_NAMES = [
 'raster-long-general-training-continuation-v1/long',
 'raster-long-general-training-continuation-v1/control',
 'raster-long-general-training-pair-v1/long',
 'raster-long-general-training-pair-v1/control',
 'raster-resource-units-training-pair-v1/units',
 'raster-resource-units-training-pair-v1/raw',
 'joint-f-r2-r4-pilot-v1-R2', 'joint-f-r2-r4-pilot-v1-R4',
 'additive-root-only-r2-v3', 'additive-f-r2-stage-v2',
 'joint-f-r2-continuation-v1', 'additive-root-only-r2-v5',
 'additive-root-only-r2-v6', 'additive-root-only-r2-v7',
 'additive-root-only-r2-v8', 'additive-root-only-r2-v9']
RUNS = [ROOT/'artifacts'/n for n in RUN_NAMES]
PROD = REPOS[1]/'models/q-grown-raster-axial-12-q254-frozen-20260824-v0'
PAT = re.compile(r'step-\d{6}(?:-[0-9a-f]{16})?\.pt')
HASH = re.compile(r'\b[0-9a-f]{64}\b')

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()

def atomic(p,d):
 t=p.with_suffix(p.suffix+'.tmp')
 with t.open('x') as f:
  json.dump(d,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 t.replace(p)

def live():
 ps=subprocess.check_output(['ps','-axo','pid,command'],text=True)
 for line in ps.splitlines():
  if any(str(p) in line or str(p.relative_to(ROOT)) in line for p in RUNS):
   raise ValueError('Live target consumer: '+line)
 for p in RUNS:
  r=subprocess.run(['lsof','-nP','+D',str(p)],text=True,capture_output=True)
  if r.returncode not in (0,1) or r.stdout.strip():raise ValueError('Open/unknown target files '+str(p))
 return {str(p):sha(p) for p in [PROD/'manifest.json',PROD/'checkpoint.pt']}

def inventory():
 prod=live();pins={};entries=[];names=set();hashes=set()
 tracked=set(subprocess.check_output(['git','-C',str(ROOT),'ls-files'],text=True).splitlines())
 for run in RUNS:
  m=run/'manifest.json';d=json.loads(m.read_text());pins[str(m)]=sha(m)
  if d['status'] not in ['checkpointed','completed','round_completed']:raise ValueError('Unfinished '+str(run))
  if run.is_symlink():raise ValueError('Symlink run')
  files=sorted(run.glob('step-*.pt'),key=lambda p:int(p.name.split('-')[1].split('.')[0]))
  if len(files)<34:raise ValueError('Unexpected small run')
  milestones={round(i*(len(files)-1)/32) for i in range(33)}
  expected={e['file']:e['sha256'] for e in d.get('checkpoints',[])}
  for i,p in enumerate(files):
   st=p.lstat();h=sha(p)
   if p.is_symlink() or not p.is_file() or st.st_nlink!=1 or str(p.relative_to(ROOT)) in tracked:raise ValueError('Unsafe '+str(p))
   if expected and expected.get(p.name)!=h:raise ValueError('Manifest mismatch '+str(p))
   if not expected:
    suffix=p.stem.split('-')[-1]
    if len(suffix)!=16 or not h.startswith(suffix):raise ValueError('Hashed filename mismatch')
   entries.append(dict(path=str(p),sha256=h,bytes=st.st_size,inode=st.st_ino,device=st.st_dev,mtime_ns=st.st_mtime_ns,milestone=i in milestones))
   names.add(p.name);hashes.add(h)
  # Endpoint and seed checkpoint are never candidates, even if index logic changes.
  for key in ['initial_checkpoint','checkpoint']:
   e=d.get(key)
   if e:
    names.add(e['file']);hashes.add(e['sha256'])
 roots=REPOS+[Path('/Users/artemvorozhtsov/.codex/automations')]
 for repo in REPOS:
  for line in subprocess.check_output(['git','-C',str(repo),'worktree','list','--porcelain'],text=True).splitlines():
   if line.startswith('worktree '):
    p=Path(line[9:])
    if not any(p==r or r in p.parents for r in roots):roots.append(p)
 globs=[]
 for g in ['*.json','*.jsonl','*.md','*.py','*.toml','*.yaml','*.yml','*.sh','*.txt','*.js','*.ts','*.tsx','*.rs','*.plist']:globs+=['-g',g]
 for g in ['.git','.venv','node_modules','target','__pycache__']:globs+=['-g','!**/'+g+'/**']
 cmd=['rg','--hidden','--no-ignore',*map(str,roots),*globs]
 allfiles=subprocess.check_output(['rg','--files','--hidden','--no-ignore',*map(str,roots),*globs],text=True).splitlines()
 r=subprocess.run(['rg','-l','-F','--hidden','--no-ignore','-f','-',*map(str,roots),*globs],input='\n'.join(sorted(names|hashes))+'\n',capture_output=True,text=True)
 if r.returncode not in (0,1):raise ValueError(r.stderr)
 refs=set();evidence=[]
 for f in r.stdout.splitlines():
  p=Path(f)
  if p.resolve()==Path(__file__).resolve() or p.name.startswith('cleanup-three-knot-repos-20261015'):continue
  txt=p.read_text(errors='ignore')
  if p.parent in RUNS and (p.name=='manifest.json' or p.name.startswith('metrics-')):
   # Historical output enumeration is evidence, not a live input dependency.
   d=json.loads(txt)
   for key in ['checkpoints','checkpoint','metrics_history']:d.pop(key,None)
   txt=json.dumps(d)
  ns=set(PAT.findall(txt))&names;hs=set(HASH.findall(txt))&hashes
  if ns or hs:refs.update(ns|hs);evidence.append(dict(file=f,sha256=sha(p)))
 # Explicit initial/final and every external exact name/hash reference are protected.
 for run in RUNS:
  d=json.loads((run/'manifest.json').read_text())
  for key in ['initial_checkpoint','checkpoint']:
   if d.get(key):refs.update([d[key]['file'],d[key]['sha256']])
 retained=[];deleted=[]
 for e in entries:
  e['referenced']=Path(e['path']).name in refs or e['sha256'] in refs
  (retained if e['milestone'] or e['referenced'] else deleted).append(e)
 return dict(status='planned',manifest_pins=pins,production_pins=prod,reference_files_scanned=len(allfiles),reference_evidence=evidence,retained=retained,deleted=deleted,delete_bytes=sum(e['bytes'] for e in deleted),script_sha256=sha(Path(__file__)))

def main():
 a=argparse.ArgumentParser();a.add_argument('--plan',type=Path,required=True);a.add_argument('--execute',action='store_true');args=a.parse_args()
 report=inventory()
 if not args.execute:
  if args.plan.exists():raise ValueError('Plan already exists')
  atomic(args.plan,report)
 else:
  old=json.loads(args.plan.read_text())
  for k in ['manifest_pins','production_pins','retained','deleted','script_sha256']:
   if old[k]!=report[k]:raise ValueError('Changed after plan '+k)
  journal=args.plan.with_name('cleanup-three-knot-repos-20261015-journal.json')
  if journal.exists():raise ValueError('Do not repeat cleanup')
  report.update(status='executing',plan_sha256=sha(args.plan),executed=[]);atomic(journal,report);live()
  for e in report['deleted']:
   p=Path(e['path']);s=p.lstat()
   if p.parent not in RUNS or p.is_symlink() or s.st_nlink!=1 or (s.st_ino,s.st_dev,s.st_mtime_ns,s.st_size)!=(e['inode'],e['device'],e['mtime_ns'],e['bytes']) or sha(p)!=e['sha256']:raise ValueError('Target changed '+str(p))
   p.unlink();report['executed'].append(str(p))
   if len(report['executed'])%100==0:atomic(journal,report)
  for e in report['retained']:
   if sha(Path(e['path']))!=e['sha256']:raise ValueError('Kept changed')
  for f,h in {**report['manifest_pins'],**report['production_pins']}.items():
   if sha(Path(f))!=h:raise ValueError('Pinned evidence changed')
  report['status']='completed';atomic(journal,report)
 print(json.dumps(dict(status=report['status'],delete=len(report['deleted']),retain=len(report['retained']),GiB=report['delete_bytes']/2**30,reference_files=report['reference_files_scanned'])))

if __name__=='__main__':main()
