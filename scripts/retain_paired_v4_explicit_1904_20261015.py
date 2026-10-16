"""Explicitly approved remaining1904 cleanup with prior640 deletion ledger."""

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path('/Users/artemvorozhtsov/projects/rf-knots')
REPOS = [ROOT, ROOT.parent / 'unknotdb', ROOT.parent / 'pgx-mcts-bench',
         Path('/Users/artemvorozhtsov/.codex/automations')]
RUNS = {arm: ROOT / 'artifacts' / f'joint-f-r2-r4-smoothed-v4-{arm}'
        for arm in ('R2', 'R4')}
FINAL = {'R2': '1ffe7346e0725fcb98125ebf03ab1faaaf7048985d69eeb00300e2828cca6946',
         'R4': '573f7891a24595e13250cc4f9cf54281024d8d59cceb2ff824c63d91f42d672f'}
PROD = ROOT.parent / 'unknotdb/models/q-grown-raster-axial-12-q254-frozen-20260824-v0'
PRIOR = ROOT / 'artifacts/paired-v4-retention-20261015-conservative-journal.json'
PRIOR_SHA = '73fc53ebd038b7d932e83baf0c9351ea611995bc4faa30f2baa4283db80cb3e0'
PROPOSAL = ROOT / 'artifacts/paired-v4-retention-20261015-plan.json'
PROPOSAL_SHA = '9f93d507dae47d569efd873110a83b2c22b463c77258423d922514079dc8cc89'
NAME = re.compile(r'step-\d{6}\.pt')
HASH = re.compile(r'\b[0-9a-f]{64}\b')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def atomic(path, value, new=False):
    if new and path.exists():
        raise ValueError('immutable plan/journal already exists')
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def live_check():
    result = subprocess.run(['ps', '-axo', 'pid,command'], check=True,
                            text=True, capture_output=True)
    for line in result.stdout.splitlines():
        if ('train_joint_f_r2_r4_smoothed_v4.py' in line
                or 'run_joint_f_r2_r4_smoothed_v4.py' in line
                or any(str(run) in line or str(run.relative_to(ROOT)) in line
                       for run in RUNS.values())):
            raise ValueError('target run may have a live consumer: ' + line)
    for run in RUNS.values():
        opened = subprocess.run(['lsof', '-nP', '+D', str(run)],
                                text=True, capture_output=True)
        if opened.returncode not in (0, 1) or opened.stdout.strip():
            raise ValueError('target open-file check not clean: ' + str(run))
    return sha(PROD / 'manifest.json'), sha(PROD / 'checkpoint.pt')


def inventory():
    protected_prod = live_check()
    if sha(PRIOR) != PRIOR_SHA or sha(PROPOSAL) != PROPOSAL_SHA:
        raise ValueError('prior deletion/proposal evidence pin differs')
    previous = json.loads(PRIOR.read_text())
    proposal = json.loads(PROPOSAL.read_text())
    missing = {e['path']: e for e in previous['deleted']}
    if (previous['status'] != 'completed' or len(missing) != 640
            or set(previous['executed']) != set(missing)):
        raise ValueError('prior640 completed deletion ledger differs')
    approved = {e['path'] for e in proposal['deleted']} - set(missing)
    if len(approved) != 1904:
        raise ValueError('explicitly approved remaining upper limit differs')
    prod = json.loads((PROD / 'manifest.json').read_text())
    if protected_prod[1] != prod['checkpoint_sha256']:
        raise ValueError('production model pin differs')
    manifests, entries, names, hashes = {}, [], set(), set()
    tracked = set(subprocess.check_output(['git', '-C', str(ROOT), 'ls-files'],
                                          text=True).splitlines())
    for arm, run in RUNS.items():
        if run.is_symlink() or not run.is_dir():
            raise ValueError('target directory identity differs')
        mp = run / 'manifest.json'
        m = json.loads(mp.read_text())
        points = m['checkpoints']
        if (m['status'] != 'checkpointed' or m['optimizer_steps'] != 114561
                or m['active_seconds'] < 43200 or len(points) != 1305
                or points[-1]['sha256'] != FINAL[arm]):
            raise ValueError('completed paired run identity differs')
        manifests[str(mp)] = sha(mp)
        if manifests[str(mp)] != previous['manifest_pins'][str(mp)]:
            raise ValueError('original paired manifest changed since prior cleanup')
        milestones = {round(i * (len(points) - 1) / 32) for i in range(33)}
        for i, e in enumerate(points):
            p = run / e['file']
            if not p.exists() and not p.is_symlink():
                old = missing.get(str(p))
                if old is None or old['sha256'] != e['sha256'] or old['step'] != e['step']:
                    raise ValueError('unaccounted missing checkpoint: ' + str(p))
                continue
            if str(p) in missing:
                raise ValueError('previously removed file was restored; fresh audit required')
            st = p.lstat()
            if (p.parent != run or e['file'] != f"step-{e['step']:06d}.pt"
                    or p.is_symlink() or not p.is_file() or st.st_nlink != 1
                    or str(p.relative_to(ROOT)) in tracked or sha(p) != e['sha256']):
                raise ValueError('unsafe/missing/changed/tracked checkpoint: ' + str(p))
            entries.append({'path': str(p), 'arm': arm, 'step': e['step'],
                            'sha256': e['sha256'], 'bytes': st.st_size,
                            'inode': st.st_ino, 'device': st.st_dev,
                            'mtime_ns': st.st_mtime_ns, 'milestone': i in milestones})
            names.add(e['file']); hashes.add(e['sha256'])
    roots = list(REPOS)
    for repo in REPOS[:3]:
        listing = subprocess.check_output(['git', '-C', str(repo), 'worktree', 'list', '--porcelain'], text=True)
        for line in listing.splitlines():
            if line.startswith('worktree '):
                p = Path(line.removeprefix('worktree '))
                if not any(p == root or root in p.parents for root in roots):
                    roots.append(p)
    command = ['rg', '--files', '--hidden', '--no-ignore', *map(str, roots)]
    for glob in ['*.json', '*.jsonl', '*.md', '*.py', '*.toml', '*.yaml', '*.yml',
                 '*.sh', '*.txt', '*.js', '*.ts', '*.tsx', '*.rs', '*.plist']:
        command += ['-g', glob]
    for folder in ['.git', '.venv', 'node_modules', 'target', '__pycache__']:
        command += ['-g', f'!**/{folder}/**']
    files = subprocess.check_output(command, text=True).splitlines()
    scan = subprocess.run(['rg', '-l', '-F', '--hidden', '--no-ignore', '-f', '-',
                           *map(str, roots), *command[command.index('-g'):]],
                          input='\n'.join(sorted(names | hashes)) + '\n',
                          text=True, capture_output=True)
    if scan.returncode not in (0, 1):
        raise ValueError('dependency scan failed: ' + scan.stderr[:500])
    by_name, by_hash, evidence = set(), set(), []
    excluded = set(manifests) | {str(Path(__file__).resolve())}
    for name in scan.stdout.splitlines():
        p = Path(name)
        if str(p) in excluded or p.name.startswith('paired-v4-retention-20261015'):
            continue
        matched_names, matched_hashes, tail = set(), set(), ''
        with p.open('r', errors='ignore') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), ''):
                text = tail + chunk
                matched_names.update(set(NAME.findall(text)) & names)
                matched_hashes.update(set(HASH.findall(text)) & hashes)
                tail = text[-512:]
        if matched_names or matched_hashes:
            by_name.update(matched_names); by_hash.update(matched_hashes)
            evidence.append({'file': str(p), 'sha256': sha(p),
                             'names': sorted(matched_names), 'hashes': sorted(matched_hashes)})
    retained, deleted = [], []
    for e in entries:
        e['referenced'] = Path(e['path']).name in by_name or e['sha256'] in by_hash
        (retained if e['milestone'] or e['referenced'] or e['path'] not in approved else deleted).append(e)
    if len(deleted) > 1904 or sum(e['bytes'] for e in deleted) > 6393580592:
        raise ValueError('fresh approval count/bytes exceeded')
    return {'schema': 'rf-knots-explicitly-approved-remaining1904-retention-20261015',
            'status': 'planned', 'manifest_pins': manifests,
            'production_pins': {str(PROD / 'manifest.json'): protected_prod[0],
                                str(PROD / 'checkpoint.pt'): protected_prod[1]},
            'reference_files_scanned': len(files), 'reference_evidence': evidence,
            'retained': retained, 'deleted': deleted,
            'delete_bytes': sum(e['bytes'] for e in deleted),
            'script_sha256': sha(Path(__file__)),
            'prior_completed_journal_sha256': PRIOR_SHA,
            'previously_removed_count': 640}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    report = inventory()
    if not args.execute:
        atomic(args.plan, report, new=True)
    else:
        old = json.loads(args.plan.read_text())
        if any(old[k] != report[k] for k in ('manifest_pins', 'production_pins',
                                            'retained', 'deleted', 'script_sha256')):
            raise ValueError('targets/pins/reference protection changed after plan')
        journal = args.plan.with_name('paired-v4-retention-20261015-expanded-journal.json')
        report.update(status='executing', plan_sha256=sha(args.plan), executed=[])
        atomic(journal, report, new=True)
        live_check()
        for e in report['deleted']:
            p = Path(e['path']); st = p.lstat()
            if (p.parent != RUNS[e['arm']] or p.is_symlink() or st.st_nlink != 1
                    or (st.st_ino, st.st_dev, st.st_mtime_ns, st.st_size)
                    != (e['inode'], e['device'], e['mtime_ns'], e['bytes'])
                    or sha(p) != e['sha256']):
                raise ValueError('target changed immediately before removal')
            p.unlink()
            report['executed'].append(e['path'])
            if len(report['executed']) % 100 == 0:
                atomic(journal, report)
        for e in report['retained']:
            if sha(Path(e['path'])) != e['sha256']:
                raise ValueError('retained snapshot SHA differs')
        for p, h in {**report['manifest_pins'], **report['production_pins']}.items():
            if sha(Path(p)) != h:
                raise ValueError('retained manifest/production pin differs')
        report['status'] = 'completed'
        atomic(journal, report)
    print(json.dumps({'status': report['status'], 'delete': len(report['deleted']),
                      'retain': len(report['retained']), 'GiB': report['delete_bytes'] / 1024**3,
                      'reference_files': report['reference_files_scanned']}, sort_keys=True))


if __name__ == '__main__':
    main()
