"""Recreate full audit-only vendor checkouts from sources.lock.json.

Runtime uses the compact, committed ``third_party`` snapshots.  This helper is
only for inspecting the complete upstream repositories and never overwrites an
existing checkout.
"""
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def main():
    lock=json.loads((ROOT/'sources.lock.json').read_text(encoding='utf-8'))
    for name,entry in lock.items():
        dest=ROOT/'vendor'/name
        if dest.exists():
            sha=subprocess.check_output(['git','-c',f'safe.directory={dest.as_posix()}','-C',str(dest),'rev-parse','HEAD'],text=True).strip()
            if sha!=entry['commit']:
                raise RuntimeError(f'{name}: local commit differs; refusing to overwrite')
            print(f'{name}: already present at {sha}')
            continue
        dest.mkdir(parents=True)
        commands=[['init',str(dest)],['-C',str(dest),'remote','add','origin',entry['url']],
                  ['-C',str(dest),'fetch','--depth','1','origin',entry['commit']],
                  ['-C',str(dest),'-c','core.longpaths=true','checkout','--detach','FETCH_HEAD']]
        for cmd in commands:
            subprocess.run(['git',*cmd],check=True)

if __name__=='__main__': main()
