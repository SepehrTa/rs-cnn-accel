"""Snapshot real logs/reports and source/bitstream hashes for review.

Run after completed simulation and board builds. Fails on missing evidence;
never manufactures a pass result. Generated large vectors stay in build/.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--board',default='build/board-final')
    args=ap.parse_args()
    board=(ROOT/args.board).resolve()
    if not board.is_relative_to(ROOT/'build'):
        raise ValueError('board evidence must be under the repository build folder')
    dest=ROOT/'docs'/'reports'
    dest.mkdir(parents=True,exist_ok=True)
    for name in ('rs_accel.bit','rs_accel.hwh','utilization.rpt','timing.rpt','drc.rpt'):
        if not (board/name).is_file(): raise FileNotFoundError(board/name)
    for cmd,name in (([sys.executable,'tests/test_nn_numpy.py'],'python-numpy.txt'),
                     ([sys.executable,'-m','unittest','discover','-s','tests','-p','test_protocol.py','-v'],'python-unit.txt')):
        result=subprocess.run(cmd,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (dest/name).write_text('COMMAND: '+' '.join(cmd)+'\n'+result.stdout)
        if result.returncode: raise RuntimeError(f'{name}: failed')
    rs=(ROOT/'build'/'python-rs.txt').read_text(encoding='utf-8-sig')
    if 'ALL PASS' not in rs or 'FAIL ' in rs: raise RuntimeError('RS software tests not passing')
    (dest/'python-rs.txt').write_text('COMMAND: python tests/test_rs.py\n'+rs)
    perf=[]
    for suite in ('pe','control','directed','network','network-cols4','network-cols12'):
        folder=ROOT/'build'/'sim'/suite
        log=(folder/'xsim.stdout.txt').read_text()
        if 'PASS ' not in log or 'Fatal:' in log or 'Error:' in log:
            raise RuntimeError(f'{suite}: missing successful simulation')
        for tool in ('xvlog','xelab','xsim'):
            shutil.copyfile(folder/f'{tool}.stdout.txt',dest/f'{suite}-{tool}.txt')
        match=re.search(r'PASS RTL inputs=(\d+) outputs=(\d+) packets=(\d+) busy_cycles=(\d+) wall_cycles=(\d+)',log)
        if match and suite.startswith('network'):
            cols=8 if suite=='network' else int(suite.split('cols')[1])
            ins,outs,packets,busy,wall=map(int,match.groups())
            perf.append(dict(columns=cols,pes=cols*6,input_words=ins,output_words=outs,
                packets=packets,busy_cycles=busy,wall_cycles=wall,
                stream_payload_bytes=4*(ins+outs),
                simulated_ms_at_100mhz=wall/100000,
                pe_peak_utilization=12500992/(cols*6*wall)))
        if (folder/'manifest.json').exists():
            shutil.copyfile(folder/'manifest.json',dest/f'{suite}-manifest.json')
    for name in ('utilization','timing','drc'):
        shutil.copyfile(board/f'{name}.rpt',dest/f'board-{name}.rpt')
        shutil.copyfile(ROOT/'build'/'synth'/f'{name}_route.rpt',dest/f'ooc-{name}.rpt')
    boardlog=(ROOT/'build'/'board-final.log').read_text(errors='replace')
    # Keep warnings and run completion in a compact exact excerpt. Full generated
    # IP logs remain in build/board-final and are reproducible with board Tcl.
    lines=boardlog.splitlines()
    excerpt=[x for x in lines if any(s in x for s in ('WARNING:','ERROR:','completed successfully','Bitgen','Failed'))]
    (dest/'board-build-excerpt.txt').write_text('\n'.join(excerpt+lines[-25:])+'\n')
    perf.sort(key=lambda x:x['columns'])
    (dest/'performance.json').write_text(json.dumps(perf,indent=2)+'\n')
    sources=list((ROOT/'hw').glob('*'))+list((ROOT/'host').glob('*.py'))+list((ROOT/'model').glob('*.py'))
    sources+=list((ROOT/'scripts').glob('*'))+list((ROOT/'sim').glob('*.sv'))
    sources+=[board/'rs_accel.bit',board/'rs_accel.hwh',ROOT/'build'/'vendor'/'pynq_z2_ps.tcl']
    hashes={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(sources) if p.is_file()}
    (dest/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    (dest/'environment.txt').write_text(f'Python {sys.version}\nNumPy {np.__version__}\nOS {platform.platform()}\n'
        'Vivado/XSim 2026.1, build 6511674\nTarget xc7z020clg400-1; FCLK0 100 MHz\n'
        'Board execution: NOT RUN\nTrained accuracy: NOT RUN\n')
    print(json.dumps(perf,indent=2))
    print('Evidence captured in',dest)


if __name__=='__main__': main()
