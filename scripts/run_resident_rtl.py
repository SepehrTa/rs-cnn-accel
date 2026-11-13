"""Run real BRAM-dependent resident graph simulation against all layer tensors."""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'host'),str(ROOT/'scripts')]
from resident import compile_model
from protocol import load_vector
from run_rtl import run

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--archive',required=True)
    ap.add_argument('--image',type=int,default=0)
    ap.add_argument('--vivado',default='C:/AMDDesignTools/2025.2/Vivado')
    ap.add_argument('--cols',type=int,default=14)
    ap.add_argument('--filters',type=int,default=4)
    ap.add_argument('--p',type=int,default=2)
    ap.add_argument('--q',type=int,default=2)
    a=ap.parse_args()
    folder=ROOT/'build'/'sim'/f'resident-f{a.filters}-c{a.cols}-p{a.p}-q{a.q}-image{a.image}'
    folder.mkdir(parents=True,exist_ok=True)
    model=compile_model(a.archive)
    _,graph,t=load_vector(a.archive,a.image)
    packets=[model.upload_packet(),model.input_packet(t['input']),model.input_packet(t['input'])]
    np.savetxt(folder/'stim.hex',np.concatenate(packets),fmt='%08x')
    expected=[]
    for l in graph:
        expected.extend(t[l['dst']].astype(np.int32).view(np.uint32).ravel())
    np.savetxt(folder/'expect.hex',np.asarray(expected,np.uint32),fmt='%08x')
    (folder/'tb_config.sv').write_text(f'module tb_config; tb_resident #(.INPUTS({sum(map(len,packets))}),.GOLDEN({len(expected)}),.COLS({a.cols}),.FILTERS({a.filters}),.P_CONTEXTS({a.p}),.Q_CONTEXTS({a.q})) test(); endmodule\n')
    b=Path(a.vivado)/'bin';ext='.bat' if os.name=='nt' else ''
    run([b/f'xvlog{ext}','--sv',*sorted((ROOT/'hw').glob('*.sv')),ROOT/'sim'/'tb_resident.sv',folder/'tb_config.sv'],folder)
    run([b/f'xelab{ext}','tb_config','--timescale','1ns/1ps','-s','resident_snapshot'],folder)
    run([b/f'xsim{ext}','resident_snapshot','-runall'],folder)
    s=(folder/'xsim.stdout.txt').read_text()
    if 'PASS RESIDENT' not in s or 'Fatal:' in s or 'Error:' in s: raise RuntimeError('resident simulation failed')
if __name__=='__main__': main()
