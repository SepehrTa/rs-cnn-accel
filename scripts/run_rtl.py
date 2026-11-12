"""Compile once and run XSim against deterministic packet vectors.

python scripts/run_rtl.py --vivado C:/vivado/2026.1/Vivado --suite directed
python scripts/run_rtl.py --vivado C:/vivado/2026.1/Vivado --suite network
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'model'), str(ROOT/'host')]
from graph import resnet8_graph
from nn_numpy import conv2d, forward_int, requant
from protocol import conv_packets, add_packet, pool_packet, fc_packet, load_model


class Vectors:
    def __init__(self, folder, cols=8):
        self.folder = folder
        self.cols=cols
        folder.mkdir(parents=True, exist_ok=True)
        self.stim = (folder/'stim.hex').open('w')
        self.expect = (folder/'expect.hex').open('w')
        self.outputs = 0
        self.inputs = 0
        self.packets = 0
        self.macs = self.act_writes = self.weight_writes = 0
        self.tensors = {}

    def emit(self, packet, expected, refs=None):
        h=packet[:16].astype(np.int64)
        if h[0]==0:
            k,pw,ow,e,f,c=(int(h[i]) for i in (1,3,4,5,6,7))
            self.macs+=c*f*e*ow*k*k
            self.act_writes+=c*f*k*e*pw
            self.weight_writes+=c*f*k*k*e
        p = packet.astype(np.uint64)
        if refs is not None:
            pos, indices = refs
            p[pos] = (1 << 32) | np.asarray(indices, dtype=np.uint64)
        self.stim.writelines(f'{int(v):016x}\n' for v in p)
        y = np.asarray(expected).ravel().astype(np.int64).astype(np.uint32).astype(np.uint64)
        y[-1] |= np.uint64(1 << 32)
        self.expect.writelines(f'{int(v):09x}\n' for v in y)
        indices = np.arange(self.outputs, self.outputs+len(y), dtype=np.int64)
        self.outputs += len(y)
        self.inputs += len(p)
        self.packets += 1
        return indices

    def close(self):
        self.stim.close()
        self.expect.close()
        (self.folder/'metrics.txt').write_text(f'{self.macs} {self.act_writes} {self.weight_writes}\n')
        (self.folder/'manifest.json').write_text(json.dumps(dict(
            inputs=self.inputs, outputs=self.outputs, packets=self.packets,
            macs=self.macs,pe_activation_writes=self.act_writes,
            pe_weight_writes=self.weight_writes,tensors=self.tensors), indent=2))


def conv(v, x, p, layer, ref, refs=None):
    outrefs = np.empty(ref.shape, np.int64)
    for packet, region, _ in conv_packets(x, p, layer,cols=v.cols):
        references = None
        if refs is not None:
            # Reuse the packet compiler's address generation on index tensors.
            # Sentinel -1 padding is emitted as a literal zero, never an address.
            f, y, ox = region
            h = packet[:16].astype(np.int64)
            k, u, pw, nr, nf, nc = (int(h[i]) for i in (1, 2, 3, 5, 6, 7))
            ph = (nr-1)*u+k
            rp = np.pad(refs, ((0,0),(layer['pad'],layer['pad']),(layer['pad'],layer['pad'])), constant_values=-1)
            positions, addresses = [], []
            pos = 16
            for c in range(nc):
                patch = rp[c,y.start*u:y.start*u+ph,ox.start*u:ox.start*u+pw].ravel()
                valid = np.flatnonzero(patch >= 0)
                positions.extend(pos+valid)
                addresses.extend(patch[valid])
                pos += ph*pw+nf*k*k
            references = (positions, addresses)
        outrefs[region] = v.emit(packet, ref[region], references).reshape(ref[region].shape)
    return outrefs


def directed(v):
    rng = np.random.default_rng(20261112)
    # Pure bias tests exercise signed INT32 extremes and every supported shift,
    # without arithmetic overflow in a MAC or reference floating convolution.
    for shift in (1,2,15,30,31,32,45,62):
        p=dict(w=np.zeros((2,1,1,1),np.int8),b=np.array([-2**31,2**31-1]),
               M=np.array([2**31-1,2**30+3]),S=np.full(2,shift-31))
        layer=dict(k=1,stride=1,pad=0,relu=False)
        y=np.array([max(-128,min(127,(int(b)*int(m)+(1<<(shift-1)))>>shift))
                    for b,m in zip(p['b'],p['M'])]).reshape(2,1,1)
        conv(v,np.zeros((1,1,1),np.int8),p,layer,y)
    for case in range(18):
        k = (1,3)[case%2]
        u = (1,2)[case//2%2]
        c, m, h, w = (int(rng.integers(1,n)) for n in (5,6,19,39))
        h, w = max(h,k), max(w,k)
        x = rng.integers(-128,128,(c,h,w),dtype=np.int64)
        weights = rng.integers(-128,128,(m,c,k,k),dtype=np.int64)
        p = dict(w=weights,b=rng.integers(-1000,1001,m),M=rng.integers(2**30,2**31,m),S=rng.integers(-2,16,m))
        layer = dict(k=k,stride=u,pad=case%3, relu=bool(case%2))
        raw = conv2d(x[None],weights,u,layer['pad'])[0].astype(np.int64)+p['b'][:,None,None]
        ref = np.clip(requant(raw,p['M'][:,None,None],p['S'][:,None,None]),0 if layer['relu'] else -128,127)
        conv(v,x,p,layer,ref)
    for shift in (1,2,29,31,45,62):
        a = rng.integers(-128,128,(3,4,5),dtype=np.int64)
        b = rng.integers(-128,128,a.shape,dtype=np.int64)
        a.ravel()[:6] = [-128,-3,-1,0,1,127]
        b.ravel()[:6] = [127,0,0,0,0,-128]
        p = dict(Ma=1073741824,Mb=2147483647,S=shift-31)
        y = (a*p['Ma']+b*p['Mb']+(1<<(shift-1))) >> shift
        for relu in (False,True):
            v.emit(add_packet(a,b,p,relu),np.clip(y,0 if relu else -128,127))
    x = rng.integers(-128,128,(7,8,8),dtype=np.int64)
    pooled=x.sum((1,2))
    poolrefs=v.emit(pool_packet(x),pooled)
    p=dict(w=rng.integers(-128,128,(10,7)),b=rng.integers(-10000,10000,10))
    v.emit(fc_packet(pooled,p),pooled@p['w'].T+p['b'],(np.arange(16,23),poolrefs))


def network(v, archive=None):
    rng=np.random.default_rng(8)
    if archive:
        q,graph=load_model(archive)
        with np.load(archive,allow_pickle=False) as z:
            x=z['vec.input'][:1].astype(np.int64)
    else:
        graph=resnet8_graph()
        x=rng.integers(-128,128,(1,3,32,32),dtype=np.int64)
        q={}
        for l in graph:
            if l['op']=='conv':
                q[l['name']]=dict(w=rng.integers(-8,9,(l['cout'],l['cin'],l['k'],l['k']),dtype=np.int8),
                    b=rng.integers(-500,501,l['cout'],dtype=np.int32),
                    M=rng.integers(2**30,2**31,l['cout']),S=np.full(l['cout'],8))
            elif l['op']=='add': q[l['name']]=dict(Ma=1610612736,Mb=1073741824,S=0)
            elif l['op']=='fc': q[l['name']]=dict(w=rng.integers(-64,65,(10,64),dtype=np.int8),b=rng.integers(-10000,10001,10,dtype=np.int32))
    t=forward_int(q,graph,x)
    if archive:
        with np.load(archive,allow_pickle=False) as z:
            for name,value in t.items():
                if not np.array_equal(value,z[f'vec.{name}'][:1]):
                    raise AssertionError(f'archive tensor vec.{name} differs from golden recomputation')
    else:
        export={'graph':np.array(json.dumps(graph)),'vec.labels':np.array([-1]),'test_logits':t['logits'].astype(np.int32)}
        export.update({f'{name}.{key}':np.asarray(value) for name,p in q.items() for key,value in p.items()})
        export.update({f'vec.{name}':value.astype(np.int32) for name,value in t.items()})
        np.savez(v.folder/'synthetic_int8.npz',**export)
    refs={}
    for l in graph:
        dst=l['dst']; p=q.get(l['name']); expected=t[dst][0]
        before=v.outputs
        if l['op']=='conv': refs[dst]=conv(v,t[l['src']][0],p,l,expected,refs.get(l['src']))
        elif l['op']=='add':
            a,b=t[l['a']][0],t[l['b']][0]
            addr=np.stack((refs[l['a']].ravel(),refs[l['b']].ravel()),axis=1).ravel()
            refs[dst]=v.emit(add_packet(a,b,p,l['relu']),expected,(np.arange(16,16+len(addr)),addr)).reshape(expected.shape)
        elif l['op']=='gap':
            addr=refs[l['src']].ravel()
            refs[dst]=v.emit(pool_packet(t[l['src']][0]),expected,(np.arange(16,16+len(addr)),addr))
        elif l['op']=='fc':
            addr=refs[l['src']].ravel()
            refs[dst]=v.emit(fc_packet(t[l['src']][0],p),expected,(np.arange(16,16+len(addr)),addr))
        v.tensors[dst]=dict(start=before,count=v.outputs-before,shape=list(expected.shape))
    np.savez(v.folder/'golden.npz',**t)
    print('Golden classification:',int(t['logits'].argmax()), 'logits:',t['logits'].tolist(),flush=True)


def run(cmd,cwd):
    print('RUN', ' '.join(map(str,cmd)),flush=True)
    start=time.monotonic()
    r=subprocess.run(list(map(str,cmd)),cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    name=Path(cmd[0]).stem
    (cwd/f'{name}.stdout.txt').write_text(r.stdout,encoding='utf-8')
    print(r.stdout[-5000:],flush=True)
    if r.returncode: raise RuntimeError(f'{name} failed ({r.returncode})')
    print(f'{name}: {time.monotonic()-start:.2f}s',flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--vivado',default=os.environ.get('XILINX_VIVADO','C:/vivado/2026.1/Vivado'))
    ap.add_argument('--suite',choices=['directed','network','control','pe'],default='directed')
    ap.add_argument('--archive',help='existing quantize.py NPZ; network suite only')
    ap.add_argument('--cols',type=int,default=8,choices=range(1,33))
    args=ap.parse_args()
    folder=ROOT/'build'/'sim'/(args.suite if args.cols==8 else f'{args.suite}-cols{args.cols}')
    folder.mkdir(parents=True,exist_ok=True)
    functional=args.suite in ('directed','network')
    if functional:
        v=Vectors(folder,args.cols)
        (directed(v) if args.suite=='directed' else network(v,args.archive))
        v.close()
    bin=Path(args.vivado)/'bin'
    suffix='.bat' if os.name=='nt' else ''
    tb={'control':'tb_control','pe':'tb_pe'}.get(args.suite,'tb_stream')
    extra=[]
    top=tb
    if functional:
        wrapper=folder/'tb_config.sv'
        wrapper.write_text(f'module tb_config; tb_stream #(.COLS({args.cols})) test(); endmodule\n')
        extra=[wrapper]
        top='tb_config'
    run([bin/f'xvlog{suffix}','--sv',*sorted((ROOT/'hw').glob('*.sv')),ROOT/'sim'/f'{tb}.sv',*extra],folder)
    run([bin/f'xelab{suffix}',top,'-s',f'{tb}_snapshot'],folder)
    run([bin/f'xsim{suffix}',f'{tb}_snapshot','-runall'],folder)
    output=(folder/'xsim.stdout.txt').read_text()
    marker={'control':'PASS CONTROL','pe':'PASS PE'}.get(args.suite,'PASS RTL')
    if marker not in output or 'Fatal:' in output or 'Error:' in output:
        raise RuntimeError('simulation did not finish successfully')
    if functional:
        print(f'PASS {args.suite}: {v.inputs} input words, {v.outputs} exact output words, {v.packets} packets')


if __name__=='__main__': main()
