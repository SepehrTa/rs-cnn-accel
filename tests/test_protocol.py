"""Independent packet interpreter: tests host compilation without FPGA/PYNQ."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'host'),str(ROOT/'model'),str(ROOT/'mapper')]
from protocol import conv_packets, forward, add_packet, fc_packet
from graph import resnet8_graph
from nn_numpy import forward_int, requant, quantize_multiplier, checked_int32
from rs_model import legal, Layer, Mapping, HwConfig, count, best_mapping
from accelerator import Accelerator


def interpret(packet, size):
    p=packet.view(np.int32).astype(np.int64)
    h=p[:16]; data=p[16:]; out=[]
    if h[0]==0:
        _,k,u,pw,ow,e,f,c,relu=h[:9]
        ph=(e-1)*u+k
        acc=np.zeros((f,e,ow),np.int64)
        offset=0
        # Direct scalar convolution independent of NumPy im2col/RS simulation.
        for ch in range(c):
            x=data[offset:offset+ph*pw].reshape(ph,pw); offset+=ph*pw
            w=data[offset:offset+f*k*k].reshape(f,k,k); offset+=f*k*k
            for g in range(f):
                for y in range(e):
                    for xx in range(ow):
                        for r in range(k):
                            for s in range(k):
                                acc[g,y,xx]+=int(x[y*u+r,xx*u+s])*int(w[g,r,s])
        for g in range(f):
            b,m,sh=map(int,h[9+g*3:12+g*3])
            for a in acc[g].ravel():
                v=((int(a)+b)*m+(1<<(sh-1))) >> sh
                out.append(max(0 if relu else -128,min(127,v)))
    elif h[0]==1:
        for a,b in data.reshape(-1,2):
            y=(int(a)*int(h[10])+int(b)*int(h[13])+(1<<(int(h[11])-1))) >> int(h[11])
            out.append(max(0 if h[8] else -128,min(127,y)))
    elif h[0]==2:
        out=data.reshape(int(h[2]),int(h[1])).sum(1)
    elif h[0]==3:
        n,m=int(h[1]),int(h[2]); x=data[:n]
        for row in data[n:].reshape(m,n+1):
            out.append(int(row[0])+sum(int(a)*int(b) for a,b in zip(x,row[1:])))
    if len(out)!=size: raise AssertionError('output length mismatch')
    return np.asarray(out,np.int64)


class ProtocolTests(unittest.TestCase):
    def test_dma_transport_order_and_cleanup(self):
        events=[]
        class Buffer(np.ndarray):
            def freebuffer(self): events.append('free')
            def invalidate(self): events.append('invalidate')
        def allocate(shape,dtype): return np.zeros(shape,dtype).view(Buffer)
        class Channel:
            idle=True
            error=False
            def __init__(self,name): self.name=name
            def transfer(self,buf):
                events.append(self.name)
                self.transferred=buf.nbytes
                if self.name=='recv': buf[:]=[-128,127]
            def wait(self): events.append('wait'+self.name)
            def stop(self): events.append('stop')
        mmio=SimpleNamespace(read=lambda off:{0:0x52530001,0x14:0x02030820,8:0}[off],
                             write=lambda off,val:events.append('reset'))
        dma=SimpleNamespace(sendchannel=Channel('send'),recvchannel=Channel('recv'))
        overlay=SimpleNamespace(axi_dma_0=dma,rs_accel_0=SimpleNamespace(mmio=mmio))
        driver=Accelerator(overlay)
        with patch.dict(sys.modules,{'pynq':SimpleNamespace(allocate=allocate)}):
            result=driver(np.array([1,2,3],np.uint32),2)
            np.testing.assert_array_equal(result,[-128,127])
            self.assertIs(type(result),np.ndarray)
            self.assertEqual(events,['reset','recv','send','waitsend','waitrecv','invalidate','free','free'])
            events.clear()
            dma.sendchannel.idle=False
            dma.sendchannel.error=True
            with self.assertRaisesRegex(RuntimeError,'DMA error'):
                driver(np.array([1],np.uint32),2)
            self.assertEqual(events,['recv','send','stop','stop','free','free'])

    def test_all_graph_operations(self):
        rng=np.random.default_rng(20)
        # Same residual topology, reduced channels/spatial size for scalar oracle.
        graph=resnet8_graph()
        for l in graph:
            if 'cin' in l: l['cin']=min(l['cin'],3)
            if 'cout' in l: l['cout']=3
        q={}
        for l in graph:
            if l['op']=='conv':
                q[l['name']]=dict(w=rng.integers(-8,9,(3,3,l['k'],l['k']),dtype=np.int8),
                    b=rng.integers(-30,31,3),M=np.full(3,2**30),S=np.full(3,4))
            elif l['op']=='add': q[l['name']]=dict(Ma=2**30,Mb=2**29,S=0)
            elif l['op']=='fc': q[l['name']]=dict(w=rng.integers(-8,9,(3,3),dtype=np.int8),b=rng.integers(-30,31,3))
        x=rng.integers(-128,128,(1,3,9,37),dtype=np.int8)
        got=forward(interpret,q,graph,x)
        ref=forward_int(q,graph,x)
        for name in ref:
            np.testing.assert_array_equal(got[name],ref[name],err_msg=name)

    def test_reject_overflow(self):
        with self.assertRaisesRegex(ValueError,'overflow'):
            fc_packet(np.full(64,2**30),dict(w=np.full((10,64),127),b=np.zeros(10)))

    def test_reject_bad_shift(self):
        with self.assertRaisesRegex(ValueError,'shift'):
            add_packet(np.zeros(2),np.zeros(2),dict(Ma=1,Mb=1,S=-31),False)

    def test_negative_ties(self):
        np.testing.assert_array_equal(requant(np.array([-3,-1,1,3]),2**30,0),[-1,0,1,2])

    def test_mapping_positivity(self):
        self.assertFalse(legal(Layer('x',3,4,8,3,1,1),Mapping(1,0,1,1,1),HwConfig()))

    def test_analytical_capacity_is_enforced(self):
        layer=Layer('x',3,4,8,3,1,1)
        hw=HwConfig(glb_bytes=16)
        mapping=Mapping(1,1,1,1,1)
        self.assertFalse(legal(layer,mapping,hw))
        with self.assertRaisesRegex(ValueError,'capacity'): count(layer,mapping,hw)
        with self.assertRaisesRegex(ValueError,'capacity'): best_mapping(layer,hw)
        with self.assertRaisesRegex(ValueError,'no legal'): best_mapping(layer,HwConfig(rows=1))

    def test_golden_rejects_invalid_ranges(self):
        for m in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError): quantize_multiplier(m)
        with self.assertRaisesRegex(ValueError,'overflow'): checked_int32([2**31],'test bias')
        with self.assertRaisesRegex(ValueError,'shift'): requant(np.array([1]),2**30,32)


if __name__=='__main__': unittest.main()
