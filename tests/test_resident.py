"""Model memory layout, residual lifetimes and signed numeric ABI checks."""
import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'host'),str(ROOT/'model')]
from graph import resnet8_graph
from resident import compile_model,MAGIC,BANK_BYTES,packed_bytes
from protocol import load_model

class ResidentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.path=Path(cls.temp.name)/'model.npz'
        graph=resnet8_graph();rng=np.random.default_rng(123)
        export={'graph':np.array(json.dumps(graph))}
        for l in graph:
            name=l['name']
            if l['op']=='conv':
                values=dict(w=rng.integers(-128,128,(l['cout'],l['cin'],l['k'],l['k']),dtype=np.int8),b=np.arange(l['cout'],dtype=np.int32)-13,M=np.full(l['cout'],2**30),S=np.full(l['cout'],3))
            elif l['op']=='add':values=dict(Ma=np.array(2**30),Mb=np.array(2**29),S=np.array(0))
            elif l['op']=='fc':values=dict(w=rng.integers(-128,128,(10,64),dtype=np.int8),b=np.arange(10,dtype=np.int32)-100)
            else:continue
            export.update({f'{name}.{k}':v for k,v in values.items()})
        np.savez(cls.path,**export)
        cls.image=compile_model(cls.path)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_signed_weight_and_numeric_roundtrip(self):
        m=self.image;q,_=load_model(self.path)
        b=m.words.astype('<u4').tobytes()
        for l,d in zip(m.graph,m.descriptors):
            p=q.get(l['name'])
            if l['op'] in ('conv','fc'):
                recovered=np.frombuffer(b,dtype=np.int8,count=p['w'].size,offset=int(d[11])).reshape(p['w'].shape)
                np.testing.assert_array_equal(recovered,p['w'])
                if l['op']=='conv':
                    values=m.words[int(d[12]):int(d[12])+3*len(p['b'])].view(np.int32).reshape(-1,3)
                    np.testing.assert_array_equal(values,np.column_stack((p['b'],p['M'],31+p['S'])))
    def test_residual_sources_live_until_last_use(self):
        m=self.image;live={'input'}
        remaining={n:sum(n in [l.get(k) for k in ('src','a','b')] for l in m.graph) for n in m.offsets}
        for l,d in zip(m.graph,m.descriptors):
            self.assertNotIn(int(d[3]),[m.offsets[n] for n in live])
            self.assertLessEqual(int(d[13])*(4 if d[14] else 1),BANK_BYTES)
            live.add(l['dst'])
            for k in ('src','a','b'):
                n=l.get(k)
                if n:
                    remaining[n]-=1
                    if remaining[n]==0:live.remove(n)
            self.assertLessEqual(len(live),3)
        self.assertEqual(m.macs,12500992)
        self.assertEqual(len(m.words),20607)
        self.assertEqual(m.words[0],MAGIC)
    def test_packet_lengths_signed_image_and_int32_logits(self):
        m=self.image;x=np.arange(3072,dtype=np.int32).reshape(1,3,32,32)%256-128
        p=m.input_packet(x)
        self.assertEqual(len(p),784)
        np.testing.assert_array_equal(p[16:].astype('<u4').view(np.int8).reshape(x.shape),x)
        logits=np.array([-2**31,-1000,-1,0,1,2,3,4,1000,2**31-1],np.int32)
        np.testing.assert_array_equal(m.decode_output(logits.view(np.uint32)),logits[None])
        self.assertEqual(m.output_count(1),4096)
        with self.assertRaises(ValueError):m.input_packet(x+256)
        with self.assertRaises(ValueError):m.input_packet(x,layers=15)
    def test_byte_tail_zero_and_sign(self):
        self.assertEqual(packed_bytes(np.array([-128,-1,1,127,-3])).tolist(),[0x7f01ff80,253])
if __name__=='__main__':unittest.main()
