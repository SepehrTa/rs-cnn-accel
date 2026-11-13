"""Resident ResNet-8 model image and hardware layer-descriptor compiler.

All addresses are explicit: tensor/weight addresses are bytes; descriptor and
numeric-parameter addresses are 32-bit words. Three 16 KiB tensor banks avoid
overwriting residual inputs. Model upload is amortized across all images.
"""
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'model'))
try:
    from .protocol import load_model,check_int8,check_shift,words
except ImportError:
    from protocol import load_model,check_int8,check_shift,words
from graph import resnet8_graph
from nn_numpy import checked_int32

MAGIC=0x52534e32
MODEL_WORDS=24576
BANK_BYTES=16384
ACT_WORDS=3*BANK_BYTES//4
UPLOAD=4
INFER=5


def packed_bytes(values):
    values=np.asarray(values)
    check_int8(values)
    data=np.zeros((values.size+3)//4*4,np.uint8)
    data[:values.size]=values.astype(np.int8).view(np.uint8).ravel()
    b=data.reshape(-1,4).astype(np.uint32)
    return b[:,0]|(b[:,1]<<8)|(b[:,2]<<16)|(b[:,3]<<24)


@dataclass
class ModelImage:
    words: np.ndarray
    descriptors: np.ndarray
    graph: list
    shapes: dict
    offsets: dict
    macs: int

    def upload_packet(self):
        h=np.zeros(16,np.uint32); h[0]=UPLOAD;h[1]=len(self.words)
        return np.concatenate((h,self.words))

    def input_packet(self,x,layers=0):
        x=np.asarray(x)
        if x.shape not in ((1,3,32,32),(3,32,32)):
            raise ValueError('resident engine accepts a CIFAR N=1 CHW image')
        if not np.issubdtype(x.dtype,np.integer): raise ValueError('integer image required')
        check_int8(x)
        if not 0<=layers<=len(self.graph): raise ValueError('invalid layer limit')
        h=np.zeros(16,np.uint32);h[0]=INFER;h[1]=3072;h[2]=layers;h[15]=1
        return np.concatenate((h,packed_bytes(x)))

    def output_count(self,layers=0):
        d=self.descriptors[(layers or len(self.graph))-1]
        return int(d[13]) if d[14] else (int(d[13])+3)//4

    def decode_output(self,raw,layers=0):
        d=self.descriptors[(layers or len(self.graph))-1]
        raw=np.asarray(raw,np.uint32)
        if len(raw)!=self.output_count(layers): raise ValueError('resident output length mismatch')
        if d[14]: value=raw.view(np.int32).copy()
        else:
            b=np.column_stack([(raw>>s)&255 for s in (0,8,16,24)]).astype(np.uint8).ravel()[:int(d[13])]
            value=b.view(np.int8).copy()
        name=self.graph[(layers or len(self.graph))-1]['dst']
        return value.reshape((1,*self.shapes[name]))


def compile_model(path):
    q,graph=load_model(path)
    if graph!=resnet8_graph(): raise ValueError('resident descriptor ABI currently requires the exported ResNet-8 graph')
    descriptors=np.zeros((len(graph),16),np.uint32)
    image=[0]*(16+len(graph)*16)
    shapes={'input':(3,32,32)}; offsets={'input':0}
    consumers={name:sum(name in [l.get(k) for k in ('src','a','b')] for l in graph)
               for name in ['input']+[l['dst'] for l in graph]}
    live={'input'}; macs=0
    def append(values):
        address=len(image);image.extend(int(v) for v in words(values));return address
    for index,layer in enumerate(graph):
        op=layer['op'];name=layer['name'];p=q.get(name)
        d=descriptors[index]
        busy={offsets[n]//BANK_BYTES for n in live}
        free=[b for b in range(3) if b not in busy]
        if not free: raise ValueError(f'{name}: three activation banks are insufficient')
        dst=layer['dst']; offsets[dst]=free[0]*BANK_BYTES
        d[0]={'conv':0,'add':1,'gap':2,'fc':3}[op] | (int(layer.get('relu',False))<<8)
        d[1]=offsets[layer.get('src',layer.get('a'))]
        d[2]=offsets[layer['b']] if op=='add' else 0
        d[3]=offsets[dst]
        if op=='conv':
            c,h,w=shapes[layer['src']];m=layer['cout'];k=layer['k'];u=layer['stride'];pad=layer['pad']
            oh=(h+2*pad-k)//u+1;ow=(w+2*pad-k)//u+1
            if p['w'].shape!=(m,c,k,k): raise ValueError(f'{name}: weight shape mismatch')
            check_shift(31+np.asarray(p['S']))
            if np.any((p['M']<=0)|(p['M']>=2**31)): raise ValueError('invalid multiplier')
            checked_int32(p['b'],'bias')
            d[4:11]=[c,h,w,m,oh,ow,k|(u<<8)|(pad<<16)]
            d[11]=append(packed_bytes(p['w']))*4
            d[12]=append(np.column_stack((p['b'],p['M'],31+p['S'])).ravel())
            shapes[dst]=(m,oh,ow)
            d[15]=c*m*oh*ow*k*k;macs+=int(d[15])
        elif op=='add':
            if shapes[layer['a']]!=shapes[layer['b']]: raise ValueError('residual shapes differ')
            shapes[dst]=shapes[layer['a']]
            check_shift(31+int(p['S']))
            if not (0<=int(p['Ma'])<2**31 and 0<=int(p['Mb'])<2**31): raise ValueError('invalid residual multiplier')
            d[4]=np.prod(shapes[dst]);d[12]=append([p['Ma'],p['Mb'],31+int(p['S'])])
        elif op=='gap':
            c,h,w=shapes[layer['src']];d[4]=c;d[5]=h*w;shapes[dst]=(c,);d[14]=1
        elif op=='fc':
            c=shapes[layer['src']][0];m=len(p['b'])
            if p['w'].shape!=(m,c): raise ValueError('classifier weight shape mismatch')
            checked_int32(p['b'],'classifier bias')
            d[4]=c;d[7]=m;d[11]=append(packed_bytes(p['w']))*4;d[12]=append(p['b'])
            shapes[dst]=(m,);d[14]=1
        count=int(np.prod(shapes[dst]));d[13]=count
        if count*(4 if d[14] else 1)>BANK_BYTES: raise ValueError(f'{name}: output exceeds bank capacity')
        live.add(dst)
        for key in ('src','a','b'):
            source=layer.get(key)
            if source:
                consumers[source]-=1
                if not consumers[source]:live.remove(source)
    image[:10]=[MAGIC,1,len(graph),3072,0,len(image),macs,10,16,BANK_BYTES]
    image[16:16+len(graph)*16]=descriptors.ravel().tolist()
    if len(image)>MODEL_WORDS: raise ValueError('resident model exceeds BRAM allocation')
    return ModelImage(np.asarray(image,np.uint32),descriptors,graph,shapes,offsets,macs)
