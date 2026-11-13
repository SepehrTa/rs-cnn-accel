"""Exact exported tensor liveness and resident model payload budget."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'host'))
from protocol import load_model


def audit(archive):
    q,graph=load_model(archive)
    with np.load(archive,allow_pickle=False) as z:
        tensors={k[4:]:z[k][:1] for k in z.files if k.startswith('vec.') and k!='vec.labels'}
    weights=sum(p['w'].size for p in q.values() if 'w' in p)
    params=sum((3*len(p['b']) if 'M' in p else len(p['b']) if 'b' in p else 3)*4 for p in q.values())
    consumers={n:sum(n in [l.get(k) for k in ('src','a','b')] for l in graph) for n in tensors}
    live={'input'}; peak=0; steps=[]
    def size(name):
        return tensors[name].size*(4 if name in ('pool','logits') else 1)
    for layer in graph:
        live.add(layer['dst'])
        occupancy=sum(map(size,live)); peak=max(peak,occupancy)
        steps.append(dict(layer=layer['name'],live=sorted(live),bytes=occupancy))
        for key in ('src','a','b'):
            name=layer.get(key)
            if name:
                consumers[name]-=1
                if not consumers[name]: live.remove(name)
    return dict(weight_bytes=weights,parameter_bytes=params,descriptor_bytes=len(graph)*64,
                peak_live_tensor_bytes=peak,activation_bank_bytes=3*16384,
                model_bram36_capacity_lower_bound=24,activation_bram36_capacity_lower_bound=12,
                vivado2025_2_inferred_model_bram36=32,vivado2025_2_inferred_activation_bram36=16,steps=steps)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('archive',type=Path)
    ap.add_argument('--out',type=Path,default=ROOT/'build'/'resource-audit.json')
    args=ap.parse_args()
    result=audit(args.archive)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='steps'},indent=2))
