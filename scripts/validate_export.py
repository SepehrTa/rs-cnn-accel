"""Check trained export tensors, labelled accuracy and optional PyTorch parity."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'host'), str(ROOT/'model')]
from protocol import load_vector
import cifar10
from nn_numpy import fold_bn, forward_float


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('archive',type=Path)
    ap.add_argument('--float',dest='float_archive',type=Path)
    ap.add_argument('--data',type=Path)
    ap.add_argument('--report',type=Path,default=Path('build/export-validation.json'))
    args = ap.parse_args()
    with np.load(args.archive,allow_pickle=False) as z:
        count = len(z['vec.input'])
        labels = z['vec.labels'].astype(int)
        inputs = z['vec.input'].copy()
        logits = z['test_logits'].copy()
    vectors = [load_vector(args.archive,i) for i in range(count)]
    if not count or labels.shape != (count,):
        raise ValueError('export must contain images and matching labels')
    embedded = np.concatenate([t['logits'] for _,_,t in vectors])
    np.testing.assert_array_equal(embedded,logits[:count])
    report = dict(archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                  embedded_images=count,exact_values=sum(v.size for _,_,t in vectors for v in t.values()),
                  labels=labels.tolist(),predictions=embedded.argmax(1).tolist(),
                  embedded_logits=embedded.tolist(),status='PASS')
    if args.data:
        images,truth = cifar10.load(args.data,train=False)
        if logits.shape != (len(truth),10):
            raise ValueError('test_logits must contain the complete CIFAR-10 test set')
        np.testing.assert_array_equal(inputs,cifar10.to_int8(images[:count]))
        np.testing.assert_array_equal(labels,truth[:count])
        report['test_images'] = len(truth)
        report['integer_correct'] = int((logits.argmax(1)==truth).sum())
        report['integer_accuracy_percent'] = 100*report['integer_correct']/len(truth)
    if args.float_archive:
        import torch
        from resnet8 import ResNet8
        with np.load(args.float_archive,allow_pickle=False) as z:
            params = {k:z[k].copy() for k in z.files}
        model = ResNet8().eval()
        model.load_state_dict({k:torch.from_numpy(v) for k,v in params.items()})
        x = inputs.astype(np.float64)/128
        graph = vectors[0][1]
        reference = forward_float(fold_bn(params,graph),graph,x)['logits']
        with torch.no_grad():
            actual = model(torch.tensor(x,dtype=torch.float32)).numpy()
        np.testing.assert_allclose(actual,reference,rtol=1e-4,atol=1e-4)
        report['pytorch_numpy_max_abs_logit_error'] = float(np.max(np.abs(actual-reference)))
        report['float_archive_sha256'] = hashlib.sha256(args.float_archive.read_bytes()).hexdigest()
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
