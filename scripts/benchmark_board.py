"""Measure real PYNQ inference, including Python tiling and DMA allocation.

Requires owner authorization: constructing Board programs the supplied overlay.
Checks every exported tensor before and after every inference; no hex parsing
or golden-model execution is included in the measured inference interval.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'host'), str(ROOT/'model')]
from protocol import forward, load_vector
from accelerator import Accelerator


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('bitstream',type=Path)
    ap.add_argument('archive',type=Path)
    ap.add_argument('--repeat',type=int,default=3)
    ap.add_argument('--report',type=Path,default=Path('board-inference.json'))
    args = ap.parse_args()
    if args.repeat < 1:
        ap.error('--repeat must be positive')
    start = time.monotonic()
    with np.load(args.archive,allow_pickle=False) as z:
        count = len(z['vec.input'])
        labels = z['vec.labels'].astype(int).tolist()
    if len(labels) != count:
        raise ValueError('embedded labels do not match image count')
    vectors = [load_vector(args.archive,i) for i in range(count)]
    preflight_seconds = time.monotonic()-start
    from pynq_board import Board
    import pynq
    board = Board(str(args.bitstream))
    accel = Accelerator(board.overlay)
    report = dict(platform=platform.platform(),pynq=pynq.__version__,numpy=np.__version__,
                  bitstream_sha256=hashlib.sha256(args.bitstream.read_bytes()).hexdigest(),
                  archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                  preflight_seconds=preflight_seconds,runs=[],status='RUNNING')
    def save():
        args.report.write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        for repeat in range(args.repeat):
            for image,(q,graph,expected) in enumerate(vectors):
                commands = accel.commands
                words = accel.input_words+accel.output_words
                cycles = accel.control.read(12)
                transport_seconds = 0
                def transport(packet,size):
                    nonlocal transport_seconds
                    start = time.monotonic()
                    result = accel(packet,size)
                    transport_seconds += time.monotonic()-start
                    return result
                start = time.monotonic()
                actual = forward(transport,q,graph,expected['input'],cols=accel.cols,filters=accel.filters)
                elapsed = time.monotonic()-start
                for name,value in expected.items():
                    np.testing.assert_array_equal(actual[name],value,err_msg=f'image {image} tensor {name}')
                if accel.control.read(16) != accel.commands:
                    raise AssertionError('completed command counter mismatch')
                predicted = int(actual['logits'].argmax())
                result = dict(repeat=repeat,image=image,label=labels[image],prediction=predicted,
                              logits=actual['logits'].tolist(),exact_values=sum(v.size for k,v in actual.items() if k!='input'),
                              host_seconds=elapsed,transport_seconds=transport_seconds,
                              compile_and_assembly_seconds=elapsed-transport_seconds,
                              busy_cycles=(accel.control.read(12)-cycles)&0xffffffff,
                              commands=accel.commands-commands,
                              axis_payload_bytes=4*(accel.input_words+accel.output_words-words))
                report['runs'].append(result)
                save()
                print('PASS INFERENCE',json.dumps(result),flush=True)
        times = [r['host_seconds'] for r in report['runs']]
        report['mean_host_seconds'] = float(np.mean(times))
        report['mean_images_per_second'] = 1/report['mean_host_seconds']
        report['status'] = 'PASS'
    except BaseException as exc:
        report['status'] = 'FAIL'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save()
    print('PASS BOARD INFERENCE',args.report,flush=True)


if __name__ == '__main__':
    main()
