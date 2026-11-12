"""Execute the RTL runner's exact packet vectors on a real PYNQ overlay.

Run on the board with its PYNQ Python environment and root privileges. Loading
the supplied overlay programs the PL. This script never installs software.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'host'), str(ROOT/'model')]
from accelerator import Accelerator
from protocol import pool_packet


def temperature_c():
    sensor = Path('/sys/bus/iio/devices/iio:device0')
    try:
        raw = float((sensor/'in_temp0_raw').read_text())
        offset = float((sensor/'in_temp0_offset').read_text())
        scale = float((sensor/'in_temp0_scale').read_text())
        return (raw+offset)*scale/1000
    except (OSError, ValueError):
        return None


def check_fault_recovery(accel):
    """Send an invalid header with no receive armed, then reset the sticky fault."""
    from pynq import allocate
    tx = allocate(shape=(16,),dtype=np.uint32)
    tx[:] = 0
    tx[0] = 99
    try:
        accel.dma.sendchannel.transfer(tx)
        deadline = time.monotonic()+accel.timeout
        while not accel.dma.sendchannel.idle:
            if accel.dma.sendchannel.error or time.monotonic() > deadline:
                raise RuntimeError('fault-test header DMA did not complete')
            time.sleep(0.0001)
        accel.dma.sendchannel.wait()
        while not accel.control.read(8) & 2:
            if time.monotonic() > deadline:
                raise RuntimeError('invalid header did not set sticky fault')
        accel.control.write(4,1)
        while accel.control.read(8) & 2:
            if time.monotonic() > deadline:
                raise RuntimeError('soft reset did not clear sticky fault')
    finally:
        if not accel.dma.sendchannel.idle:
            accel.dma.sendchannel.stop()
        tx.freebuffer()


def input_count(header):
    """Number of payload words implied by the version-one packet header."""
    op = int(header[0])
    if op == 0:
        k, stride, pw, rows, filters, channels = (int(header[i]) for i in (1,2,3,5,6,7))
        return channels * (((rows-1)*stride+k)*pw + filters*k*k)
    if op == 1:
        return 2*int(header[1])
    if op == 2:
        return int(header[1])*int(header[2])
    if op == 3:
        return int(header[1])+int(header[2])*(int(header[1])+1)
    raise ValueError(f'unknown opcode {op}')


def execute_vectors(accel, folder):
    """Resolve dependencies from actual board outputs, then compare every word."""
    saved = []
    start_commands = accel.commands
    start_bytes = 4*(accel.input_words+accel.output_words)
    start_cycles = accel.control.read(12)
    start = time.monotonic()
    with (folder/'stim.hex').open() as sf, (folder/'expect.hex').open() as ef:
        stim = (int(line,16) for line in sf if line.strip())
        expected = (int(line,16) for line in ef if line.strip())
        while True:
            first = next(stim, None)
            if first is None:
                break
            header = [first, *itertools.islice(stim,15)]
            if len(header) != 16 or any(v >> 32 for v in header):
                raise ValueError('invalid/truncated packet header')
            count = input_count(header)
            tokens = list(itertools.islice(stim,count))
            if len(tokens) != count:
                raise ValueError('truncated packet payload')
            payload = []
            for token in tokens:
                if token >> 32:
                    addr = token & 0xffffffff
                    if addr >= len(saved):
                        raise ValueError(f'unresolved dependency {addr}')
                    payload.append(saved[addr])
                else:
                    payload.append(token)
            target = []
            for value in expected:
                target.append(value & 0xffffffff)
                if value >> 32:
                    break
            else:
                raise ValueError('missing expected packet TLAST')
            result = accel(np.asarray(header+payload,dtype=np.uint32),len(target))
            actual = result.view(np.uint32)
            mismatch = np.flatnonzero(actual != np.asarray(target,dtype=np.uint32))
            if len(mismatch):
                i = int(mismatch[0])
                raise AssertionError(f'output {len(saved)+i}: got {int(actual[i]):08x}, expected {target[i]:08x}')
            saved.extend(int(v) for v in actual)
        if next(expected,None) is not None:
            raise ValueError('unconsumed expected outputs')
    elapsed = time.monotonic()-start
    manifest = json.loads((folder/'manifest.json').read_text())
    commands = accel.commands-start_commands
    if len(saved) != manifest['outputs'] or commands != manifest['packets']:
        raise AssertionError('manifest count mismatch')
    input_bytes = 4*(accel.input_words+accel.output_words)-start_bytes
    if input_bytes != 4*(manifest['inputs']+manifest['outputs']):
        raise AssertionError('manifest traffic mismatch')
    hardware_commands = accel.control.read(16)
    if hardware_commands != accel.commands:
        raise AssertionError(f'command counter mismatch {hardware_commands}/{accel.commands}')
    return dict(suite=folder.name, vector_source=manifest.get('source',{}),
                exact_values=len(saved), commands=commands,
                host_seconds=elapsed, busy_cycles=(accel.control.read(12)-start_cycles)&0xffffffff,
                axis_payload_bytes=input_bytes,
                dma_mm2s_status=accel.dma.mmio.read(4),
                dma_s2mm_status=accel.dma.mmio.read(0x34))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('bitstream',type=Path)
    ap.add_argument('vectors',type=Path,nargs='+')
    ap.add_argument('--repeat',type=int,default=3)
    ap.add_argument('--stress',type=int,default=100)
    ap.add_argument('--report',type=Path,default=Path('board-results.json'))
    args = ap.parse_args()
    if args.repeat < 1 or args.stress < 0:
        ap.error('repeat must be positive and stress nonnegative')
    for folder in args.vectors:
        for name in ('stim.hex','expect.hex','manifest.json'):
            if not (folder/name).is_file():
                raise FileNotFoundError(folder/name)
    from pynq_board import Board
    import pynq
    board = Board(str(args.bitstream))
    accel = Accelerator(board.overlay)
    report = dict(platform=platform.platform(), python=sys.version,
                  numpy=np.__version__,pynq=pynq.__version__,
                  bitstream_sha256=hashlib.sha256(args.bitstream.read_bytes()).hexdigest(),
                  hwh_sha256=hashlib.sha256(args.bitstream.with_suffix('.hwh').read_bytes()).hexdigest(),
                  temperature_start_c=temperature_c(),
                  runs=[],stress_iterations=0,status='RUNNING')
    def save():
        args.report.write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        check_fault_recovery(accel)
        report['header_fault_reset'] = 'PASS'
        # Repeated reset between independent signed DMA round trips; allocation,
        # DDR/cache, TLAST, count and result checks all use the real hardware.
        rng = np.random.default_rng(29)
        for i in range(args.stress):
            if i % 10 == 0:
                accel = Accelerator(board.overlay)
            x = rng.integers(-128,128,(3,8,8),dtype=np.int32)
            np.testing.assert_array_equal(accel(pool_packet(x),3),x.sum((1,2)))
            if accel.control.read(16) != accel.commands:
                raise AssertionError('stress command counter mismatch')
            report['stress_iterations'] += 1
        for repeat in range(args.repeat):
            # Alternate reset recovery with uninterrupted back-to-back passes.
            if repeat % 2 == 0:
                accel = Accelerator(board.overlay)
            for folder in args.vectors:
                result = execute_vectors(accel,folder)
                result['repeat'] = repeat
                report['runs'].append(result)
                save()
                print('PASS BOARD',json.dumps(result),flush=True)
        report['status'] = 'PASS'
    except BaseException as exc:
        report['status'] = 'FAIL'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['temperature_end_c'] = temperature_c()
        save()
    print('PASS BOARD VALIDATION',args.report,flush=True)


if __name__ == '__main__':
    main()
