"""Pure NumPy packet compiler shared by simulation and the PYNQ DMA driver.

Packets contain a 16-word header followed by payload, all little-endian uint32.
Input may be wider than INT8 for FC only. Overflow is rejected before submission.
"""
import json
from pathlib import Path
import sys

import numpy as np


def words(values):
    return np.asarray(values, dtype=np.int64).astype(np.uint32)


def int32_bound(x, w, bias):
    """Conservative absolute bound; covers every intermediate sum as well."""
    peak = int(np.abs(np.asarray(x, dtype=np.int64)).max(initial=0))
    span = np.abs(np.asarray(w, dtype=np.int64)).reshape(len(w), -1).sum(1) * peak
    b = np.asarray(bias, dtype=np.int64)
    if np.any(span > np.iinfo(np.int32).max) or np.any(b-span < -(2**31)) or np.any(b+span > 2**31-1):
        raise ValueError("possible signed INT32 accumulator overflow")


def check_shift(shift):
    if np.any((np.asarray(shift) < 1) | (np.asarray(shift) > 62)):
        raise ValueError("hardware rounding shift must be in 1..62")


def check_int8(x):
    if np.any((x < -128) | (x > 127)):
        raise ValueError("activation/weight is outside signed INT8")


def conv_packets(x, p, layer, cols=8, filters=2, max_out=32):
    """Yield (packet, (filter slice, row slice, col slice), output shape).

    x is C,H,W; tiles have two filters, <=8 rows and <=32 output columns.
    Payload interleaves each channel's patch and filters' stationary weight rows.
    """
    x = np.asarray(x, dtype=np.int64)
    w = np.asarray(p['w'], dtype=np.int64)
    check_int8(x)
    check_int8(w)
    if x.ndim != 3 or w.ndim != 4 or w.shape[1] != x.shape[0]:
        raise ValueError("invalid convolution tensor shapes")
    k, stride, pad = layer['k'], layer['stride'], layer['pad']
    if k not in (1, 3) or stride not in (1, 2) or w.shape[2:] != (k, k):
        raise ValueError("unsupported kernel/stride")
    if not 1 <= cols <= 32 or not 1 <= filters <= 16 or not 1 <= max_out <= 32:
        raise ValueError("tile configuration exceeds default RTL capacities")
    int32_bound(x, w, p['b'])
    shifts = 31 + np.asarray(p['S'], dtype=np.int64)
    check_shift(shifts)
    if np.any((p['M'] <= 0) | (p['M'] >= 2**31)):
        raise ValueError("invalid convolution multiplier")
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    oh, ow = ((v-k)//stride+1 for v in xp.shape[1:])
    if min(oh, ow) <= 0:
        raise ValueError("empty output")
    # Local activation storage limits padded width, even when stride is two.
    tile_width = min(max_out, (34-k)//stride+1)
    for f in range(0, len(w), filters):
        nf = min(filters, len(w)-f)
        for y in range(0, oh, cols):
            nr = min(cols, oh-y)
            ph = (nr-1)*stride+k
            for ox in range(0, ow, tile_width):
                nw = min(tile_width, ow-ox)
                pw = (nw-1)*stride+k
                h = np.zeros(16, dtype=np.int64)
                h[:9] = [0, k, stride, pw, nw, nr, nf, x.shape[0], int(layer['relu'])]
                for g in range(min(nf,2)):
                    h[9+3*g:12+3*g] = [p['b'][f+g], p['M'][f+g], shifts[f+g]]
                payload = [h]
                if nf>2:
                    payload.append(np.column_stack((p['b'][f+2:f+nf],p['M'][f+2:f+nf],shifts[f+2:f+nf])).ravel())
                for c in range(x.shape[0]):
                    payload.append(xp[c, y*stride:y*stride+ph, ox*stride:ox*stride+pw].ravel())
                    payload.append(w[f:f+nf, c].ravel())
                yield words(np.concatenate(payload)), (slice(f, f+nf), slice(y, y+nr), slice(ox, ox+nw)), (len(w), oh, ow)


def add_packet(a, b, p, relu):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not a.size:
        raise ValueError("residual shapes must match and be nonempty")
    check_int8(a)
    check_int8(b)
    shift = 31+int(p['S'])
    check_shift(shift)
    if not (0 <= int(p['Ma']) < 2**31 and 0 <= int(p['Mb']) < 2**31):
        raise ValueError("invalid residual multiplier")
    h = np.zeros(16, np.int64)
    h[0], h[1], h[8], h[10], h[11], h[13] = 1, a.size, int(relu), p['Ma'], shift, p['Mb']
    return words(np.concatenate((h, np.stack((a.ravel(), b.ravel()), axis=1).ravel())))


def pool_packet(x):
    x = np.asarray(x)
    check_int8(x)
    if x.ndim != 3 or not x.size or x.shape[1]*x.shape[2] > 65535:
        raise ValueError("invalid sum-pool shape")
    h = np.zeros(16, np.int64)
    h[:3] = [2, x.shape[1]*x.shape[2], x.shape[0]]
    return words(np.concatenate((h, x.ravel())))


def fc_packet(x, p):
    x = np.asarray(x, dtype=np.int64).ravel()
    if np.any((x < -(2**31)) | (x > 2**31-1)):
        raise ValueError("classifier input must fit signed INT32")
    w, b = np.asarray(p['w']), np.asarray(p['b'])
    check_int8(w)
    if w.ndim != 2 or w.shape[1] != len(x) or not 1 <= len(x) <= 256:
        raise ValueError("invalid classifier shape")
    int32_bound(x, w, b)
    h = np.zeros(16, np.int64)
    h[:3] = [3, len(x), len(w)]
    return words(np.concatenate((h, x, np.column_stack((b, w)).ravel())))


def load_model(path):
    """Read the existing quantize.py NPZ contract without pickle."""
    with np.load(Path(path), allow_pickle=False) as z:
        graph = json.loads(str(z['graph']))
        q = {}
        for layer in graph:
            name = layer['name']
            keys = {'conv': ('w', 'b', 'M', 'S'), 'add': ('Ma', 'Mb', 'S'),
                    'fc': ('w', 'b'), 'gap': ()}.get(layer['op'])
            if keys is None:
                raise ValueError(f"unknown operation {layer['op']}")
            if keys:
                q[name] = {k: z[f'{name}.{k}'].copy() for k in keys}
        return q, graph


def load_vector(path, image=0):
    """Load one exported image and check every tensor before hardware access."""
    # The CLI also runs directly as ``python host/accelerator.py``.
    model_dir = str(Path(__file__).resolve().parents[1] / 'model')
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    from nn_numpy import forward_int

    q, graph = load_model(path)
    with np.load(Path(path), allow_pickle=False) as z:
        inputs = z['vec.input']
        if inputs.ndim != 4 or inputs.shape[0] == 0:
            raise ValueError('vec.input must be a nonempty NCHW batch')
        if not 0 <= image < len(inputs):
            raise ValueError(f'image index {image} outside 0..{len(inputs)-1}')
        x = inputs[image:image+1]
        if not np.issubdtype(x.dtype, np.integer):
            raise ValueError('vec.input must contain integer activations')
        check_int8(x)
        tensors = forward_int(q, graph, x.astype(np.int64))
        for name, value in tensors.items():
            key = f'vec.{name}'
            if key not in z:
                raise ValueError(f'archive is missing {key}')
            saved = z[key]
            if saved.shape != (len(inputs), *value.shape[1:]):
                raise ValueError(f'{key}: invalid exported tensor shape {saved.shape}')
            expected = saved[image:image+1]
            if not np.array_equal(value, expected):
                where = tuple(int(i) for i in np.argwhere(value != expected)[0])
                raise ValueError(f'{key}: image {image} differs from golden recomputation at {where}')
    return q, graph, tensors


def forward(transport, q, graph, x, cols=8, filters=2):
    """Execute one N=1 image. transport(packet, output_words) -> signed INT32."""
    x = np.asarray(x)
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError("driver accepts one NCHW image at a time")
    t = {'input': x.astype(np.int64)}
    for layer in graph:
        op, dst = layer['op'], layer['dst']
        p = q.get(layer['name'])
        if op == 'conv':
            out = None
            for packet, region, shape in conv_packets(t[layer['src']][0], p, layer, cols=cols,filters=filters):
                if out is None:
                    out = np.empty(shape, np.int64)
                tile_shape = tuple(s.stop-s.start for s in region)
                out[region] = np.asarray(transport(packet, int(np.prod(tile_shape)))).reshape(tile_shape)
            t[dst] = out[None]
        elif op == 'add':
            a, b = t[layer['a']], t[layer['b']]
            t[dst] = np.asarray(transport(add_packet(a, b, p, layer['relu']), a.size)).reshape(a.shape)
        elif op == 'gap':
            a = t[layer['src']][0]
            t[dst] = np.asarray(transport(pool_packet(a), a.shape[0]))[None]
        elif op == 'fc':
            t[dst] = np.asarray(transport(fc_packet(t[layer['src']], p), len(p['w'])))[None]
        else:
            raise ValueError(f"unknown operation {op}")
    return t
