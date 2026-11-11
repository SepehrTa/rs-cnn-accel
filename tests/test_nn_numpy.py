"""
test_nn_numpy.py - checks of the NumPy models that need no trained weights or dataset.

    python tests/test_nn_numpy.py

1. conv2d matches a direct, loop-based convolution (stride 1 and 2, 3x3 and 1x1)
2. batch-norm folding gives the same outputs as applying batch norm explicitly
3. requant() matches exact rational arithmetic with round-half-up
4. the integer model tracks the float model on a random network
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from graph import resnet8_graph                                    # noqa: E402
from nn_numpy import (BN_OF, calibrate, conv2d, fold_bn, forward_float,   # noqa: E402
                      forward_int, quantize, quantize_multiplier, requant)

rng = np.random.default_rng(1)
failures = 0


def check(name, ok):
    global failures
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    failures += 0 if ok else 1


# 1) conv2d against loops
def conv_loops(x, w, stride, pad):
    n, c, h, wd = x.shape
    o, _, k, _ = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    ho, wo = (h + 2 * pad - k) // stride + 1, (wd + 2 * pad - k) // stride + 1
    y = np.zeros((n, o, ho, wo))
    for i in range(ho):
        for j in range(wo):
            patch = xp[:, :, i * stride:i * stride + k, j * stride:j * stride + k]
            y[:, :, i, j] = np.einsum("nckl,ockl->no", patch, w)
    return y


for k, stride in [(3, 1), (3, 2), (1, 2)]:
    x = rng.integers(-128, 128, (2, 5, 9, 9)).astype(np.int64)
    w = rng.integers(-127, 128, (4, 5, k, k)).astype(np.int64)
    check(f"conv2d k={k} stride={stride}",
          np.array_equal(conv2d(x, w, stride, k // 2), conv_loops(x, w, stride, k // 2)))

# 2) batch-norm folding
graph = resnet8_graph()
params = {}
for L in graph:
    if L["op"] == "conv":
        params[f"{L['name']}.weight"] = rng.normal(0, 0.3, (L["cout"], L["cin"], L["k"], L["k"]))
        if L["name"] in BN_OF:
            bn = BN_OF[L["name"]]
            params[f"{bn}.weight"] = rng.uniform(0.5, 1.5, L["cout"])
            params[f"{bn}.bias"] = rng.normal(0, 0.1, L["cout"])
            params[f"{bn}.running_mean"] = rng.normal(0, 0.5, L["cout"])
            params[f"{bn}.running_var"] = rng.uniform(0.5, 2.0, L["cout"])
        else:
            params[f"{L['name']}.bias"] = rng.normal(0, 0.1, L["cout"])
params["fc.weight"] = rng.normal(0, 0.3, (10, 64))
params["fc.bias"] = rng.normal(0, 0.1, 10)


def forward_explicit_bn(x):
    """Reference: conv, then batch norm applied as a separate step."""
    t = {"input": x}
    for L in graph:
        if L["op"] == "conv":
            y = conv2d(t[L["src"]], params[f"{L['name']}.weight"], L["stride"], L["pad"])
            if L["name"] in BN_OF:
                bn = BN_OF[L["name"]]
                y = ((y - params[f"{bn}.running_mean"][None, :, None, None])
                     / np.sqrt(params[f"{bn}.running_var"][None, :, None, None] + 1e-5)
                     * params[f"{bn}.weight"][None, :, None, None] + params[f"{bn}.bias"][None, :, None, None])
            else:
                y = y + params[f"{L['name']}.bias"][None, :, None, None]
            t[L["dst"]] = np.maximum(y, 0) if L["relu"] else y
        elif L["op"] == "add":
            y = t[L["a"]] + t[L["b"]]
            t[L["dst"]] = np.maximum(y, 0) if L["relu"] else y
        elif L["op"] == "gap":
            t[L["dst"]] = t[L["src"]].mean(axis=(2, 3))
        elif L["op"] == "fc":
            t[L["dst"]] = t[L["src"]] @ params["fc.weight"].T + params["fc.bias"]
    return t["logits"]


u8 = rng.integers(0, 256, (16, 3, 32, 32))
x_float = (u8 - 128.0) / 128.0
layers = fold_bn(params, graph)
check("batch-norm folding", np.allclose(forward_float(layers, graph, x_float)["logits"],
                                        forward_explicit_bn(x_float), rtol=1e-9, atol=1e-9))

# 3) requant against exact arithmetic
for m in [0.3, 0.5, 0.75, 0.999, 1e-4, 2.5]:
    M, S = quantize_multiplier(m)
    acc = rng.integers(-2 ** 24, 2 ** 24, 20000)
    got = requant(acc, M[0], S[0])
    ok = abs(M[0] / 2.0 ** 31 / 2.0 ** S[0] - m) / m < 1e-8
    # exact: acc * M / 2^(31+S), rounded half up, with Python's unbounded integers
    sh = 31 + int(S[0])
    ref = np.array([(int(a) * int(M[0]) + (1 << (sh - 1))) >> sh for a in acc])
    ok &= np.array_equal(got, ref) and np.max(np.abs(got - acc * m)) <= 0.5 + 1e-6 * np.abs(acc).max()
    check(f"requant m={m} (M={M[0]}, S={S[0]})", ok)

# 4) integer model against float model on the random network
scales = calibrate(layers, graph, x_float)
q = quantize(layers, graph, scales)
lf = forward_float(layers, graph, x_float)["logits"]
li = forward_int(q, graph, (u8 - 128).astype(np.int8))["logits"]
scale = q["fc"]["s_w"] * scales["a3"] / 64
err = np.abs(li * scale - lf).max() / np.abs(lf).max()
check(f"int8 model tracks float (max logit error {100 * err:.1f}% of range)", err < 0.08)

print("\nALL PASS" if failures == 0 else f"\n{failures} FAILED")
