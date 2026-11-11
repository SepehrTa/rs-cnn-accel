"""
nn_numpy.py - float and integer ResNet models in plain NumPy.

  fold_bn()      : trained parameters -> per-layer float (W, b), batch norm folded in
  forward_float(): the float reference (used for calibration and accuracy)
  quantize()     : float layers -> int8 layers with fixed-point requantization
  forward_int()  : the bit-exact integer model, the golden reference for the hardware

Integer arithmetic (identical in this model, the mapper's tests and the RTL):
  activations   int8, symmetric (zero point 0); the input image is pixel - 128
  weights       int8, symmetric, one scale per output channel (conv), one per layer (fc)
  conv          acc = sum(x * w) + bias  (int32);  y = requant(acc, M[c], S[c])
  requant       y = (acc * M + 2^(30+S)) >> (31+S)   (M in [2^30, 2^31), one rounding)
  ReLU          clamp to [0, 127], otherwise clamp to [-128, 127]
  add           y = (a * Ma + b * Mb + 2^(30+S)) >> (31+S), then clamp (one rounding)
  gap           int32 sum over the 8 x 8 positions (the 1/64 is folded into fc)
  fc            logits = W @ sum + bias  (int32, all classes on one scale: argmax is exact)
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

BN_OF = {"stem": "stem_bn"}                    # conv name -> batch-norm name
for _b in ("block1", "block2", "block3"):
    BN_OF[f"{_b}.conv1"] = f"{_b}.bn1"
    BN_OF[f"{_b}.conv2"] = f"{_b}.bn2"


# --------------------------------------------------------------------------- basic ops
def conv2d(x, w, stride, pad):
    """x: N x C x H x W, w: O x C x k x k  ->  N x O x Ho x Wo (cross-correlation, like PyTorch).
    Integer inputs are computed in float64, which is exact here (|sums| < 2^53)."""
    n, c, _, _ = x.shape
    o, _, k, _ = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    win = sliding_window_view(xp, (k, k), axis=(2, 3))[:, :, ::stride, ::stride]
    ho, wo = win.shape[2], win.shape[3]
    cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(n * ho * wo, c * k * k)
    out = cols.astype(np.float64) @ w.reshape(o, -1).T.astype(np.float64)
    return out.reshape(n, ho, wo, o).transpose(0, 3, 1, 2)


# --------------------------------------------------------------------------- float model
def fold_bn(params, graph, eps=1e-5):
    """Trained parameters (PyTorch state_dict names) -> {layer: (W, b)} with BN folded in."""
    layers = {}
    for L in graph:
        if L["op"] == "conv":
            w = params[f"{L['name']}.weight"].astype(np.float64)
            b = params.get(f"{L['name']}.bias", np.zeros(w.shape[0])).astype(np.float64)
            bn = BN_OF.get(L["name"])
            if bn is not None:
                g = params[f"{bn}.weight"] / np.sqrt(params[f"{bn}.running_var"] + eps)
                w = w * g[:, None, None, None]
                b = (b - params[f"{bn}.running_mean"]) * g + params[f"{bn}.bias"]
            layers[L["name"]] = (w, b)
        elif L["op"] == "fc":
            layers[L["name"]] = (params["fc.weight"].astype(np.float64),
                                 params["fc.bias"].astype(np.float64))
    return layers


def forward_float(layers, graph, x):
    """x: N x 3 x 32 x 32 in model range ((pixel - 128) / 128). Returns all tensors."""
    t = {"input": x.astype(np.float64)}
    for L in graph:
        if L["op"] == "conv":
            w, b = layers[L["name"]]
            y = conv2d(t[L["src"]], w, L["stride"], L["pad"]) + b[None, :, None, None]
            t[L["dst"]] = np.maximum(y, 0) if L["relu"] else y
        elif L["op"] == "add":
            y = t[L["a"]] + t[L["b"]]
            t[L["dst"]] = np.maximum(y, 0) if L["relu"] else y
        elif L["op"] == "gap":
            t[L["dst"]] = t[L["src"]].mean(axis=(2, 3))
        elif L["op"] == "fc":
            w, b = layers[L["name"]]
            t[L["dst"]] = t[L["src"]] @ w.T + b
    return t


# --------------------------------------------------------------------------- quantization
def quantize_multiplier(m):
    """Real multipliers m > 0 -> (M, S): m ~= M / 2^31 / 2^S, M in [2^30, 2^31)."""
    m = np.atleast_1d(np.asarray(m, np.float64))
    s = np.ceil(-np.log2(m)).astype(np.int64) - 1         # puts m * 2^s into [0.5, 1)
    frac = m * np.power(2.0, s)
    M = np.round(frac * 2.0 ** 31).astype(np.int64)
    over = M == (1 << 31)                                 # rounding reached 1.0
    M[over] //= 2
    s[over] -= 1
    return M, s


def requant(acc, M, S):
    """Rounding fixed-point multiply: (acc * M + 2^(30+S)) >> (31+S), M and S per channel.
    One multiply, one add, one arithmetic shift, a single rounding step (valid for S > -31)."""
    sh = 31 + np.asarray(S, np.int64)
    return (acc.astype(np.int64) * M + np.left_shift(np.int64(1), sh - 1)) >> sh


def calibrate(layers, graph, x, pct=99.99):
    """Activation scales: the given percentile of |value| maps to 127."""
    t = forward_float(layers, graph, x)
    scales = {"input": 1.0 / 128}
    for L in graph:
        if L["op"] in ("conv", "add"):
            v = np.percentile(np.abs(t[L["dst"]]), pct)
            scales[L["dst"]] = max(v, 1e-8) / 127
    return scales


def quantize(layers, graph, scales):
    """Float layers + activation scales -> integer parameters for every layer."""
    q, shapes = {}, {}
    for L in graph:
        n = L["name"]
        if L["op"] == "conv":
            w, b = layers[n]
            s_in, s_out = scales[L["src"]], scales[L["dst"]]
            s_w = np.maximum(np.abs(w).reshape(w.shape[0], -1).max(axis=1), 1e-12) / 127
            wq = np.clip(np.round(w / s_w[:, None, None, None]), -127, 127).astype(np.int8)
            bq = np.round(b / (s_in * s_w)).astype(np.int64)
            M, S = quantize_multiplier(s_in * s_w / s_out)
            q[n] = dict(w=wq, b=bq.astype(np.int32), M=M, S=S, s_w=s_w)
        elif L["op"] == "add":
            ra = scales[L["a"]] / scales[L["dst"]]
            rb = scales[L["b"]] / scales[L["dst"]]
            _, s = quantize_multiplier(max(ra, rb))       # one shared shift
            s = int(s[0])
            Ma = int(np.round(ra * 2.0 ** (31 + s)))
            Mb = int(np.round(rb * 2.0 ** (31 + s)))
            assert Ma < (1 << 31) and Mb < (1 << 31) and 31 + s > 0
            q[n] = dict(Ma=Ma, Mb=Mb, S=s)
        elif L["op"] == "fc":
            w, b = layers[n]
            area = 64.0                                    # 8 x 8 positions summed by gap
            s_w = max(np.abs(w).max(), 1e-12) / 127
            wq = np.clip(np.round(w / s_w), -127, 127).astype(np.int8)
            bq = np.round(b / (s_w * scales[_gap_src(graph)] / area)).astype(np.int32)
            q[n] = dict(w=wq, b=bq, s_w=s_w)
    return q


def _gap_src(graph):
    return next(L["src"] for L in graph if L["op"] == "gap")


# --------------------------------------------------------------------------- integer model
def forward_int(q, graph, x_q):
    """x_q: N x 3 x 32 x 32 int8 (pixel - 128). Returns all integer tensors."""
    t = {"input": x_q.astype(np.int64)}
    for L in graph:
        n = L["name"]
        if L["op"] == "conv":
            p = q[n]
            acc = conv2d(t[L["src"]], p["w"], L["stride"], L["pad"])
            acc = np.rint(acc).astype(np.int64) + p["b"].astype(np.int64)[None, :, None, None]
            y = requant(acc, p["M"][None, :, None, None], p["S"][None, :, None, None])
            t[L["dst"]] = np.clip(y, 0 if L["relu"] else -128, 127)
        elif L["op"] == "add":
            p = q[n]
            s = 31 + p["S"]
            y = (t[L["a"]] * p["Ma"] + t[L["b"]] * p["Mb"] + (1 << (s - 1))) >> s
            t[L["dst"]] = np.clip(y, 0 if L["relu"] else -128, 127)
        elif L["op"] == "gap":
            t[L["dst"]] = t[L["src"]].sum(axis=(2, 3))
        elif L["op"] == "fc":
            p = q[n]
            t[L["dst"]] = t[L["src"]] @ p["w"].astype(np.int64).T + p["b"].astype(np.int64)
    return t
