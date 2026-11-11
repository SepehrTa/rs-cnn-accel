"""
quantize.py - turn the trained float ResNet-8 into the integer model the hardware runs.

    python quantize.py                 # reads resnet8_float.npz and data/, writes resnet8_int8.npz

Steps: fold batch norm, check the float accuracy, calibrate activation scales on
training images, quantize, measure the integer model on the whole test set, and
export the integer parameters plus test vectors (every tensor for a few images)
for checking the hardware layer by layer.
"""
import argparse
import json

import numpy as np

import cifar10
from graph import resnet8_graph
from nn_numpy import calibrate, fold_bn, forward_float, forward_int, quantize


def batched(fn, x, size=1000):
    return np.concatenate([fn(x[i:i + size]) for i in range(0, len(x), size)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--float", default="resnet8_float.npz")
    ap.add_argument("--data", default="data")
    ap.add_argument("--calib", type=int, default=2000, help="training images for calibration")
    ap.add_argument("--out", default="resnet8_int8.npz")
    ap.add_argument("--vectors", type=int, default=4, help="test images to export every tensor for")
    args = ap.parse_args()

    graph = resnet8_graph()
    layers = fold_bn(dict(np.load(args.float)), graph)
    test_x, test_y = cifar10.load(args.data, train=False)
    train_x, _ = cifar10.load(args.data, train=True)

    float_logits = batched(lambda b: forward_float(layers, graph, cifar10.to_float(b))["logits"], test_x)
    float_acc = 100.0 * (float_logits.argmax(1) == test_y).mean()
    print(f"float model (BN folded): {float_acc:.2f}%")

    rng = np.random.default_rng(0)
    calib = train_x[rng.choice(len(train_x), args.calib, replace=False)]
    scales = calibrate(layers, graph, cifar10.to_float(calib))
    q = quantize(layers, graph, scales)

    int_logits = batched(lambda b: forward_int(q, graph, cifar10.to_int8(b))["logits"], test_x)
    int_acc = 100.0 * (int_logits.argmax(1) == test_y).mean()
    agree = 100.0 * (int_logits.argmax(1) == float_logits.argmax(1)).mean()
    print(f"int8 model:              {int_acc:.2f}%   (same prediction as float on {agree:.2f}%)")

    # export: integer parameters, per layer, plus the graph and the scales
    out = {}
    for name, p in q.items():
        for key, val in p.items():
            out[f"{name}.{key}"] = np.asarray(val)
    out["graph"] = np.array(json.dumps(graph))
    out["scales"] = np.array(json.dumps({k: float(v) for k, v in scales.items()}))
    out["test_logits"] = int_logits.astype(np.int32)
    vec = forward_int(q, graph, cifar10.to_int8(test_x[:args.vectors]))
    for name, tensor in vec.items():
        out[f"vec.{name}"] = tensor.astype(np.int32)
    out["vec.labels"] = test_y[:args.vectors]
    np.savez(args.out, **out)
    print(f"saved {args.out}: integer parameters, graph, and every tensor for {args.vectors} test images")


if __name__ == "__main__":
    main()
