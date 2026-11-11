"""cifar10.py - read CIFAR-10 in the format torchvision downloads (data/cifar-10-batches-py)."""
import os
import pickle
import numpy as np


def _batch(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    return d[b"data"].reshape(-1, 3, 32, 32), np.array(d[b"labels"], np.int64)


def load(root="data", train=False):
    """uint8 images N x 3 x 32 x 32 and int64 labels."""
    base = os.path.join(root, "cifar-10-batches-py")
    files = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]
    parts = [_batch(os.path.join(base, f)) for f in files]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def to_int8(images_u8):
    """The integer model's input: pixel - 128."""
    return (images_u8.astype(np.int16) - 128).astype(np.int8)


def to_float(images_u8):
    """The float model's input, the same values divided by 128."""
    return (images_u8.astype(np.float64) - 128.0) / 128.0
