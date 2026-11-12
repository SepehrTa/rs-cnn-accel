"""
test_rs.py - the Row Stationary simulator against an ordinary convolution, and
the analytical model against the simulator.

    python tests/test_rs.py

For every ResNet-8 convolution, a few mappings (the best one plus random legal
ones): rs_sim.run_layer must give exactly the convolution's partial sums, and
every access count must equal rs_model.count() for the same mapping.
"""
import os
import random
import sys

import numpy as np

here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(here, "..", "model"))
sys.path.insert(0, os.path.join(here, "..", "mapper"))
from graph import resnet8_graph                                          # noqa: E402
from nn_numpy import conv2d                                              # noqa: E402
from rs_model import HwConfig, Mapping, best_mapping, count, layers_from_graph, legal   # noqa: E402
from rs_sim import run_layer                                             # noqa: E402

KEYS = ["mac", "spad", "array", "glb_filter", "glb_ifmap", "glb_psum", "noc_filter", "noc_ifmap", "passes"]


def random_mappings(layer, hw, n, rng):
    out = []
    while len(out) < n:
        m = Mapping(e=rng.randint(1, min(layer.E, hw.cols)), p=rng.randint(1, min(layer.M, 24)),
                    q=rng.randint(1, max(1, min(layer.C, hw.ifmap_spad // layer.R))),
                    r=rng.randint(1, 4), t=rng.randint(1, 4))
        if legal(layer, m, hw):
            out.append(m)
    return out


def main():
    hw = HwConfig()
    rng = random.Random(0)
    nrng = np.random.default_rng(0)
    failures = 0
    for L in layers_from_graph(resnet8_graph()):
        x = nrng.integers(-128, 128, (L.C, L.H, L.H))
        w = nrng.integers(-127, 128, (L.M, L.C, L.R, L.R))
        ref = conv2d(x[None], w, L.U, L.pad)[0]
        best, _ = best_mapping(L, hw)
        for m in [best] + random_mappings(L, hw, 2, rng):
            out, sim = run_layer(L, m, x, w)
            model = count(L, m, hw)
            ok_data = np.array_equal(out, ref)
            bad = [k for k in KEYS if sim[k] != model[k]]
            ok = ok_data and not bad
            failures += 0 if ok else 1
            print(f"{'PASS' if ok else 'FAIL'}  {L.name:16} e={m.e:2} p={m.p:2} q={m.q:2} r={m.r} t={m.t}"
                  f"  data {'exact' if ok_data else 'WRONG'}"
                  + (f", counts differ: {bad}" if bad else ", counts match"))
    print("\nALL PASS" if failures == 0 else f"\n{failures} FAILED")
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
