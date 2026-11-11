"""
report.py - the best Row Stationary mapping for every ResNet-8 convolution.

    python report.py            # lowest weighted access cost (the paper's objective)
    python report.py cycles     # fewest cycles
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from graph import resnet8_graph                                  # noqa: E402
from rs_model import HwConfig, best_mapping, layers_from_graph   # noqa: E402


def main():
    objective = sys.argv[1] if len(sys.argv) > 1 else "cost"
    hw = HwConfig()
    print(f"PE array {hw.rows} x {hw.cols} = {hw.pes} PEs, spads {hw.filt_spad}/{hw.ifmap_spad}/{hw.psum_spad}, "
          f"objective: {objective}\n")
    head = (f"{'layer':16} {'C':>3} {'M':>3} {'H':>3} {'R':>2} {'U':>2}  {'e p q r t':>14} {'passes':>6} "
            f"{'MACs':>10} {'cycles':>9} {'util':>5} {'GLB/MAC':>8} {'DRAM/MAC':>9} {'cost/MAC':>8}")
    print(head)
    print("-" * len(head))
    tot = dict(mac=0, cycles=0, glb=0, dram=0, cost=0)
    for L in layers_from_graph(resnet8_graph()):
        m, c = best_mapping(L, hw, objective)
        for k in tot:
            tot[k] += c[k]
        print(f"{L.name:16} {L.C:3} {L.M:3} {L.H:3} {L.R:2} {L.U:2}  "
              f"{m.e:3}{m.p:3}{m.q:3}{m.r:3}{m.t:3}  {c['passes']:6} {c['mac']:10,} {c['cycles']:9,} "
              f"{100 * c['util']:4.0f}% {c['glb'] / c['mac']:8.3f} {c['dram'] / c['mac']:9.4f} "
              f"{c['cost'] / c['mac']:8.2f}")
    print("-" * len(head))
    print(f"{'total':16} {'':31} {'':6} {tot['mac']:10,} {tot['cycles']:9,} "
          f"{100 * tot['mac'] / (tot['cycles'] * hw.pes):4.0f}% {tot['glb'] / tot['mac']:8.3f} "
          f"{tot['dram'] / tot['mac']:9.4f} {tot['cost'] / tot['mac']:8.2f}")
    print(f"\nconvolutions only: {tot['cycles']:,} cycles per image -> "
          f"{100e6 / tot['cycles']:,.0f} images/s at 100 MHz")


if __name__ == "__main__":
    main()
