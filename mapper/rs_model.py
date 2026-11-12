"""
rs_model.py - the Row Stationary (RS) dataflow of Eyeriss, as an analytical model.

For one convolution layer and one PE-array configuration, a *mapping* says how
the layer is cut into processing passes (Chen et al., JSSC 2017, Sec. IV-A):

    PE set      R x e PEs: filter rows reused horizontally, ifmap rows diagonally,
                psums accumulated vertically (Fig. 4); e = ofmap rows per strip
    p, q        each PE runs p filters x q channels, interleaved in its spads (Fig. 6)
    r, t        r x t PE sets in the array: r channel groups, t filter groups (Fig. 5)

A pass computes e ofmap rows for p*t filters over q*r channels. The layer takes
ceil(E/e) strips x ceil(M/pt) filter groups x ceil(C/qr) channel groups passes.

For every mapping, count() returns the accesses at each level of the memory
hierarchy, the cycles, and the PE utilization; best_mapping() searches all legal
mappings for the lowest weighted access cost, with the paper's relative costs
(normalized to one MAC): DRAM 200, global buffer 6, inter-PE 2, spad 1.

rs_sim.py executes the same mappings on real data and must reproduce these
counts exactly; tests/test_rs.py checks that.
"""
from dataclasses import dataclass, field
from math import ceil


@dataclass(frozen=True)
class HwConfig:
    rows: int = 12                 # PE array height (filter rows of a set)
    cols: int = 14                 # PE array width  (ofmap rows of a set)
    filt_spad: int = 224           # entries (Eyeriss: 224 x 16 b)
    ifmap_spad: int = 12
    psum_spad: int = 24
    glb_bytes: int = 108 * 1024    # global buffer
    act_bytes: int = 1             # int8 activations and weights
    psum_bytes: int = 4            # 32-bit partial sums

    @property
    def pes(self):
        return self.rows * self.cols


COST = dict(dram=200, glb=6, array=2, spad=1, mac=1)


@dataclass(frozen=True)
class Layer:
    name: str
    C: int      # input channels
    M: int      # filters = output channels
    H: int      # input height = width, before padding
    R: int      # filter height = width
    U: int      # stride
    pad: int

    @property
    def Hp(self):                  # padded input size
        return self.H + 2 * self.pad

    @property
    def E(self):                   # output size
        return (self.Hp - self.R) // self.U + 1

    @property
    def macs(self):
        return self.M * self.C * self.E * self.E * self.R * self.R


@dataclass(frozen=True)
class Mapping:
    e: int
    p: int
    q: int
    r: int
    t: int


def layers_from_graph(graph, size=32):
    """Conv layers of a graph (model/graph.py) with their spatial sizes."""
    sizes, out = {"input": size}, []
    for L in graph:
        if L["op"] == "conv":
            h = sizes[L["src"]]
            lay = Layer(L["name"], L["cin"], L["cout"], h, L["k"], L["stride"], L["pad"])
            out.append(lay)
            sizes[L["dst"]] = lay.E
        elif L["op"] == "add":
            sizes[L["dst"]] = sizes[L["a"]]
    return out


def global_buffer_bytes(layer, hw):
    """Resident ifmap + INT32 ofmap assumed by this analytical traffic model."""
    return (layer.C * layer.H * layer.H * hw.act_bytes
            + layer.M * layer.E * layer.E * hw.psum_bytes)


def legal(layer, m, hw):
    S = layer.R
    return (min(m.e, m.p, m.q, m.r, m.t) >= 1
            and global_buffer_bytes(layer, hw) <= hw.glb_bytes
            and layer.R <= hw.rows and 1 <= m.e <= min(layer.E, hw.cols)
            and m.r * m.t <= (hw.rows // layer.R) * (hw.cols // m.e)
            and m.q * S <= hw.ifmap_spad and m.p * m.q * S <= hw.filt_spad and m.p <= hw.psum_spad
            and m.p * m.t <= ceil(layer.M / 1) and m.q * m.r <= layer.C)


def passes(layer, m):
    """Every pass: (first ofmap row, rows, filter list per set column, channel list per set row)."""
    pt, qr = m.p * m.t, m.q * m.r
    for e0 in range(0, layer.E, m.e):
        ea = min(m.e, layer.E - e0)
        for f0 in range(0, layer.M, pt):
            filt = [list(range(f0 + ti * m.p, min(f0 + (ti + 1) * m.p, layer.M))) for ti in range(m.t)]
            for c0 in range(0, layer.C, qr):
                chan = [list(range(c0 + ri * m.q, min(c0 + (ri + 1) * m.q, layer.C))) for ri in range(m.r)]
                yield e0, ea, [f for f in filt if f], [c for c in chan if c], c0


def count(layer, m, hw=HwConfig()):
    """Accesses per memory level, cycles and utilization for one mapping (batch size 1)."""
    if not legal(layer, m, hw):
        raise ValueError("illegal mapping or insufficient resident global-buffer capacity")
    R, S, F, Wp = layer.R, layer.R, layer.E, layer.Hp
    c = dict(mac=0, spad=0, array=0, glb=0, dram=0, cycles=0, passes=0,
             glb_filter=0, glb_ifmap=0, glb_psum=0, noc_filter=0, noc_ifmap=0)
    for e0, ea, filt, chan, c0 in passes(layer, m):
        n_f = sum(len(f) for f in filt)
        n_c = sum(len(ch) for ch in chan)
        rows_in = (ea - 1) * layer.U + R
        macs = n_f * n_c * ea * F * R * S
        c["mac"] += macs
        c["spad"] += 4 * macs                                  # read filter, ifmap, psum; write psum
        c["glb_filter"] += n_f * n_c * R * S                   # each weight read once per pass
        c["glb_ifmap"] += n_c * rows_in * Wp                   # each ifmap value read once (multicast)
        c["glb_psum"] += n_f * ea * F * (2 if c0 > 0 else 1)   # read back + write, or first write
        # inter-PE: vertical psum hops inside each set, then between the r sets of a column
        c["array"] += sum(len(f) for f in filt) * len(chan) * (R - 1) * ea * F
        c["array"] += sum(len(f) for f in filt) * (len(chan) - 1) * ea * F
        # what each PE receives from the global buffer over the NoC
        c["noc_filter"] += n_f * n_c * R * S * ea
        c["noc_ifmap"] += sum(len(ch) for ch in chan) * len(filt) * R * ea * Wp
        # time: every PE runs its p x q primitives of F outputs x S taps, all PEs in parallel
        pmax = max(len(f) for f in filt)
        qmax = max(len(ch) for ch in chan)
        c["cycles"] += pmax * qmax * F * S + R + 2              # + pipeline fill per pass
        c["passes"] += 1
    c["glb"] = c["glb_filter"] + c["glb_ifmap"] + c["glb_psum"]
    # DRAM: ifmap in once and ofmap out once if they fit next to the psums in the global
    # buffer; weights come from DRAM for every strip (they are not kept between strips)
    strips = ceil(layer.E / m.e)
    weights = layer.M * layer.C * R * S
    c["dram_weight"] = weights * strips
    c["dram_ifmap"] = layer.C * layer.H * layer.H
    c["dram_ofmap"] = layer.M * layer.E * layer.E
    c["dram"] = c["dram_weight"] + c["dram_ifmap"] + c["dram_ofmap"]
    c["glb_need"] = global_buffer_bytes(layer, hw)
    c["cost"] = sum(COST[k] * c[k] for k in COST)
    c["util"] = c["mac"] / (c["cycles"] * hw.pes)
    return c


def best_mapping(layer, hw=HwConfig(), objective="cost"):
    """Search every legal mapping; return (mapping, counts) with the lowest objective."""
    if global_buffer_bytes(layer, hw) > hw.glb_bytes:
        raise ValueError("resident tensors exceed analytical model global-buffer capacity")
    best = None
    for e in range(1, min(layer.E, hw.cols) + 1):
        for q in range(1, layer.C + 1):
            for p in range(1, min(layer.M, hw.psum_spad) + 1):
                if q * layer.R > hw.ifmap_spad or p * q * layer.R > hw.filt_spad:
                    continue
                slots = (hw.rows // layer.R) * (hw.cols // e)
                for r in range(1, slots + 1):
                    if q * r > layer.C:
                        break
                    for t in range(1, slots // r + 1):
                        if p * t > layer.M:
                            break
                        m = Mapping(e, p, q, r, t)
                        c = count(layer, m, hw)
                        key = (c[objective], c["cycles"]) if objective == "cost" else (c["cycles"], c["cost"])
                        if best is None or key < best[0]:
                            best = (key, m, c)
    if best is None:
        raise ValueError("no legal mapping for layer and hardware capacities")
    return best[1], best[2]
