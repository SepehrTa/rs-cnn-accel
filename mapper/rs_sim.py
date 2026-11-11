"""
rs_sim.py - execute a convolution layer with the Row Stationary dataflow, PE by PE.

This is a functional model of the PE array, not a cycle-accurate one: it moves
real data the way the hardware will, and counts the accesses as it goes.

    PE (i, j) of a set holds filter row i (p filters x q channels) and receives
    ifmap row (e0 + j) * U + i (q channels): the diagonal of Fig. 4. It runs the
    1-D convolution primitive (Fig. 3) for each of its p filters, accumulating
    over its q channels in the psum spad. Column j then accumulates its R psum
    rows vertically, and the r sets of the same filters add their psums.

The result must equal an ordinary convolution, and the counts must equal
rs_model.count() for the same mapping.
"""
import numpy as np

from rs_model import passes


def conv1d_primitive(ifmap_row, filter_row, out_len, stride):
    """One PE, one filter, one channel: out[x] = sum_s ifmap_row[stride*x + s] * filter_row[s]."""
    S = len(filter_row)
    out = np.zeros(out_len, np.int64)
    for s in range(S):
        out += filter_row[s] * ifmap_row[s: s + stride * (out_len - 1) + 1: stride]
    return out


def run_layer(layer, mapping, x, w):
    """x: C x H x H (int), w: M x C x R x R (int). Returns (psums M x E x E, counts)."""
    R, S, F, U = layer.R, layer.R, layer.E, layer.U
    xp = np.pad(x.astype(np.int64), ((0, 0), (layer.pad, layer.pad), (layer.pad, layer.pad)))
    glb_psum = np.zeros((layer.M, layer.E, F), np.int64)
    c = dict(mac=0, spad=0, array=0, glb_filter=0, glb_ifmap=0, glb_psum=0,
             noc_filter=0, noc_ifmap=0, passes=0)

    for e0, ea, filt, chan, c0 in passes(layer, mapping):
        c["passes"] += 1
        all_f = [f for fl in filt for f in fl]
        all_c = [ch for cl in chan for ch in cl]
        rows_in = (ea - 1) * U + R
        c["glb_filter"] += len(all_f) * len(all_c) * R * S       # filters read for this pass
        c["glb_ifmap"] += len(all_c) * rows_in * xp.shape[2]      # each ifmap row read once, multicast
        pass_psum = np.zeros((len(all_f), ea, F), np.int64)

        for ti, fl in enumerate(filt):                            # set column of the array: filter group
            col_psum = np.zeros((len(fl), ea, F), np.int64)
            for ri, cl in enumerate(chan):                        # set row of the array: channel group
                set_psum = np.zeros((len(fl), ea, F), np.int64)
                for j in range(ea):                               # PE column: one ofmap row
                    column = np.zeros((len(fl), F), np.int64)
                    for i in range(R):                            # PE row: one filter row
                        y = (e0 + j) * U + i                      # the diagonal
                        c["noc_ifmap"] += len(cl) * xp.shape[2]
                        c["noc_filter"] += len(fl) * len(cl) * S
                        pe_psum = np.zeros((len(fl), F), np.int64)   # the psum spad
                        for k, f in enumerate(fl):
                            for ch in cl:
                                pe_psum[k] += conv1d_primitive(xp[ch, y], w[f, ch, i], F, U)
                                c["mac"] += F * S
                                c["spad"] += 4 * F * S
                        if i > 0:
                            c["array"] += len(fl) * F             # psum row passed up one PE
                        column += pe_psum
                    set_psum[:, j] = column
                if ri > 0:
                    c["array"] += len(fl) * ea * F                # psums added across sets
                col_psum += set_psum
            k0 = sum(len(f) for f in filt[:ti])
            pass_psum[k0:k0 + len(fl)] = col_psum

        rows = slice(e0, e0 + ea)
        c["glb_psum"] += len(all_f) * ea * F * (2 if c0 > 0 else 1)
        glb_psum[all_f, rows] += pass_psum                        # read-modify-write in the GLB
    return glb_psum, c
