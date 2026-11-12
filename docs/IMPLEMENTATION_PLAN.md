# Implementation plan and initial audit

Date: 2026-11-12. Branch: `codex/rs-rtl-integration`.
Initial user files `AGENTS.md` and `rs-cnn-accel.xpr` are untracked and preserved.

## File-by-file audit

| Existing file | Finding / required work |
|---|---|
| README.md, docs/plan.md | Describe software RS mapping and estimated 12x14 throughput; no hardware evidence. Retain history, distinguish estimates from RTL. |
| model/resnet8.py, graph.py | Nine convolutions (including two projections), three adds, sum GAP, FC; graph agrees with PyTorch topology. |
| model/train.py, cifar10.py | Existing training/data workflow; torch and torchvision absent here. No trained archive or dataset present. Preserve pipeline. |
| model/nn_numpy.py | Numerical contract: symmetric INT8, INT32 bias/accumulator intent, single round toward +infinity at ties; GAP sum with /64 folded into FC bias scale. Lacks overflow/range rejection. |
| model/quantize.py | NPZ graph, scales, weights, intermediates exporter; needs reusable validated loader and synthetic reproducible vectors. |
| mapper/rs_model.py | Analytical 12x14 mapping; GLB capacity reported but not enforced in traffic estimate. Mapping legality missing positivity checks. Not cycle-accurate RTL model. |
| mapper/rs_sim.py | Correct functional RS decomposition; no cycle/port/backpressure implementation. Preserve. |
| mapper/report.py | Prints analytical estimates; must not imply measured board performance. |
| tests/test_nn_numpy.py | Baseline ALL PASS, but failures do not set exit status. |
| tests/test_rs.py | Baseline running; same exit-status defect. Mapping search is expensive. |
| host/pynq_board.py | Working overlay/MMIO helper only. No inference protocol/driver. |
| hw/, sim/ | No existing RTL or testbenches. |
| rs-cnn-accel.xpr | User's untracked empty Vivado project, target part correct. Do not modify. |

## Prioritized vertical slices

1. Preserve baseline; fix test exit codes; document numeric limits and validate archives.
2. Implement 6x8 physical array: two 3-row filter groups, eight output rows.
   PE holds one activation row, three weights, and an output-row psum scratchpad.
   Multicast weights horizontally and input rows diagonally/across filter groups;
   reduce row psums with a clocked neighbor chain. Process channels sequentially.
   Runtime 1x1 uses one row per filter group; unused physical rows clock gated.
3. Integrate bounded packet ingress, controller and AXI-stream output;
   per-filter bias/requant/ReLU, shared-round residual, sum pooling, classifier.
   Host pads and tiles; avoid a full INT32 feature map in FPGA BRAM.
4. Build directed/random tests, stalls/reset checks, full graph simulation with
   actual simulated outputs feeding subsequent layers and per-tensor comparison.
5. Add PYNQ DMA host, AXI-Lite status/control, PS7/HP DMA block design and explicit
   FCLK/reset setup. Run installed XSim and Vivado synthesis/implementation.
6. Record actual cycles/resources/timing, limitations and board procedure.

## Architecture decisions and validation gates

Initial packet design prioritizes bounded storage and correctness over peak
throughput. Activations and weights are INT8 represented in 32-bit AXIS words;
tile streams repeat weights/halo data. This is intentionally less bandwidth
efficient than a future resident-feature-map design; measure actual traffic.
No 119k-cycle analytical claim applies to this RTL schedule.

Reference architecture: Chen et al., [Eyeriss ISCA 2016](https://eems.mit.edu/wp-content/uploads/2016/04/eyeriss_isca_2016.pdf),
primitive row pairs, horizontal weights, diagonal inputs and vertical psums.
This implementation restricts mapping (p=q=r=1, t=2); it does not implement the
paper's general configurable multicast NoC or analytical mapper's full search space.

Require passing software and executable RTL comparisons before timing claims.
Actual trained accuracy, bitstream deployment and board measurements require
the trained archive/data and accessible board; synthetic vectors establish
arithmetic and graph correctness only. Do not download/install/flash automatically.
