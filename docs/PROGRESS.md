# Live implementation status

## Actual board deployment (2026-11-12)

- User authorized programming and supplied access to PYNQ-Z2 at 192.168.2.99.
- Newly rebuilt Vivado 2025.2 pair in `build/board-resume-v2` deployed at 100 MHz;
  setup/hold slack +0.152/+0.012 ns; 9,013 LUT, 6,619 FF, 59 DSP, four BRAM tiles.
- Board header-fault reset, 100 DMA/reset stress transfers, three directed
  passes and six full-network synthetic passes all PASS, every tensor exact.
- Normal host inference also PASS; about 2.41 s/image including tiling/DMA,
  ~36.61 ms core-busy time. See BOARD_RESULTS.md and reports/board-2025.2.
- Original 60-epoch training/export completed: PyTorch 86.68%, NumPy float
  86.70%, INT8 86.55% on all 10,000 CIFAR-10 test images.
- All four trained images pass full XSim and three board host-inference passes
  (1,377,144 exact board values); trained board latency averages 2.40833 s/image.
- Six AGENTS.md acceptance categories satisfied; see BOARD_RESULTS.md for
  source changes, commands, evidence and measurement limits. Earlier "no board"
  and "no trained archive" notes below describe the original PC's state.

## Resumed on second PC (2026-11-12)

- Restored `codex/rs-rtl-integration` at `bc6088c` from GitHub; current work is
  on `codex/resume-validation`.
- Python 3.14.7 / NumPy 2.5.4 baseline passes: 11 NumPy checks, 27 mapper checks,
  eight host/protocol tests. Vivado/XSim 2025.2 is installed on this PC.
- Added explicit XSim timescale for 2025.2 elaboration compatibility.
- Added checked archive image selection to RTL and board CLI; board CLI checks
  archive integrity before overlay loading. Seven new regression tests pass.
- Original PE/control/directed/full-network RTL suites pass here. Full network:
  114,762 exact values, 293 packets, 3,802,331 wall cycles, matching prior evidence.
- A distinct second synthetic image also passes the complete RTL graph with
  all 114,762 values exact using the new archive `--image 1` path.
- See RESUME_VALIDATION.md for changed files, commands and remaining work.
- Trained NPZ files, datasets, `.bit`/`.hwh` files are excluded from the Git
  backup. This checkout has source and retained reports, not those artifacts.

The sections below record the original PC's implementation and evidence.

## Completed / evidence

- Audited all existing source files; created dedicated feature branch.
- Preserved untracked user AGENTS.md and Vivado XPR.
- `python tests/test_nn_numpy.py`: 11 PASS, ALL PASS (Python 3.13.14, NumPy 2.4.3).
- Integer graph/rounding contract established; see IMPLEMENTATION_PLAN.md.

## Implemented and tested

- Existing RS baseline: 27 PASS, all raw tensors exact and analytical counts match.
- Added bounded 48-PE RTL, local row/psum stores, multicast, reduction,
  requant/ReLU, residual, pooling and classifier.
- Added two-bank BRAM ingress FIFO and AXI-Lite reset/status with independent AW/W.
- XSim directed and full graph tests pass; actual outputs feed later layers.
- Added runtime reuse counters/checks in testbench and signed bias/shift edge cases.
- Replaced the initial shared reduction arithmetic with direct registered
  row-neighbor links and adders inside every PE. Final numeric tests pass.
- Added protocol compiler and PYNQ DMA inference driver, independent scalar host
  interpreter tests, overflow checks and synthetic NPZ in original export format.
- 8 new Python tests pass; original NumPy suite still 11 PASS.
- Analytical mapper now rejects insufficient resident GLB capacities; it no
  longer silently reports one-load DRAM traffic for an infeasible buffer.
- Added OOC and board Vivado scripts. Actual block-design validation succeeded.
- Pipeline optimization driven by real synthesis reports, not timing estimates.

## Final implementation evidence

- Final direct-neighbor board build completed in `build/board-final`: matched
  `.bit`/`.hwh`, 100 MHz, WNS +0.261 ns, WHS +0.024 ns, no timing failures.
- Full PS design: 9,014 LUT, 6,615 FF, 59 DSP, 2 RAMB36 + 4 RAMB18.
- Final OOC route in `build/synth`: 6,452 LUT, 3,052 FF, 59 DSP, 2 RAMB18;
  WNS +0.161 ns, WHS +0.144 ns. Earlier shared-reduction builds are superseded.
- All source/artifact hashes, actual logs and reports captured in `docs/reports`.
- `docs/VALIDATION_REPORT.md` records commands, exact results, fixed failures,
  retained warnings and performance limitations.
- Measured XSim 24/48/72-PE graph cycles: 4,512,027 / 3,802,331 / 3,736,486.
  Keep 48 PEs: 72 PEs improve this schedule by only ~1.7%.

## Original-PC remaining TODO (superseded by board results above)

- [ ] Obtain trained archive/CIFAR data; run trained-vector RTL comparison and
  labelled dataset accuracy. Existing training/export entry points preserved.
- [ ] With owner approval and an accessible board, program the final matching
  pair and execute BOARD_VALIDATION.md, including DMA/DDR/reset stress checks.
- [ ] Review preset DDR-generation warnings on the actual board/image. No
  warning suppression or invented board validation was used.
- [ ] Measure actual board cycles, host latency, DDR traffic, throughput/power.
- [ ] Optimize packed streams, resident tiles and output overlap before scaling
  PE count. General Eyeriss mapping/NoC remains outside this restricted schedule.

## Known external limitations

- torch/torchvision/pytest not installed; use NumPy + stdlib unittest for new tests.
- No trained NPZ/CIFAR dataset supplied. No training/accuracy claim.
- No PYNQ connection supplied; no board programming or measurement performed.
