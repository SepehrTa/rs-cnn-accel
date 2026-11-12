# Live implementation status

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

## Remaining TODO

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
