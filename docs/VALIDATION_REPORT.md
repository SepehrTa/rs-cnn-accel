# Validation and performance report

Date: 2026-11-12. Branch: `codex/rs-rtl-integration`.
This report distinguishes executable software/RTL results, Vivado implementation,
and board measurements. The project has a working simulated inference path and
a timing-closed PYNQ-Z2 bitstream; actual deployment validation and trained
accuracy are still open.

## Delivered implementation

- A default six-row/eight-column, 48-PE RS array. Each PE retains a filter row,
  activation row and local psum row; weights multicast horizontally, activations
  multicast diagonally/across groups, and adjacent rows have direct registered
  PE-to-PE psum links. See ARCHITECTURE.md for the actual schedule and limitations.
- Stride-1/2 3x3/1x1 convolutions, arbitrary channel passes and output tile tails,
  signed INT8 products/INT32 sums, per-filter bias/requant/ReLU, shared-round
  scale-mismatched residuals, global sum pool and INT32 classifier.
- Two-bank synchronous BRAM ingress, backpressure-safe AXIS output, AXI-Lite
  reset/status/capacities, PS7 GP0 control and HP0 DDR through AXI DMA.
- A packet compiler, validated NPZ loader and PYNQ DMA host with timeouts,
  receive-before-send ordering, standard PYNQ completion/cache handling, actual
  receive-length checks, and stop-before-free on errors. DMA API checked against
  the [official PYNQ v3.1 source](https://github.com/Xilinx/PYNQ/blob/v3.1/pynq/lib/dma.py);
  software mocks exercise ordering/cleanup, but board behavior remains untested.
- Reference overflow/shift checks, failing test exit codes, and analytical mapper
  GLB-capacity rejection. Existing training/quantization/export workflows preserved.

## Commands and observed results

Commands run at the repository root; executable paths below are actual installed
tools. Full stdout and tool reports are collected in `docs/reports/` after build
completion by `python scripts/capture_evidence.py`.

| Command | Result |
|---|---|
| `python tests/test_nn_numpy.py` | 11 PASS; ALL PASS; exit 0 |
| `python tests/test_rs.py` | 27 PASS; all convolution tensors exact, all access counts match; exit 0 |
| `python -m unittest discover -s tests -p test_protocol.py -v` | 8 tests, OK; exit 0 |
| `python scripts/run_rtl.py --suite pe` | 1,632 signed MAC checks, 544 neighbor reductions; PASS |
| `python scripts/run_rtl.py --suite control` | Independent AW/W, response stalls, decode errors, fault, active abort and reset recovery; PASS |
| `python scripts/run_rtl.py --suite directed` | 94 commands, 10,524 exact values; PASS |
| `python scripts/run_rtl.py --suite network` | Complete graph, 293 commands, 114,762 exact values; PASS |
| `python scripts/run_rtl.py --suite network --archive build/sim/network/synthetic_int8.npz` | Existing NPZ loader/export contract and every embedded tensor independently recomputed; all RTL values exact; PASS |
| `python scripts/run_rtl.py --suite network --cols 4` | 24 PEs, all 114,762 values exact; PASS |
| `python scripts/run_rtl.py --suite network --cols 12` | 72 PEs, all 114,762 values exact; PASS |
| `python -m compileall -q host model mapper scripts tests` | exit 0 |
| `git diff --check` | exit 0 |

The runner invokes the installed `xvlog.bat --sv`, `xelab.bat` and `xsim.bat
-runall` and treats either tool failures or simulation `$fatal` as failures.
An explicit Verilog parameter wrapper avoids Windows batch argument splitting.
The PE test includes -128 * -128 and repeated scratchpad reloads. Directed tests
include rectangular/large-width/tail tiles, one/three kernels, both strides,
padding 0/1/2, negative values, saturation, bias extremes, shifts 1..62, negative
ties, scale-mismatched residuals with/without ReLU, signed pooling and classifier.
Tests randomize input bubbles and output backpressure and assert stalled AXIS
data/TLAST stability. Reset memories are not assumed initialized.

## Full network comparison

The synthetic archive uses seed 8, a 32x32x3 image, the unmodified ResNet-8 graph,
nine convolutions, three adds, sum pool and 64-to-10 classifier. Parameters are
synthetic and deliberately exercise signed arithmetic; label -1 denotes no
classification ground truth. No training or CIFAR accuracy is inferred.

The stimulus refers to previously emitted **actual RTL outputs** when feeding
later layers. It does not substitute golden intermediate inputs after the stem.
Every output is checked in tile order; each convolution raw result also passes
the directed/scalar numerical checks. The final golden/RTL logits are:

```text
[2802, -23499, -14789, -1685, 2527, 282, 7802, -3357, -20564, 16431]
argmax = 9
```

Zero mismatches across all 114,762 values. Tensor names, shapes and output ranges
are recorded in `reports/network-manifest.json`. Runtime counters verify exactly
12,500,992 PE MACs, 5,841,920 local activation writes and 849,408 local weight
writes. This is observable RS reuse rather than an unverified array label.

## Simulation performance (not board measurements)

All configurations use identical data, host tile layout and deterministic stalls.
Cycles include loading, compute, reduction, postprocessing and stream stalls.
The six-stage postprocessor is intentionally serialized per output in this first
version. Input buffering overlaps some stream traffic with computation.

| Physical array | Commands | Input words | Output words | Wall cycles | Simulated time at 100 MHz |
|---|---:|---:|---:|---:|---:|
| 6x4 (24 PEs) | 581 | 2,262,426 | 114,762 | 4,512,027 | 45.12027 ms |
| 6x8 (48 PEs) | 293 | 1,994,586 | 114,762 | 3,802,331 | 38.02331 ms |
| 6x12 (72 PEs) | 269 | 1,970,122 | 114,762 | 3,736,486 | 37.36486 ms |

The 48-PE run has 3,802,033 core-busy cycles, 8,437,392 bytes of **AXIS payload**
traffic (headers, repeated INT8 scalars in 32-bit words, and outputs), and ~6.85%
of nominal one-MAC/PE/clock utilization. Each PE currently issues one MAC every
two clocks, so that nominal peak is intentionally not attainable by this schedule.
Translating the trace at 100 MHz gives ~26.3 inferences/s **in simulation**, not a
measured PYNQ throughput. Host allocation, Python tiling, DMA launch/cache costs,
CPU copies, DDR burst overhead and real contention are not included in that rate.
Actual DDR traffic and board latency remain unmeasured.

Adding 50% more PEs improves this trace by only ~1.7%; keep 6x8 as the default.
Prioritize packed stream payloads, resident input/weight tiles, overlapping PE
loads, and pipelined multi-output reduction/requantization before increasing PE
count. No comparable prior output-stationary RTL exists in this repository, so
no measured RS-versus-OS speedup is claimed. The README's original 12x14 mapper
results use a different schedule/bandwidth model and are not this RTL's cycles.

## Vivado and bitstream status

Vivado/XSim 2026.1, SW build 6511674; device `xc7z020clg400-1`; valid BASIC license.
Commands:

```powershell
& 'C:/vivado/2026.1/Vivado/bin/vivado.bat' -mode batch -source scripts/synth.tcl
python scripts/fetch_ps_preset.py
& 'C:/vivado/2026.1/Vivado/bin/vivado.bat' -mode batch -source scripts/build_board.tcl -log build/board-final.log -journal build/board-final.jou -tclargs bitstream board-final
```

Both final direct-neighbor builds completed successfully:

| Actual routed design | LUTs | FFs | DSP | RAMB36 | RAMB18 | Setup WNS | Hold WHS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Accelerator OOC | 6,452 | 3,052 | 59 | 0 | 2 | +0.161 ns | +0.144 ns |
| PS7 + DMA + interconnect + accelerator | 9,014 | 6,615 | 59 | 2 | 4 | +0.261 ns | +0.024 ns |

Both target 100 MHz. Full board use is 16.94% of LUTs, 6.22% of FFs, 26.82% of
DSPs and four BRAM36-equivalent tiles (2.86%). No failing setup/hold endpoints;
no unclocked or unconstrained internal endpoints. OOC lacks real clock-tree/pin
context; the complete PS design establishes full implementation timing. Raw
[board timing](reports/board-timing.rpt), [utilization](reports/board-utilization.rpt)
and [DRC](reports/board-drc.rpt) reports are included.

Vivado printed `Bitgen Completed Successfully`; the final command exited 0.
Delivered generated artifacts (not committed as binaries):

- `build/board-final/rs_accel.bit`, 4,045,769 bytes,
  SHA256 `7888c9c1adfb5def6f4d357f537d7be382f246973726dcd99b44af9bc66afce9`.
- `build/board-final/rs_accel.hwh`, 309,166 bytes,
  SHA256 `b79ed6a4286af2ce034a493b0d96eaaf2fe93ae2aed0c0c78758bf45f0f7d041`.

Source and artifact hashes are in `reports/sha256.json`. Use this matching pair;
superseded `build/board` outputs do not represent the final direct-neighbor RTL.
The full final build log remains `build/board-final.log`; an exact warning/run
excerpt is in `reports/board-build-excerpt.txt`.

Warnings are retained, not suppressed. Final DRC lists 289 checks, all Warning
or Advisory, with zero Error/Critical Warning checks: DSP pipelining/opmode
recommendations, inferred IP LUT terms, vendor FIFO async-control/write-first
advisories and unused internal FIFO nets. Block-design generation also reports
AXI ID-width adaptations and a DMA memory/stream width propagation warning.
The official PS preset emits **critical generation warnings** for DDR
DQS-to-clock delay indices 2/3 (-0.009/-0.033). These are distinct from final
implementation DRC severity. They remain part of board validation; no real DDR
operation or full PS/DMA bus functional simulation has been performed here.

## Failures found and corrected

- Existing scripts could print FAIL while exiting zero: fixed process status.
- FIFO introduced legal queued dependent commands: testbench initially treated
  a not-yet-emitted reference as an error. It now waits for the actual output.
- Initial standalone place tried to bind internal AXI ports to package pins:
  corrected accelerator builds to OOC and built the actual PS peripheral wrapper.
- Initial combinational requantization missed 100 MHz (routed WNS -9.487 ns).
  Added smaller registered multipliers and separate combine/round/shift stages,
  PE sampling/accumulation stages, and classifier stages.
- Header validation/patch sizing inferred generic multipliers despite only two
  legal strides: replaced them with bounded shift/add expressions.
- Windows batch split `=` in simulator/build arguments: use Verilog parameter
  wrappers and positional board-folder/column arguments. One obsolete generated
  rebuild was interrupted; final build runs in a separate fresh folder.
- Mapper previously assumed resident tensors even for a too-small configured GLB:
  now rejects infeasible capacity instead of understating DRAM traffic.

## Changed files and remaining work

| Files | Purpose |
|---|---|
| `hw/rs_pe.sv`, `rs_stream_core.sv`, `rs_postprocess.sv` | RS local datapath, direct neighbor links, scheduler and all graph operations |
| `hw/rs_axis_buffer.sv`, `rs_accel_top.sv`, `fclk.xdc` | Banked memory, AXIS/AXI-Lite wrapper, OOC constraints |
| `sim/tb_pe.sv`, `tb_control.sv`, `tb_stream.sv` | Self-checking PE, bus/reset, reuse and full graph tests |
| `host/protocol.py`, `accelerator.py` | Archive loading, range checks, tiling, inference and PYNQ DMA |
| `model/nn_numpy.py` | Reject invalid shifts/multipliers and INT32 bias/accumulator overflow |
| `mapper/rs_model.py` | Mapping positivity and resident GLB-capacity checks |
| `tests/test_nn_numpy.py`, `test_rs.py`, `test_protocol.py` | Meaningful failure exits and independent host/scalar/DMA-mock checks |
| `scripts/run_rtl.py`, `synth.tcl`, `build_board.tcl` | Reproducible XSim/OOC/PS7-DMA builds |
| `scripts/fetch_ps_preset.py`, `probe_vivado.tcl`, `capture_evidence.py` | Pinned local preset, tool probe, reproducible evidence |
| `requirements-test.txt`, `.gitignore`, `.gitattributes`, `README.md` | Verification environment, generated-artifact exclusions, verbatim report formatting and usage |
| `docs/plan.md`, `IMPLEMENTATION_PLAN.md`, `PROGRESS.md`, `ARCHITECTURE.md`, `BOARD_VALIDATION.md`, `VALIDATION_REPORT.md`, `reports/*` | Audit, decisions, actual results, protocol and remaining deployment work |

Preserved the user's untracked `AGENTS.md` and root XPR. No packages installed,
no system configuration changed, no user files deleted, no board programmed,
and nothing pushed remotely. Generated Vivado/vector files remain under build
(older Vivado diagnostics `NA/` and `clockInfo.txt` are ignored and preserved).

Remaining external validation: trained archive/CIFAR test set, compatible PYNQ
image and accessible PYNQ-Z2. Board execution requires authorization before
programming; follow BOARD_VALIDATION.md. Existing torch/torchvision dependencies
are absent and were not installed. Performance tuning and a general Eyeriss NoC
are future work; current protocol/schedule is deliberately restricted and is not
the paper's full mapping search. No power or actual board-throughput claim.
