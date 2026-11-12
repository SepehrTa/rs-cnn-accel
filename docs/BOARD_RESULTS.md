# Actual PYNQ-Z2 validation

Date: 2026-11-12. Board: `pynq`, `192.168.2.99`, ARMv7 Zynq-7000,
PYNQ 3.1.1, NumPy 1.21.5, Linux 6.6.10-xilinx-v2024.1.
The owner explicitly requested actual board deployment and supplied SSH access.
The PL was programmed with the newly built matched bitstream/HWH pair.
No other project's files were overwritten.

## Build actually deployed

Vivado 2025.2, `xc7z020clg400-1`, six rows/eight columns (48 PEs), 100 MHz.
Generated artifacts: `build/board-resume-v2/rs_accel.bit` and `rs_accel.hwh`.
Setup WNS **+0.152 ns**, hold WHS **+0.012 ns**, zero failing endpoints.
Full PS7/DMA design: **9,013 LUTs, 6,619 registers, 59 DSPs, four BRAM tiles**.
Bitstream generation and final build command exited successfully.
Raw timing/utilization/DRC and build warnings are in `reports/board-2025.2/`.

Vivado 2025.2 does not accept a SystemVerilog file as a block-design module
reference top. `build_board.tcl` marks the existing Verilog-2001-compatible
`rs_accel_top.sv` wrapper as Verilog; datapath children remain SystemVerilog.
This fixed the failed first build without replacing any functional RTL.

The vendor preset retains negative DQS delay warnings for byte lanes 2/3.
The same preset specifies a **16-bit** DDR interface (lanes 0/1); the preset was
not changed or warnings suppressed. Actual DMA testing exercises the running
board's DDR/cache/HP0 path. It is not a DDR signal-integrity margin sweep, and
loading an overlay does not re-run boot-time PS DDR initialization.

## Synthetic hardware tests completed

- Invalid command header sets the sticky fault; software reset clears it.
- 100 deterministic signed sum-pool DMA round trips pass, with reset every ten
  transfers, exact arithmetic and hardware completed-command counter checks.
- Three directed passes: 94 packets and 10,524 exact values per pass.
- Three passes of each of two distinct full-network images: 293 packets and
  **114,762 exact intermediate/final values per image**; actual hardware outputs
  feed all dependent layers. Alternate passes run without resetting the core.
- Every transfer checks receive length, and every suite checks manifest traffic
  and hardware command counts. Both DMA status registers end at `0x1002`
  (idle/complete, no error bits).
- XADC temperature rose from about 43 to 44 degrees C during this test.

Evidence: `reports/board-2025.2/synthetic-board.json` and `.txt`.
The packet-vector replay takes about 22 seconds per network image because it
parses text stimulus and resolves reference tokens on the ARM. Use the actual
host API measurements below for inference latency.

## Normal host inference measurements

`scripts/benchmark_board.py` runs the existing `host.protocol.forward` compiler
and actual DMA driver, with both synthetic images repeated twice. All tensors
match; mean latency is about **2.41 s/image (~0.415 images/s)**. Measured core
busy time is approximately **3,661,000 cycles (~36.61 ms at 100 MHz)**.
Each inference performs 293 commands and transports **8,437,392 AXIS bytes**.

About 1.63–1.70 s is spent in transport (allocation, cache maintenance, DMA and
polling); about 0.74–0.76 s is Python packet compilation/tensor assembly.
Golden recomputation, overlay loading and model loading are measured/executed
outside the inference interval. These are real board measurements, not the
earlier simulation's 26.3 images/s estimate. Busy cycles differ slightly from
XSim because actual DMA backpressure differs from the deterministic testbench.
Evidence: `reports/board-2025.2/synthetic-inference.json`.

## Repeatable commands

On the development PC, from repository root:

```powershell
python scripts/fetch_ps_preset.py
& 'C:/AMDDesignTools/2025.2/Vivado/bin/vivado.bat' -mode batch -source scripts/build_board.tcl -log build/board-resume-v2.log -journal build/board-resume-v2.jou -tclargs bitstream board-resume-v2
```

Copy the matching pair, `host/`, `model/`, `scripts/`, and required vectors into
a dedicated board directory. The executed deployment is in
`/home/xilinx/rs-cnn-accel-20261112`. No password belongs in source or reports.
From that directory on the board:

```sh
sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 -u scripts/validate_board.py rs_accel.bit build/sim/directed build/sim/network build/sim/network-image1 --repeat 3 --stress 100 --report board-synthetic.json
sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 -u scripts/benchmark_board.py rs_accel.bit synthetic_int8.npz --repeat 2 --report board-inference-synthetic.json
sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 -u host/accelerator.py rs_accel.bit synthetic_int8.npz --image 1
```

An initial launch without `XILINX_XRT=/usr` failed with `No Devices Found`.
The corrected command uses this board image's existing XRT and PYNQ virtual
environment; no system configuration was changed.

## Trained-model validation

The original 60-epoch training run completed on the RTX 4070. Final PyTorch
test accuracy is **86.68%**; the folded float64 NumPy model measures **86.70%**.
The INT8 reference achieves **86.55% (8,655/10,000 test images)**, with 98.80%
prediction agreement with the folded float reference. PyTorch versus NumPy
float logits for the exported images differ by at most **2.75984e-6**.

All four exported trained images pass full-network XSim independently, each
with **114,762 exact values and 293 packets**. The same trained NPZ was copied
to the board and run through the normal host compiler/DMA path three times:
**12 successful inferences, 1,377,144 exact intermediate/final comparisons**.
Mean measured host inference is **2.40833 s/image (0.41523 images/s)**; mean core
busy count is **3,661,019 cycles (36.61019 ms at 100 MHz)**.

The four labels are `[3, 8, 8, 0]`; reference/RTL/board predictions are
`[3, 8, 8, 8]`. Image 3 is a model classification error, not a hardware
arithmetic mismatch. The full-set 86.55% accuracy was evaluated in the NumPy
reference; the board ran these four distinct labelled images repeatedly,
not the entire 10,000-image test set.

Raw training, quantization, export parity, trained XSim and board inference
results are in `reports/board-2025.2/`. Each trained simulation manifest records
the model archive SHA256; the board report independently records the same hash.

Executed training/export workflow on the RTX 4070 development PC:

```powershell
python -u model/train.py --data build/data --out build/resnet8_float.npz --workers 0
python -u model/quantize.py --float build/resnet8_float.npz --data build/data --out build/resnet8_int8.npz --vectors 4
python scripts/validate_export.py build/resnet8_int8.npz --float build/resnet8_float.npz --data build/data --report build/export-validation.json
# Repeat for image indices 0, 1, 2 and 3, with matching separate output folders:
python scripts/run_rtl.py --vivado C:/AMDDesignTools/2025.2/Vivado --suite network --archive build/resnet8_int8.npz --image 0 --output build/sim/trained-image0
```

Executed on the board after copying the trained archive:

```sh
sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 -u scripts/benchmark_board.py rs_accel.bit resnet8_int8.npz --repeat 3 --report board-inference-trained.json
```

The original Toronto dataset host was slow; the downloaded unmodified archive
from the [MIT course mirror](https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python)
matched torchvision's expected MD5 `c58f30108f718f92721af3b95e74349a` before
training. The expected checksum is defined in the
[official torchvision loader](https://github.com/pytorch/vision/blob/main/torchvision/datasets/cifar.py).
No dataset variants, alternate network topology or shortened training schedule
were substituted. Training used the repository's default 60 epochs and seed 0.

## Changes, acceptance and delivery

| Files changed from the restored branch | Purpose / validation |
|---|---|
| `host/protocol.py`, `host/accelerator.py` | Checked exported-image selection and golden preflight before board programming; unit tests and actual board CLI/API pass |
| `scripts/run_rtl.py` | Selected image, separate build output, model hash provenance, XSim timescale; all original and four trained RTL runs pass |
| `scripts/build_board.tcl` | Verilog module-reference compatibility; routed/timing-closed 2025.2 board build actually deployed |
| `scripts/validate_board.py` | Actual dependent-vector execution, DMA/reset/fault stress, command/traffic checking, temperatures and JSON evidence |
| `scripts/benchmark_board.py` | Normal host inference with every-tensor comparison and separated transport/compilation latency; synthetic and trained board runs pass |
| `scripts/validate_export.py` | All embedded vectors, full labelled export accuracy and independent PyTorch/NumPy float parity; PASS |
| `scripts/package_release.py` | Portable archive with source, trained weights, paired bitstream/HWH, evidence and per-file SHA256; CRC/hash verification |
| `tests/test_vectors.py` | Seven image selection/integrity regressions, including independent scalar packet execution |
| `README.md`, `docs/BOARD_VALIDATION.md`, `PROGRESS.md`, `VALIDATION_REPORT.md`, `RESUME_VALIDATION.md`, this report | Current versus historical results, reproducible commands and limitations |
| `docs/reports/resume-2025.2/*`, `docs/reports/board-2025.2/*` | Retained real tool/test results, manifests, training/accuracy and hardware measurements |

No functional RTL, numerical graph, training algorithm, mapper, AGENTS.md or
root Vivado XPR was replaced. The original NumPy checks (11), mapper checks
(27), protocol tests (8) and new vector tests (7) pass. Final raw fast-test logs
are retained under `reports/board-2025.2/`; compileall and whitespace checks pass.
This completes the six AGENTS.md acceptance categories: reproducible export,
software tests, executable RTL integration, bit-exact end-to-end vectors,
actual Vivado implementation evidence, and host/deployment instructions with
separate board results and limitations.

Portable delivery: `build/rs-cnn-accel-pynq-z2-validated.zip`. It includes the
trained float/INT8 archives and matched `.bit`/`.hwh` at its root, source,
documentation, evidence and `MANIFEST.json`. These generated binaries remain
excluded from normal Git source tracking. The CIFAR-10 dataset is not bundled.
After extracting on the board, run the trained benchmark command above.

Recreate and verify the package with:

```powershell
python scripts/package_release.py --board build/board-resume-v2 --model build/resnet8_int8.npz --float build/resnet8_float.npz
```

Future optimization includes buffer reuse, packed INT8 streams, resident tiles
and overlapped output scheduling. Those performance extensions are not claimed
by the current restricted row-stationary schedule. Full-dataset board endurance,
power instrumentation and DDR electrical characterization remain separate
measurements; they do not have invented pass results.

## Measurement limits

AXIS byte counts include headers, repeated patch/weight scalars and results;
they are not DDR-controller bus counters. Power was not measured: this board's
XADC exposes temperature/voltage, not board supply current. An external power
instrument is required for a power measurement. Synthetic correctness does not
establish trained classification accuracy.
