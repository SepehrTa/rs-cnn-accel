# RS-CNN Accelerator for PYNQ-Z2

A synthesizable row-stationary INT8 ResNet-8 accelerator for CIFAR-10 on the
PYNQ-Z2 (Zynq-7020). This repository contains the complete Vivado project,
SystemVerilog RTL, PS7/DMA block design, Python model and board driver, tests,
and retained implementation/board-validation evidence.

## Clone and open in Vivado

**Requirements:** Vivado **2025.2** with **Zynq-7000 device support** installed.
The target part is `xc7z020clg400-1`. Python, PyTorch, a connected board, and
separately installed PYNQ board-definition files are not required to open the
project. The PS7 configuration is already stored in the block design.

```sh
git clone https://github.com/SepehrTa/rs-cnn-accel.git
cd rs-cnn-accel
```

1. Start Vivado 2025.2, choose **Open Project**, and select
   **`rs-cnn-accel.xpr`** in the cloned repository's root folder.
2. In **IP Integrator**, choose **Open Block Design** and open **`system`**.
   The diagram contains PS7, AXI DMA, control/memory interconnects, reset logic,
   and `rs_accel_0`. The HDL top is **`system_wrapper`**.
3. For a fresh clone, choose **Tools > Run Tcl Script** and select
   **`scripts/prepare_vivado.tcl`**. This refreshes the RTL module reference,
   validates the block design, generates the vendor IP output products, and
   enables post-route physical optimization for the 100 MHz implementation.
4. Choose **Generate Bitstream**. Vivado runs synthesis and implementation.
   After completion, use **Open Implemented Design** to inspect placement,
   routing and timing.

Check that setup and hold slack are nonnegative before deploying a newly built
bitstream; successful bitstream generation alone does not establish timing closure.

Alternatively, when Vivado is on your PATH, this opens and prepares the project:

```sh
vivado -mode gui -source scripts/prepare_vivado.tcl
```

Keep the cloned directory structure intact. The root `.xpr`, `hw/` RTL and
wrapper, and `rs-cnn-accel.srcs/` block-design/IP inputs are all committed.
Generated caches, IP products and run results are excluded from Git and are
recreated locally. A fresh clone therefore has no completed implementation to
open until it has been built. Use 2025.2 for the tested flow; migration to other
Vivado releases has not been validated.

## Design and verified results

The default array has **48 PEs (6 x 8)**. Each PE retains activation/weight rows
and local partial sums; the design uses horizontal weight multicast, diagonal
activation reuse, and registered neighbor partial-sum links. It supports the
exported network's 3x3/1x1 convolutions, stride 1/2, bias/requantization/ReLU,
residual addition, sum pooling and INT32 classifier. AXI-Lite supplies control;
AXI DMA connects the 32-bit streams to PS DDR through HP0.

The retained deployed-build measurements are listed below. A fresh
implementation can have different placement, utilization and timing; see
[fresh-clone validation](docs/CLONE_VALIDATION.md) for the publication check.

| Measurement | Observed result |
|---|---|
| Device / fabric clock | PYNQ-Z2 `xc7z020clg400-1`, 100 MHz |
| Vivado 2025.2 routed setup / hold slack | +0.152 ns / +0.012 ns |
| Complete PS7/DMA design resources | 9,013 LUTs; 6,619 FFs; 59 DSPs; four BRAM36-equivalent tiles |
| INT8 NumPy reference accuracy | 86.55% on 10,000 CIFAR-10 test images |
| Trained RTL comparison | Four images; all 114,762 intermediate/final values per image exact |
| Actual board trained comparison | Four distinct images, three repetitions each; 1,377,144 exact values |
| Actual board host inference | Mean 2.408 s/image, including Python tiling/allocation/DMA |
| Actual board core-busy time | About 36.61 ms/image at 100 MHz |
| Board robustness checks | Header-fault recovery, 100 signed DMA/reset transfers, and repeated synthetic graph tests passed |

The full 10,000-image accuracy evaluation ran in the NumPy reference, not on
the board. One of the four exported examples is misclassified by the model;
its RTL/board tensors still exactly match the reference. Power and DDR
signal-integrity margins were not measured. This is a restricted row-stationary
schedule; analytical Eyeriss mapper throughput is not measured board throughput.
See [actual board results](docs/BOARD_RESULTS.md) for commands, logs and limits.

## Run on a PYNQ-Z2

Opening/building the Vivado project does not require training. To run inference,
you also need a quantized model archive and matching `.bit`/`.hwh` pair. Trained
weights, datasets and generated deployment binaries are not part of the Git
clone; regenerate them using the commands below or use a separately supplied
validated deployment package.

Copy `host/`, `model/`, `scripts/`, `resnet8_int8.npz`, `rs_accel.bit` and
`rs_accel.hwh` into the same project directory on a booted PYNQ-Z2. The bitstream
and HWH must come from the same build and have the same basename. On the tested
PYNQ 3.1.1 image, run from that directory:

```sh
sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 host/accelerator.py rs_accel.bit resnet8_int8.npz --image 0
sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 scripts/benchmark_board.py rs_accel.bit resnet8_int8.npz --repeat 3 --report board-inference-trained.json
```

These commands program the FPGA. The first checks every tensor of one exported
image; the benchmark checks every embedded image repeatedly and writes measured
latency/counter results to JSON. Adjust the PYNQ Python environment path for
your board image. No board login credentials or fixed IP address are required
by the source code.

## Regenerate the trained model and deployment pair

Run from the repository root. Training requires NumPy, PyTorch and torchvision;
choose a PyTorch installation appropriate to your CPU/GPU. The original run
used 60 epochs and an RTX 4070.

```sh
python -m pip install -r model/requirements.txt
python model/train.py --data build/data --out build/resnet8_float.npz --workers 0
python model/quantize.py --float build/resnet8_float.npz --data build/data --out build/resnet8_int8.npz --vectors 4
python scripts/validate_export.py build/resnet8_int8.npz --float build/resnet8_float.npz --data build/data --report build/export-validation.json
```

The training workflow downloads CIFAR-10 when absent. Quantization folds batch
normalization, calibrates scales, evaluates the full test set, and exports
integer parameters, the graph, labels and every tensor for four test images.
The golden arithmetic is defined in `model/nn_numpy.py`.

To build a deployment pair with the batch flow instead of the GUI:

```sh
python scripts/fetch_ps_preset.py
vivado -mode batch -source scripts/build_board.tcl -tclargs bitstream board-final
```

The hash-pinned preset is downloaded into `build/vendor/`; no system board
store is changed. The script checks run completion and setup/hold timing, then
publishes `build/board-final/rs_accel.bit` and `rs_accel.hwh`. With the GUI flow,
the corresponding generated files are under `rs-cnn-accel.runs/impl_1/` and
`rs-cnn-accel.gen/sources_1/bd/system/hw_handoff/`; copy/rename the matching pair
for deployment. See [build and board procedure](docs/BOARD_VALIDATION.md).

## Verification

NumPy is sufficient for the software tests. XSim is supplied with Vivado;
pass `--vivado /path/to/Vivado/2025.2` to the RTL runner if needed.

```sh
python tests/test_nn_numpy.py
python tests/test_rs.py
python -m unittest discover -s tests -p test_protocol.py -v
python -m unittest discover -s tests -p test_vectors.py -v
python scripts/run_rtl.py --suite pe
python scripts/run_rtl.py --suite control
python scripts/run_rtl.py --suite directed
python scripts/run_rtl.py --suite network
```

For trained vectors, use `--suite network --archive build/resnet8_int8.npz
--image 0 --output build/sim/trained-image0`; repeat with indices 1, 2 and 3
and separate output folders. Actual earlier runs and source/artifact hashes
are retained in `docs/reports/`.

## Repository layout

| Path | Contents |
|---|---|
| `rs-cnn-accel.xpr` | Open this complete project in Vivado 2025.2 |
| `rs-cnn-accel.srcs/` | Committed PS7/DMA block design and vendor IP configuration |
| `hw/` | Five accelerator SystemVerilog files, Verilog board wrapper and OOC constraints |
| `sim/` | Simulation testbenches; add as Simulation Sources for manual projects |
| `host/` | Packet compiler and PYNQ overlay/DMA driver |
| `model/` | Training, quantization/export and NumPy reference |
| `mapper/` | Analytical row-stationary mapper and functional PE-array simulator |
| `scripts/` | Project preparation, builds, validation and packaging |
| `tests/` | Software and archive/packet regression tests |
| `docs/` | Architecture, board results, historical reports and progress |

For manual RTL import, add `hw/*.sv` as Design Sources and `sim/*.sv` as
Simulation Sources. `rs_accel_top` is the accelerator fabric top; a complete
board overlay also needs the PS7/DMA block design. In Vivado 2025.2 its module
reference wrapper `rs_accel_top.sv` must have File Type **Verilog**; the remaining
four accelerator files are SystemVerilog. The committed project and preparation
script already apply this setting.

See [architecture and packet ABI](docs/ARCHITECTURE.md),
[actual board results](docs/BOARD_RESULTS.md), and [progress](docs/PROGRESS.md).
The original mapping work follows Chen et al., *Eyeriss* (ISCA 2016/JSSC 2017);
mapper estimates use a different schedule from this RTL.

Sepehr Tabrizchi — [github.com/SepehrTa](https://github.com/SepehrTa)
