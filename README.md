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
   `rs_accel_0` and the passive `rs_ddr_monitor_0`. The HDL top is **`system_wrapper`**.
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

## Resident performance implementation

The board project selects **168 PEs (12 x 14), P=2, Q=2**. The datapath packs
four INT8 values per stream word, initiates one MAC per active PE per compute
clock, overlaps PE loads through ping-pong banks, and shares activation-row
stores across filter groups. A BRAM-backed hardware sequencer executes all
14 ResNet-8 operations; Python uploads the model once and launches one DMA
pair per image using persistent buffers.

The compiled trained model occupies **82,428 bytes** including descriptors and
numeric parameters. Three **16 KiB** tensor regions preserve residual inputs.
Normal inference transfers **3,176 stream bytes** including command headers:
3,136 in and 40 out. Weights and intermediate tensors stay on chip.

**Development checkpoint; full validation is pending.** RTL checks and the
100 MHz routed build pass. One resident inference passed on PYNQ-Z2 with ten
exact logits, 27.15 ms host latency and 3,176 DMA bytes. The board subsequently
became unresponsive during a tiled follow-up test; its cause and the complete
10,000-image run remain unresolved. See [checkpoint evidence and next steps](docs/RESIDENT_CHECKPOINT.md).
See [implementation status](docs/PERFORMANCE_V2.md). Historical validated v1
results are retained in [BOARD_RESULTS.md](docs/BOARD_RESULTS.md), including
four trained images checked on the actual board. They do not establish a
full-set board result for the new implementation.

Read [resident architecture](docs/RESIDENT_ARCHITECTURE.md) and
[resident build, run and counter guide](docs/RESIDENT_USAGE.md). The analytical
Eyeriss mapper uses a different schedule; its convolution-only predictions
are separate from whole-network simulation and board measurements.

A [development checkpoint bundle](deployment/resident-v2-checkpoint.zip) includes
the matching bitstream/HWH, trained model archives and source. Its board
validation limitations are documented in [deployment/README.md](deployment/README.md).

## Run resident inference on PYNQ-Z2

Copy `host/`, `model/`, `scripts/`, the trained `resnet8_int8.npz`, CIFAR-10
`test_batch`, and a matching timing-closed `rs_accel.bit`/`rs_accel.hwh` pair
into one board directory. Generated weights, datasets and deployment binaries
are excluded from the source clone. On the tested PYNQ 3.1.1 image:

```sh
sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 scripts/benchmark_resident.py rs_accel.bit resnet8_int8.npz test_batch --count 10000 --report resident-board.json
```

This programs the FPGA, compares every intermediate tensor for the embedded
validation images, then checks every logit of all 10,000 labelled test images.
Use `--count 100` for a short smoke test. The report records actual completion,
accuracy, cycles, utilization and measured HP0 DMA beats/bytes per MAC. Model
upload/debug traffic is reported separately from normal inference.

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
vivado -mode batch -source scripts/build_board.tcl -tclargs bitstream board-final 14 f4 p2 q2
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
python -m unittest discover -s tests -p test_packed.py
python -m unittest discover -s tests -p test_resident.py
python scripts/run_rtl.py --suite monitor
python scripts/run_resident_rtl.py --archive build/resnet8_int8.npz --image 3
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
| `hw/` | Accelerator SystemVerilog files, Verilog board wrapper and OOC constraints |
| `sim/` | Simulation testbenches; add as Simulation Sources for manual projects |
| `host/` | Tile/resident compilers and persistent PYNQ DMA driver |
| `model/` | Training, quantization/export and NumPy reference |
| `mapper/` | Analytical row-stationary mapper and functional PE-array simulator |
| `scripts/` | Project preparation, builds, validation and packaging |
| `tests/` | Software and archive/packet regression tests |
| `docs/` | Architecture, board results, historical reports and progress |

For manual RTL import, add `hw/*.sv` as Design Sources and `sim/*.sv` as
Simulation Sources. `rs_accel_top` is the accelerator fabric top; a complete
board overlay also needs the PS7/DMA block design. In Vivado 2025.2 its module
reference wrappers `rs_accel_top.sv` and `rs_ddr_monitor.sv` must have File Type **Verilog**; the remaining
datapath files are SystemVerilog. The committed project and preparation
script already apply this setting.

See [architecture and packet ABI](docs/ARCHITECTURE.md),
[actual board results](docs/BOARD_RESULTS.md), and [progress](docs/PROGRESS.md).
The original mapping work follows Chen et al., *Eyeriss* (ISCA 2016/JSSC 2017);
mapper estimates use a different schedule from this RTL.

Sepehr Tabrizchi — [github.com/SepehrTa](https://github.com/SepehrTa)
