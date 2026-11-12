# RS-CNN Accelerator

An Eyeriss-style CNN accelerator for the PYNQ-Z2 (Zynq-7020), built around the
Row Stationary dataflow from Chen et al., *"Eyeriss: An Energy-Efficient
Reconfigurable Accelerator for Deep Convolutional Neural Networks"* (JSSC 2017).
The target network is ResNet-8 on CIFAR-10, the MLPerf Tiny image-classification model,
in int8.

## Open the complete project in Vivado

Open **`rs-cnn-accel.xpr` in this repository's root folder** using Vivado 2025.2.
It contains the PS7, AXI DMA, control/memory interconnects and RS accelerator
block design, with `system_wrapper` as the top module.

On the current PC, the exact file is:

```text
C:\Users\stabr\Documents\ChatGPT\FPGA\rs-cnn-accel\rs-cnn-accel.xpr
```

1. Start Vivado 2025.2 and choose **Open Project**; select that `.xpr` file.
2. Choose **IP Integrator > Open Block Design**, then open `system` to see the
   complete PS7/DMA/accelerator diagram.
3. Choose **Open Implemented Design** to inspect the routed design and timing
   from the completed local build.

The project inputs are committed: `hw/` contains RTL and the top wrapper, and
`rs-cnn-accel.srcs/` contains the block design and IP configuration. Generated
run results are present in the current folder's `rs-cnn-accel.runs/`; they are
excluded from Git. On a fresh source clone, generate output products for
`system.bd`, then run synthesis/implementation to regenerate those results.
For a copy with the existing results included, use the separate
`build/rs-cnn-accel-vivado-complete.zip` Vivado archive.

## Quick start: run on the validated board

The tested deployment is already on the PYNQ-Z2 at `192.168.2.99`, in
`/home/xilinx/rs-cnn-accel-20261112`. From your PC, connect with:

```sh
ssh xilinx@192.168.2.99
```

Then run these commands **on the board**:

```sh
cd /home/xilinx/rs-cnn-accel-20261112

# Run one embedded trained test image (indices 0, 1, 2 or 3).
sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 host/accelerator.py rs_accel.bit resnet8_int8.npz --image 0

# Check all four images three times and save measured inference results.
sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 scripts/benchmark_board.py rs_accel.bit resnet8_int8.npz --repeat 3 --report board-inference-trained.json
```

These commands load the overlay into the FPGA. The first should print
`PASS every tensor; image 0 class 3`, followed by measured time/counters.
The benchmark prints twelve `PASS INFERENCE` lines, then
`PASS BOARD INFERENCE board-inference-trained.json`. Expect about 2.4 seconds
per inference, plus model checking and overlay loading at startup.
The fourth example image is misclassified by the trained model; hardware still
passes because its tensors/logits exactly match the integer reference.

The tested board image requires its PYNQ virtual environment and
`XILINX_XRT=/usr`, as shown. PyTorch and Vivado are not needed on the board.

## Restore on another PC or board

Clone the branch containing the actual-board work:

```sh
git clone --branch codex/resume-validation https://github.com/SepehrTa/rs-cnn-accel.git
cd rs-cnn-accel
```

If you already have this repository, preserve any local edits, then use:

```sh
git fetch origin
git switch codex/resume-validation
git pull --ff-only
```

Git contains source and validation reports. Generated bitstreams, HWH files,
trained NPZ archives, datasets and `build/` are excluded. A source clone alone
does not contain the deployment files.

For immediate deployment, use the separately generated
`rs-cnn-accel-pynq-z2-validated.zip` package. It contains source, reports,
`rs_accel.bit`, `rs_accel.hwh`, both trained NPZ archives and a SHA256 manifest.
On the development PC its path is
`build/rs-cnn-accel-pynq-z2-validated.zip`; it is not included in the Git clone.
Copy this ZIP to the new board, extract it into a dedicated directory and run
the board commands above from that directory. Keep the supplied `.bit` and
`.hwh` together with the same basename.

To regenerate deployment files from source on Windows, install Vivado with
Zynq-7000 support and the Python dependencies in `model/requirements.txt`.
Training uses PyTorch/torchvision; the tested GPU was an RTX 4070. From the
repository root in PowerShell:

```powershell
New-Item -ItemType Directory -Path build -Force | Out-Null
python -m pip install -r model/requirements.txt
python -u model/train.py --data build/data --out build/resnet8_float.npz --workers 0
python -u model/quantize.py --float build/resnet8_float.npz --data build/data --out build/resnet8_int8.npz --vectors 4
python scripts/validate_export.py build/resnet8_int8.npz --float build/resnet8_float.npz --data build/data --report build/export-validation.json

python scripts/fetch_ps_preset.py
& 'C:/AMDDesignTools/2025.2/Vivado/bin/vivado.bat' -mode batch -source scripts/build_board.tcl -log build/board-resume-v2.log -journal build/board-resume-v2.jou -tclargs bitstream board-resume-v2

python scripts/package_release.py --board build/board-resume-v2 --model build/resnet8_int8.npz --float build/resnet8_float.npz
```

Adjust the Vivado path to your installation. The training step downloads
CIFAR-10 when absent and runs 60 epochs. The final command creates and checks
the portable deployment ZIP. For RTL verification with the exported weights:

```powershell
python scripts/run_rtl.py --vivado C:/AMDDesignTools/2025.2/Vivado --suite network --archive build/resnet8_int8.npz --image 0 --output build/sim/trained-image0
```

Repeat with indices 1, 2 and 3 and distinct output directories to check all
exported images. See [actual board results](docs/BOARD_RESULTS.md) for stress
tests, complete measurements and known limits.

## Implemented hardware status

The repository now includes a synthesizable 6x8 RS array, stationary row buffers,
diagonal activation and horizontal weight multicast, registered row-sum links,
INT8 convolution/requantization, residual add, sum pooling and INT32 classifier.
A banked BRAM ingress FIFO, AXI4-Lite control and AXI-stream interface connect
to PS7/AXI DMA in a reproducible PYNQ-Z2 block design. The PYNQ host compiles
tiles and checks every exported intermediate tensor.

XSim executes the full ResNet-8 graph, feeding actual RTL outputs into later
layers. All 114,762 intermediate/final values per image match the integer
reference exactly. The actual PYNQ-Z2 has now been programmed and tested:
720,144 synthetic output comparisons, header-fault recovery, 100 DMA/reset
stress transfers, and normal Python inference all pass. Measured synthetic
host inference is about 2.41 s/image, including tiling/allocation/DMA; core busy
time is about 36.61 ms. These are distinct from analytical throughput estimates.

The deployed Vivado 2025.2 PS7/DMA build meets 100 MHz: 9,013 LUTs, 6,619 FFs,
59 DSPs, four BRAM36-equivalent tiles, setup slack +0.152 ns and hold slack
+0.012 ns. Its matching files are `build/board-resume-v2/rs_accel.bit` and `.hwh`.
See [actual board results](docs/BOARD_RESULTS.md) for executed commands, evidence,
trained-model status and measurement limits. Earlier 2026.1 reports are retained
as historical evidence; they describe a separate build.

The original 60-epoch model has now been trained and exported: **86.55% INT8
accuracy on the 10,000-image CIFAR-10 test set in the NumPy reference**. All four
exported trained images match every tensor in RTL and on the board (three
repetitions each). Trained board inference averages **2.408 s/image**. Full-set
accuracy and the four-image hardware comparison are separate measurements.
The portable source/weights/deployment/evidence package is
`build/rs-cnn-accel-pynq-z2-validated.zip`; see BOARD_RESULTS.md for reproduction.

See [architecture and packet ABI](docs/ARCHITECTURE.md),
[build and board procedure](docs/BOARD_VALIDATION.md),
[live progress](docs/PROGRESS.md), and [validation report](docs/VALIDATION_REPORT.md).
For the restored checkout and validation on the second PC, see
[resume audit](docs/RESUME_VALIDATION.md).
The original 12x14 mapper estimates below describe a different analytical
schedule and must not be used as measured RTL or board throughput.

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

Verification requires NumPy and Vivado/XSim; the runner accepts `--vivado PATH`.
`--cols 4`, `--cols 8` and `--cols 12` evaluate actual parameterized RTL schedules.
Training requirements remain separate in `model/requirements.txt`.

## The model

`model/` trains ResNet-8 in PyTorch and turns it into the integer model the
hardware will run. The integer model in `model/nn_numpy.py` is the golden
reference: plain NumPy, int8 activations and weights, 32-bit accumulation, and
fixed-point requantization with a single rounding step, so the hardware can be
checked against it bit for bit.

```sh
cd model
pip install -r requirements.txt
python train.py            # trains on CIFAR-10 (downloaded to model/data), writes resnet8_float.npz
python quantize.py         # folds batch norm, calibrates, quantizes, writes resnet8_int8.npz
```

`quantize.py` prints the float and integer accuracy on the full test set and
exports the integer parameters, the layer graph, and every intermediate tensor
for a few test images, for checking the hardware layer by layer.

`tests/test_nn_numpy.py` checks the NumPy models without any trained weights:
the convolution against a direct loop implementation, batch-norm folding,
requantization against exact integer arithmetic, and the integer model against
the float model.

## The Row Stationary dataflow, in software first

`mapper/` models the Eyeriss dataflow before any RTL exists. For each layer, a
mapping fixes the PE-set size (R filter rows x e ofmap rows), how many filters
and channels each PE interleaves in its scratch pads (p, q), and how many PE sets
share the array (r channel groups x t filter groups), exactly as in the paper.

- `rs_model.py` counts, for any mapping, the accesses at every level (DRAM,
  global buffer, inter-PE, scratch pads), the cycles and the PE utilization, and
  searches all legal mappings for the lowest weighted cost, using the paper's
  relative access costs (DRAM 200, global buffer 6, inter-PE 2, spad 1).
- `rs_sim.py` runs a layer through the PE array on real data: every PE gets its
  filter row and its diagonal ifmap row, runs the 1-D convolution primitive, and
  passes its psums up the column. `tests/test_rs.py` checks, for every ResNet-8
  layer and several mappings, that the result equals an ordinary convolution
  exactly and that every access count equals the analytical model.
- `report.py` prints the chosen mapping for each layer:

| Layer | C | M | Input | Filter, stride | Mapping (e p q r t) | Cycles | PE utilization |
|---|---:|---:|---:|---|---|---:|---:|
| stem | 3 | 16 | 32x32 | 3x3, 1 | 11 4 3 1 4 | 3,471 | 76% |
| block1.conv1 | 16 | 16 | 32x32 | 3x3, 1 | 11 16 4 4 1 | 18,447 | 76% |
| block1.conv2 | 16 | 16 | 32x32 | 3x3, 1 | 11 16 4 4 1 | 18,447 | 76% |
| block2.conv1 | 16 | 32 | 32x32 | 3x3, 2 | 8 16 4 2 2 | 12,308 | 57% |
| block2.conv2 | 32 | 32 | 16x16 | 3x3, 1 | 8 16 4 4 1 | 24,616 | 57% |
| block2.shortcut | 16 | 32 | 32x32 | 1x1, 2 | 8 8 8 2 4 | 2,054 | 38% |
| block3.conv1 | 32 | 64 | 16x16 | 3x3, 2 | 8 16 4 2 2 | 12,328 | 57% |
| block3.conv2 | 64 | 64 | 8x8 | 3x3, 1 | 8 16 4 4 1 | 24,656 | 57% |
| block3.shortcut | 32 | 64 | 16x16 | 1x1, 2 | 8 16 12 2 4 | 2,566 | 30% |
| **all convolutions** | | | | | | **118,893** | **63%** |

On the paper's 12 x 14 array, the convolutions of one image take about 119,000
cycles, about 840 images per second at 100 MHz, with 0.035 global-buffer and
0.023 DRAM accesses per MAC. Utilization is limited by how the layer shapes fit
the array: 32 ofmap rows don't divide into strips of at most 14, the 16 x 16 and
8 x 8 layers fill at most 8 of the 14 columns, and the 1 x 1 shortcuts use PE
sets only one row tall.

```sh
python mapper/report.py        # takes a few minutes: it tries every legal mapping
python tests/test_rs.py
```

## Repo layout

```
model/     ResNet-8 (PyTorch), training, quantization, float and integer NumPy models
mapper/    the Row Stationary dataflow: mapping search, access counts, PE-array simulator
tests/     tests of the Python models and the dataflow simulator
host/      Python for the PYNQ board
hw/        RTL
sim/       testbenches
docs/      design notes and the plan
```

---

Sepehr Tabrizchi, [github.com/SepehrTa](https://github.com/SepehrTa)
