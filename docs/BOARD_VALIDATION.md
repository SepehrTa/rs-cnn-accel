# Reproducible build and board validation

Actual board execution is now recorded in [BOARD_RESULTS.md](BOARD_RESULTS.md),
with logs in `reports/board-2025.2/`. The original 2026.1 build instructions and
historical availability notes below are retained; the deployed 2025.2 build is
`build/board-resume-v2`.

Run from the repository root. All generated files stay under `build/`.
Vivado 2026.1 and XSim were found at `C:/vivado/2026.1/Vivado` on this machine.
No Python packages or system board files were installed during development.

```powershell
python tests/test_nn_numpy.py
python tests/test_rs.py
python -m unittest discover -s tests -p test_protocol.py -v
python scripts/run_rtl.py --suite directed
python scripts/run_rtl.py --suite control
python scripts/run_rtl.py --suite network
# Reuse an actual trained quantize.py export and its first embedded image:
python scripts/run_rtl.py --suite network --archive model/resnet8_int8.npz
# Select another exported image (zero-based; repeat for each embedded image):
python scripts/run_rtl.py --suite network --archive model/resnet8_int8.npz --image 1
python -m unittest discover -s tests -p test_vectors.py -v

& 'C:/vivado/2026.1/Vivado/bin/vivado.bat' -mode batch -source scripts/synth.tcl
python scripts/fetch_ps_preset.py
& 'C:/vivado/2026.1/Vivado/bin/vivado.bat' -mode batch -source scripts/build_board.tcl -tclargs bitstream
# The delivered final build uses a separate generated folder:
& 'C:/vivado/2026.1/Vivado/bin/vivado.bat' -mode batch -source scripts/build_board.tcl -tclargs bitstream board-final
# Optional 72-PE board variant, generated separately:
& 'C:/vivado/2026.1/Vivado/bin/vivado.bat' -mode batch -source scripts/build_board.tcl -tclargs bitstream board-cols12 12
```

The fetcher downloads the official PYNQ v3.1 PS7 preset with a fixed SHA256 into
`build/vendor/`. It changes no global Vivado configuration. Out-of-context
reports in `build/synth` describe accelerator fabric only; board reports in
`build/board` include PS7, DMA and interconnect. They are distinct measurements.
OOC clock-source/skew warnings prevent treating OOC timing as full-board closure.
The top-level COLS generic selects 1..32 columns (6*COLS physical PEs), with
measured simulations at 4/8/12. Default deployment remains eight columns; the
capability register lets the host select a matching tile height automatically.
Only the default eight-column board variant is built/claimed in this report.

The board script validates the block design before generating IP, checks run
completion and setup slack before publishing `rs_accel.bit`/`rs_accel.hwh`.
Do not use a stale copied bitstream after a failed build; check timestamps,
reports and the build log. The existing root XPR is not used or modified.

For a clean board rebuild use a fresh checkout or a new explicitly chosen build
folder. Vivado `create_project -force` replaces its generated project metadata;
no script recursively deletes files. Build prerequisites: Vivado Zynq-7000 IP
support and a working license (BASIC license was available here).

## Training and trained archive

The resumed run completed training, quantization, full-test reference accuracy,
four trained-image RTL comparisons and three board repeats per image; see
BOARD_RESULTS.md. The paragraphs below record the original PC's starting state.

Preserved original workflow: see README and `model/requirements.txt`. PyTorch,
torchvision, trained weights and CIFAR-10 are absent on the development machine.
No trained accuracy is claimed. The network RTL test emits
`build/sim/network/synthetic_int8.npz`, deterministic random quantized parameters
in the existing export's graph/parameter/vector format, with label -1 meaning
synthetic/unlabelled. This fixture tests arithmetic, not classification accuracy.
For a real archive, the RTL runner recomputes and validates every tensor for the
selected image before generating packets, then feeds actual RTL outputs into
subsequent layers. The default is image zero. Negative/out-of-range indices,
missing tensors, invalid tensor shapes, noninteger inputs and INT8 input overflow
are rejected. Nonzero image runs use separate folders such as
`build/sim/network-image1`, or `network-cols4-image1` with `--cols 4`.

On the resumed PC, Vivado is at `C:/AMDDesignTools/2025.2/Vivado`; pass
`--vivado C:/AMDDesignTools/2025.2/Vivado` explicitly to the RTL commands above.
The runner specifies a default `1ns/1ps` timescale for RTL modules; without it,
XSim 2025.2 rejects the mixed timed/untimed module hierarchy at elaboration.
These simulation results do not establish implementation timing in 2025.2.

## On-board procedure

Loading an overlay programs the PL. The owner explicitly authorized deployment
on the PYNQ-Z2 at 192.168.2.99; see BOARD_RESULTS.md for actual executed tests.

1. Use a booted PYNQ-Z2 with a known-good PYNQ image, working DDR and Python PYNQ
   DMA API. Record image/PYNQ version, board revision, temperatures and supply.
2. Check full board timing/DRC reports and review the official PS preset warnings.
   Copy matching `.bit` and `.hwh`, the `host/` and `model/` directories in their
   repository layout, and NPZ onto the board. Archive preflight uses the NumPy
   reference in `model/nn_numpy.py`; training packages are not needed for it.
3. Start with the deterministic synthetic archive, then use a trained export:

   ```sh
   sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 host/accelerator.py rs_accel.bit synthetic_int8.npz --image 0
   sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 host/accelerator.py rs_accel.bit resnet8_int8.npz --image 0
   sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 host/accelerator.py rs_accel.bit resnet8_int8.npz --image 1
   ```

4. Before loading the overlay, the CLI checks the image index and recomputes
   every exported tensor for that image. The host checks ABI, arms S2MM before
   MM2S, submits tiles and checks **every**
   tensor against the embedded export vectors. Require exact logits and argmax.
   Repeat all embedded images. On a DMA/header error, stop channels, reset DMA
   and accelerator (or reload the overlay with authorization) before retrying.
5. Record elapsed host time, core busy cycles, command count, stream bytes and
   DMA status. Measure many warm repetitions with input/model allocation and
   compilation time separately. Host Python tiling/allocation is currently part
   of end-to-end latency; core busy cycles are not board inference latency.
6. Only after exact matches, run the full labelled CIFAR-10 test set using
   `Accelerator.infer`; compare every final logit/label with the NumPy model and
   record measured accuracy/latency/throughput. Check DMA stress, repeated reset,
   back-to-back inference and long-run stability before deployment.

Software/RTL tests and bitstream generation alone do not establish real board
operation. Actual DMA/inference results and measured images/s are separately
recorded in BOARD_RESULTS.md. Power and DDR electrical margins are not measured.
