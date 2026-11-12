# Reproducible build and board validation

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

Preserved original workflow: see README and `model/requirements.txt`. PyTorch,
torchvision, trained weights and CIFAR-10 are absent on the development machine.
No trained accuracy is claimed. The network RTL test emits
`build/sim/network/synthetic_int8.npz`, deterministic random quantized parameters
in the existing export's graph/parameter/vector format, with label -1 meaning
synthetic/unlabelled. This fixture tests arithmetic, not classification accuracy.
For a real archive, the RTL runner first recomputes and validates every embedded
first-image tensor and then feeds actual RTL outputs into subsequent layers.

## On-board procedure (not executed here)

Loading an overlay programs the PL. Obtain the owner's approval before performing
this step. No accessible board address or active PYNQ session was provided.

1. Use a booted PYNQ-Z2 with a known-good PYNQ image, working DDR and Python PYNQ
   DMA API. Record image/PYNQ version, board revision, temperatures and supply.
2. Check full board timing/DRC reports and review the official PS preset warnings.
   Copy matching `.bit` and `.hwh`, the host directory and NPZ onto the board.
3. Start with the deterministic synthetic archive, then use a trained export:

   ```sh
   sudo -E python3 host/accelerator.py rs_accel.bit synthetic_int8.npz --image 0
   sudo -E python3 host/accelerator.py rs_accel.bit resnet8_int8.npz --image 0
   ```

4. The host checks ABI, arms S2MM before MM2S, submits tiles and checks **every**
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

No board execution, DDR validation, measured images/s, or measured power is
asserted by the software/RTL tests or by bitstream generation.
