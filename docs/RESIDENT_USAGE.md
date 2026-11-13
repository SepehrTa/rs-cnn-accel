# Resident accelerator ABI 3



This is the performance branch's resident implementation. Follow

PERFORMANCE_V2.md for its validation status; the v1 board evidence remains

separately retained in BOARD_RESULTS.md. A generated bitstream must pass setup

and hold timing before programming the board.



## Build and simulate



Vivado 2025.2, Zynq-7000 support, NumPy and the trained quantized model are

required for the full flow. The model archive and CIFAR dataset are generated

artifacts and are not checked into Git.



```sh

python -m unittest discover -s tests -p test_resident.py

python scripts/run_rtl.py --vivado C:/AMDDesignTools/2025.2/Vivado --suite directed --packed --cols 14 --filters 4 --p 2 --q 2

python scripts/run_resident_rtl.py --archive build/resnet8_int8.npz --vivado C:/AMDDesignTools/2025.2/Vivado --image 2

python scripts/fetch_ps_preset.py

vivado -mode batch -source scripts/build_board.tcl -tclargs bitstream board-resident 14 f4 p2 q2

```



The resident simulation uploads the model once, runs the selected image twice

without another upload, and checks every layer directly in the tensor memory.

Later layers consume the actual preceding RTL outputs. The sink stalls during

the run, and the scoreboard checks logits and MAC/completion counters.



The board builder supports integer column count plus `f1/f2/f4`, `p1/p2/p4`

and `q1/q2/q4`. `f4` with 14 columns makes a 12×14 physical array: four filter

groups, three kernel rows and fourteen output rows. P=2 and Q=2 are **time

interleaved contexts**, not extra physical multipliers. Other configurations

need their own resource/timing validation and corresponding simulation.



## Run the labelled dataset on PYNQ



Copy `host/`, `model/`, `scripts/benchmark_resident.py`, the trained

`resnet8_int8.npz`, CIFAR `test_batch`, and the matching `rs_accel.bit` /

`rs_accel.hwh` from one timing-closed build into a board directory. On the tested

PYNQ 3.1.1 image:



```sh

sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 scripts/benchmark_resident.py rs_accel.bit resnet8_int8.npz test_batch --count 10000 --report resident-board.json

```



To check one of the four embedded validation images without downloading the
complete dataset:

```sh
sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 host/resident_accelerator.py rs_accel.bit resnet8_int8.npz --image 0
```

This programs the FPGA, uploads the model once, compares every intermediate

of all embedded validation images, then runs exactly the requested number of

labelled test images. Every output logit must exactly equal the archive's

integer golden test logits. The default is all 10,000 images. `--count 100` is

a shorter board smoke test; `--skip-intermediates` skips only the prefix checks,

never the per-image logit comparison. The final report states the actual count.



Weights and numeric parameters stay in model BRAM; intermediate and residual

tensors stay in three 16 KiB tensor regions. Hardware performs all layer/tile

sequencing. Each normal image uses one receive/send DMA pair and reuses the

allocated buffers. The stream carries 64 header bytes + 3072 image bytes in,

and 40 logit bytes out. Model loading is measured separately.



## Counters and interpretation



The ABI ID is `0x52530003`. MMIO at `rs_accel_0`:



| Offset | Value |

|---|---|

| 0x04 | Write bit 0 to reset; invalidates uploaded model |

| 0x08 | Busy bit 0, sticky fault bit 1 |

| 0x0c / 0x10 | Tile-core busy clocks / completed tiles |

| 0x14 | Logical filter capacity, kernel rows, columns, output width |

| 0x18 | Physical filter groups bits 23:16, Q bits 15:8, P bits 7:0 |

| 0x20 / 0x24 | Accepted external input / output 32-bit words |

| 0x28 | Actual convolution MAC count |

| 0x2c / 0x30 | Waiting for PE bank / load handshakes overlapping MAC clocks |

| 0x40 | Resident network clocks through last result handshake |

| 0x44 / 0x48 | Model / activation synchronous RAM read requests |

| 0x4c | Activation bytes written, including input and INT32 pool/FC |

| 0x50 / 0x54 | Completed images / model-valid bit 31 and current layer |



The passive `rs_ddr_monitor_0` ID is `0x5253444d`. Offset 0x04 resets its

counters; 0x08/0x0c count accepted 64-bit HP0 read/write beats, 0x10 counts

asserted write strobes in accepted beats, and 0x14/0x18 count accepted read/write

address requests. Read bytes = eight times read beats. These measure accelerator

DMA traffic at the PS HP0 interface; they exclude CPU/cache traffic and do not

count DRAM row commands. Partial final writes are counted by their byte strobes.



All counters are unsigned 32-bit modulo counters. The runner samples per-image

deltas and accumulates into Python integers, so dataset totals survive wraps.

Model upload and debug prefix runs are excluded from normal-inference totals.

Utilization is convolution MACs / (physical PEs × complete network clocks),

including nonconvolution and sequencing overhead. The mapper's idealized

convolution-only prediction assumes a different schedule and memory service;

it must be labelled separately from this whole-network measurement.

