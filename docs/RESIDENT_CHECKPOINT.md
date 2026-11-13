# Resident accelerator development checkpoint

This branch is a backup of ongoing work, **not the final validated release**.
The full 10,000-image board evaluation has not run yet.

The packed transport, 168-PE array (12 x 14), P=Q=2 PE contexts, ping-pong loads,
resident model/tensor BRAM and hardware layer sequencing are implemented. Two
complete dependency-preserving resident RTL executions match all 229,524
intermediate/final values. The packed tiled full graph also matches all 114,762
values. Control, MAC-abort recovery and DDR-monitor RTL checks pass.

Vivado 2025.2 completed the complete PS7/DMA/monitor implementation at 100 MHz:
setup +0.034 ns, hold +0.056 ns; 29,777 LUTs, 18,264 FFs, 183 DSPs and 53
BRAM36-equivalent tiles. The matching bitstream/HWH pair is in the checkpoint
bundle; source opening/preparation also passed in Vivado.

## What actually passed on PYNQ-Z2

One embedded trained image produced ten exact INT32 logits, class 3. Observed:

| Measurement | Actual single-image board result |
|---|---:|
| Host inference latency | 27.15344 ms |
| Resident network clocks | 2,315,746 |
| Core busy clocks | 2,310,131 |
| Convolution MACs | 12,500,992 |
| Internal tiles / external image commands | 71 / 1 |
| Input / output stream words | 784 / 10 |
| HP0 64-bit read / write beats | 392 / 5 |
| HP0 bytes, input and result including header | 3,176 |
| Loads overlapping MAC clocks | 69,248 clocks |

These are accelerator DMA transfers at HP0, not physical DRAM row commands or
CPU/cache traffic. Model upload precedes inference and is excluded from this
per-image measurement. One successful image does not establish full-set board
accuracy or sustained dataset throughput. Raw logs and hashes are in
`reports/resident-v2/`.

## Current blocker and next steps

The subsequent legacy tiled benchmark did not complete its first inference,
and the board stopped responding to SSH, ping and its serial console. The
Ethernet link remains up. JTAG sees the Z7020, but the ARM cores do not respond
to halt/reset requests; remote recovery attempts did not restore Linux. The
cause has not been established. The packed tiled RTL test passes, which does
not prove the board driver/runtime path is correct.

A physical board power cycle is required. After recovery: run the resident
intermediate checks and all 10,000 images, diagnose the tiled follow-up failure,
retain exact logits/counters, and publish the final comparison. Continue with
`RESIDENT_USAGE.md`; this checkpoint must not be described as those tests having
passed. The previous validated v1 results remain in `BOARD_RESULTS.md`.

The published checkpoint was downloaded from GitHub and all 174 bundle files
were checked against their SHA256 manifest. A fresh source snapshot also passed
`scripts/prepare_vivado.tcl` in Vivado 2025.2; its log is retained in
`reports/resident-v2/published-clone-prepare.txt`.
