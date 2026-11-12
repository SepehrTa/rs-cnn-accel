# Eyeriss adaptation: improvements and measured board results

## Paper relationship

[Chen, Emer and Sze, ISCA 2016, Section V](https://eems.mit.edu/wp-content/uploads/2016/04/eyeriss_isca_2016.pdf)
describes row-pair processing with horizontal filter reuse, diagonal input reuse
and vertical partial-sum accumulation. This project implements a restricted
version of those mechanisms on an FPGA, rather than the paper's general
mapping/network architecture.

| Context | Eyeriss paper, Fig. 4 / Section V | This project |
|---|---|---|
| Platform | Fabricated ASIC | PYNQ-Z2 FPGA with PS7 and AXI DMA |
| Physical PEs | 168 | 48, arranged as two 3-row filter groups and eight output-row columns |
| Clock | 200 MHz | 100 MHz on the tested board |
| Operands | 16-bit fixed point | Signed INT8 operands, INT32 accumulation/classifier |
| Network used here for evidence | Paper uses AlexNet | ResNet-8 / CIFAR-10 |

These are implementation differences, not normalized performance or energy
improvements. The numerical graph, technology, precision, PE count, memory
system and host environment are different. The paper's energy-efficiency
results are not measurements of this FPGA.

## Implemented dataflow improvements

- `hw/rs_pe.sv` holds filter taps, activation rows and channel partial sums locally.
- `hw/rs_stream_core.sv` multicasts filter rows horizontally and input rows
  diagonally/across filter groups, and schedules channel passes without sending
  channel partial sums back to DDR.
- Each PE contains a registered partial-sum link to its row neighbor; row
  reduction is performed before bias/requantization/output.
- The actual host path supports the full exported graph, including residuals,
  pooling and classifier, and checks every intermediate tensor on the board.

The design has a fixed restricted schedule, bounded row stores, a 513-word
ingress capacity and host-side tiling. Feature maps and model weights are not
fully resident on-chip. Halos and weights repeat between tiles, and INT8 stream
values currently occupy 32-bit words. See ARCHITECTURE.md for the schedule,
ports, storage accounting and remaining packing/residency work.

## Quantified operand reuse: counter-derived, not a second board benchmark

Source: `reports/network-manifest.json` and `reports/network-xsim.txt`.
The testbench checks the actual RTL enable counts against packet dimensions.
These counters cover the nine convolutions; they do not count all operations
in residuals, pooling or the classifier.

Define the comparison explicitly: a no-reuse PE receives a freshly loaded
weight and activation for each convolution MAC. For N MACs, its operand-load
counts are N weight writes and N activation writes. This is a hypothetical
load model, not an implemented non-RS accelerator measured on the board.

| Count | No-reuse load model | Checked RTL trace | Reduction |
|---|---:|---:|---:|
| Convolution MACs | 12,500,992 | 12,500,992 | Same computation |
| PE weight-load writes | 12,500,992 | 849,408 | 93.21% |
| PE activation-load writes | 12,500,992 | 5,841,920 | 53.27% |
| Combined operand-load writes | 25,001,984 | 6,691,328 | 73.24% |

Calculations:

```text
weight reduction = 100 * (1 - 849408 / 12500992) = 93.2053%
activation reduction = 100 * (1 - 5841920 / 12500992) = 53.2683%
combined reduction = 100 * (1 - (849408 + 5841920) / (2 * 12500992)) = 73.2368%
MACs per PE weight load = 14.7173
MACs per PE activation load = 2.1399
```

This quantifies avoided PE operand reloading. It does not count all local RAM
reads, psum updates, DDR transactions, cache maintenance or energy. No 73.24%
DDR/power reduction or corresponding speedup is established by these counts.
The counter trace is simulation evidence; the board exposes busy/command
counters, not these per-PE access counters.

## Measured on the actual PYNQ-Z2

Source: `reports/board-2025.2/trained-inference.json` and
`reports/board-2025.2/synthetic-board.json`.

| Actual board result | Observed value |
|---|---:|
| Trained execution checks | Four distinct labelled images, three repeats each |
| Exact intermediate/final comparisons | 1,377,144; zero mismatches |
| Mean complete host inference latency | 2.408330 s/image |
| Measured complete host rate | 0.415226 images/s |
| Mean core-busy cycles / time | 3,661,019.33 / 36.610193 ms at 100 MHz |
| Mean transport time, including allocation/cache/DMA/polling | 1.651583 s/image |
| Mean packet compilation/tensor assembly time | 0.756746 s/image |
| Commands / AXIS payload per inference | 293 / 8,437,392 bytes |
| Additional real-board tests | Header-fault recovery and 100 signed DMA/reset transfers passed |

Host inference includes packet construction, buffer allocation and DMA; model
loading, golden preflight and overlay loading are outside that timed interval.
Core-busy time excludes idle gaps between commands and must not be inverted
and advertised as sustained board throughput. AXIS bytes are not measured DDR
controller traffic. Full-set 86.55% accuracy was evaluated in the NumPy reference;
the board checked the four exported images repeatedly.

This is the demonstrated practical result: the Eyeriss-inspired reuse schedule
runs the trained graph correctly on a resource-limited FPGA at 100 MHz. A matched
non-RS implementation and equivalent workload/system measurements would be
needed for a measured RS speedup. Supply-current instrumentation would be
needed for power/energy savings. Host overhead currently dominates end-to-end
latency; packed streams, buffer reuse and resident tiles are future optimizations.

Machine-readable counts, formulas and means: `reports/eyeriss-comparison.json`.
