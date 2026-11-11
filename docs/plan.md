# Plan

The goal: an Eyeriss-style accelerator ([Chen et al., JSSC 2017](https://doi.org/10.1109/JSSC.2016.2616357))
on a PYNQ-Z2, using the Row Stationary (RS) dataflow, running ResNet-8 on CIFAR-10
(the MLPerf Tiny image-classification model) end to end on the board.

## Decisions

| Topic | Choice | Reason |
|---|---|---|
| Arithmetic | int8 weights and activations, 32-bit accumulation, per-channel requantization | today's standard (TFLite-style), and the same fixed-point scheme as the riscv-ml-core project |
| Requantization | `y = (acc * M + 2^(30+S)) >> (31+S)`, clamp | one multiply, one add, one shift: cheap in hardware |
| Residual add | `y = (a*Ma + b*Mb + 2^(30+S)) >> (31+S)`, clamp | both branches rescaled with a single rounding |
| PE array | parameterized, target 12 x 14 = 168 PEs as in the paper | the Zynq-7020 has 220 DSP blocks |
| Configuration | AXI-Lite registers instead of the paper's scan chain | written by the ARM |
| Off-chip memory | DDR through the ARM's HP AXI ports, with a DMA engine | the paper's DRAM link |
| Control | Python on the ARM sequences the layers | as in the riscv-ml-core project |

## Phases

| Phase | Content |
|---|---|
| P0 | Train and quantize ResNet-8; bit-exact integer model; RS mapping simulator in Python (access counts per memory level) |
| P1 | One PE running the 1-D convolution primitive |
| P2 | A PE set: one 2-D convolution with horizontal, diagonal and vertical reuse |
| P3 | PE array with the NoC (multicast by row/column IDs, local psum links) |
| P4 | Global buffer, DMA to DDR, control |
| P5 | Post-processing unit: requantization, ReLU, residual add, pooling |
| P6 | ResNet-8 on the board: full test set, frames/s, GMAC/s, PE utilization, DRAM accesses per MAC |
