# Implemented RS stream accelerator

## Schedule and communication

`rs_stream_core` has FILTERS x KMAX x COLS PEs (default 2 x 3 x 8 = 48).
It is a restricted implementation of the row-pair primitive in
[Chen et al., ISCA 2016](https://eems.mit.edu/wp-content/uploads/2016/04/eyeriss_isca_2016.pdf).
PE `(g,r,j)` holds filter row `r` of filter `g`, activation row `j*stride+r`,
and an INT32 partial-sum row of up to 32 output pixels. Columns correspond to
output **rows**, not output channels. This distinction matters for RS reuse.

For each channel:

1. Stream a padded input patch of height `(tile_rows-1)*stride+k` and width
   `(tile_width-1)*stride+k`. A word at `(iy,ix)` writes every PE satisfying
   `iy=j*stride+r`, across both filter groups. The same source row is multicast
   to diagonally related PEs; no separate DDR read per destination is needed.
2. Stream filter `g`, row `r`, tap `s`; multicast it horizontally to all active
   output-row columns. Each PE keeps up to three taps throughout the channel.
3. For output x, visit k taps. Each PE reads local activation `[x*stride+s]`
   and its local weight `[s]`, registers the signed product, then adds it to
   local `psum[x]` on the next clock. Channel zero/tap zero initializes each
   entry; there is no RAM-wide reset or assumed initialized memory.
4. Process the next channel, retaining psums. No channel psums return to DDR.

After all channels, output `(g,j,x)` visits a registered row-reduction chain:
`link[0]=psum[g,0,j,x]`, then `link[r]=link[r-1]+psum[g,r,j,x]`.
Each PE owns its reduction adder and link register. Its neighbor input is wired
directly to row `r-1` of the same filter group/column, or zero at row zero. The
controller enables one row hop per clock. Placement is unconstrained; electrical
neighbor connectivity is explicit, but no physical mesh floorplan is claimed.
Bias, six-stage requantization, saturation and optional ReLU follow.
The next output begins after the previous output handshake, allowing arbitrary
output stalls without changing data/TLAST.

For 1x1 only physical row zero of each three-row group participates; the other
rows' MAC enables are low. Stride is one or two. Software handles external
padding, output-channel tiling, row tails, width tails, and tensors wider than
the PE row store. Existing analytical mapper mappings are **not** RTL commands:
this schedule fixes one filter/channel per PE at a time and two filter groups.
There is no general runtime Eyeriss multicast routing table.

`sim/tb_stream.sv` counts every PE MAC, activation write and weight write and
checks them against independently derived packet dimensions. Numeric output
checks exercise diagonal overlap, horizontal multicast and vertical reduction.
The top-level test includes synchronous ingress memory, random output stalls,
input bubbles and dependent tensors sourced from actual emitted RTL values.

## Storage and ports

| Store (default configuration) | Logical bytes | Implementation / ports |
|---|---:|---|
| 48 activation rows x 34 INT8 | 1,632 | local distributed RAM, one load write / one sample read |
| 48 weight rows x 3 INT8 | 144 | local distributed RAM, one write / one sample read |
| 48 psum rows x 32 INT32 | 6,144 | distributed RAM, one MAC read/write plus separate reduction read |
| 48 registered neighbor links x INT32 | 192 | PE-local registers and direct row-neighbor wiring |
| 256 classifier inputs x INT32 | 1,024 | distributed RAM, one input write / one classifier read |
| two ingress banks x 256 x 32-bit | 2,048 | two synchronous 1R/1W BRAM banks, striped by word address |
| total memory/link payload | 11,184 | excludes rounding to primitive depth, pipeline registers and control |

Each ingress bank handles at most one write and one read per clock. A separate
prefetch register provides an elastic output; total ingress capacity is 513
words. Full/empty arbitration prevents overwrite and underflow. PE memory
load/sample/accumulate/reduce phases are mutually exclusive. Synthesis reports
give actual rounded primitive use; logical bytes are not a BRAM utilization claim.

No full feature map or network weights reside in FPGA memory. Host memory keeps
intermediates and residual skip tensors. This bounds FPGA storage but repeats
patch halos and weights between tiles. AXIS currently transports one scalar per
32-bit word, including INT8 values; packing four bytes, resident weight/activation
tiles, double-buffered PE loads, and pipelined output scheduling remain performance
work. Input buffering overlaps transfers with some compute, but cannot hide all
repeated traffic or 293 per-image DMA launches.

## Numeric contract

- Symmetric signed INT8 weights/activations, zero point zero; CIFAR input pixel-128.
- Signed 8x8 products; signed INT32 psums and bias, with host conservative range
  rejection covering every intermediate sum. Packets bypassing the host must
  meet the same bounds; RTL does not detect arithmetic overflow.
- Conv output: `clip((acc*M + 2^(shift-1)) >> shift)`, `shift=31+S` in 1..62,
  M positive signed INT32. One rounding, ties towards positive infinity.
  Four smaller signed/unsigned partial products reconstruct the exact 64-bit
  product; separate registers perform combine, round, shift and saturation.
- Residual: `clip((a*Ma+b*Mb+2^(shift-1)) >> shift)`, shared shift and one rounding.
- Pool: signed INT32 spatial **sum**, not integer average. The existing trained
  model exporter folds the 8x8 `/64` into FC bias/output scale.
- FC: signed INT32 pooled inputs, signed INT8 weights and INT32 bias/logits;
  no output requantization, preserving the golden model argmax scale.

No floating point exists in the datapath. NumPy reference/export checks reject
out-of-range biases/accumulators and unsupported shifts instead of silently wrapping.

## Packet ABI version 1

Each command is 16 uint32 header words followed by payload. All signed scalars
use two's complement. No byte-swapping on little-endian PYNQ ARM. Every AXIS
beat carries four valid bytes; partial TKEEP is unsupported. Receive TLAST is
ignored: validated header dimensions determine input lengths. Output TLAST
marks the final result of **each command**, matching one simple-mode DMA receive.

| Header word | Conv (opcode 0) | Add (opcode 1) | Sum pool (opcode 2) | FC (opcode 3) |
|---|---|---|---|---|
| 0 | 0 | 1 | 2 | 3 |
| 1 | kernel, 1 or 3 | element count | spatial count, <=65535 | input count, <=256 |
| 2 | stride, 1 or 2 | unused | channels, <=65535 | class count, <=65535 |
| 3 | padded patch width, <=34 | unused | unused | unused |
| 4 | output tile width, <=32 | unused | unused | unused |
| 5 | tile rows, <=8 | unused | unused | unused |
| 6 | filters, 1 or 2 | unused | unused | unused |
| 7 | input channels, 1..65535 | unused | unused | unused |
| 8 | ReLU bit | ReLU bit | unused | unused |
| 9,10,11 | filter 0 bias,M,shift | unused,Ma,shift | unused | unused |
| 12,13,14 | filter 1 bias,M,shift | unused,Mb,unused | unused | unused |
| 15 | reserved | reserved | reserved | reserved |

Conv payload for each channel: row-major padded patch followed by
filter-major, row-major kernel weights for the active filters. Outputs are
filter-major, output-row-major, x-major. Add payload interleaves a,b per element.
Pool payload is channel-major spatial data, one sum per channel. FC payload
starts with all inputs then `[bias, weights...]` per output class.

Headers are range-checked; an invalid header enters sticky FAULT and deasserts
ready until reset. Truncated payloads require host timeout/recovery. Reserved
words and upper bits of INT8 payload scalars are ignored. Do not submit a new
independent image until previous DMA transfers have completed.

## Control and board integration

`rs_accel_top`: AXI4-Lite, independent AW/W latches, one outstanding read/write,
stable stalled responses, byte strobe for reset, SLVERR for unmapped accesses.

| Byte offset | Meaning |
|---|---|
| 0x00 RO | ABI ID 0x52530001 |
| 0x04 WO | bit 0 requests one-cycle core + FIFO reset |
| 0x08 RO | bit 0 core busy, bit 1 sticky header fault |
| 0x0c RO | uint32 core busy cycles, wraps; reset clears |
| 0x10 RO | uint32 completed commands, reset clears |
| 0x14 RO | capacities: filters[31:24]=2, kernel[23:16]=3, columns[15:8], width[7:0]=32 |

Busy does not include an ingress-only queued word before the core receives its
header. DMA completion plus expected command count defines completion. Soft
reset is an abort: stop DMA before reset, then restart/reset DMA before reuse.

Board Tcl connects PS7 GP0 to control and DMA registers, DMA MM2S/S2MM through
an AXI interconnect to HP0 DDR, 32-bit streams, 64-bit memory transfers and a
23-bit DMA length register. Polling is used; no interrupt driver is needed.
All fabric logic uses PS FCLK0 and its synchronized active-low reset. PS DDR/MIO
configuration comes from the hash-pinned official PYNQ v3.1 PYNQ-Z2 hardware
project, extracted locally without executing the upstream Tcl or installing
board files. The preset carries its original DDR delay warnings in Vivado 2026.1;
they must be evaluated on the actual board, not hidden by a fabricated preset.
