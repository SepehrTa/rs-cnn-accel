# Resident ResNet-8 accelerator

The ABI-3 datapath is ingress FIFO -> resident command/graph sequencer -> tile
codec -> RS compute engine. Tile outputs return to resident tensor RAM; only
the requested final tensor reaches the external output stream. Opcodes 0-3
retain the legacy tile interface. Opcode 4 uploads model words; opcode 5 runs
a packed input image through all descriptors (or a debug layer prefix).

## Storage and dataflow

The trained model contains 77,360 INT8 weight bytes, 4,108 bytes of INT32 numeric
parameters, 896 descriptor bytes and a 64-byte model header: 82,428 bytes total.
Model RAM allocates 24,576 words. Tensor liveness requires three 16 KiB regions,
including the live residual input while both convolution outputs are produced.
The compiler chooses a free region before each operation and releases sources
only after their last consumer. No input/output alias is assumed.

Model RAM has one synchronous word-read port and an upload write port. Tensor
RAM has four byte-lane banks, each with one synchronous read and one masked
write port. INT8 writes update one lane; pooling/classifier writes update four.
One-word caches convert adjacent byte requests into scalar tile payload words.
They are invalidated between layers, and tensor sources/destinations are
separate during each layer. Finite ports/cache misses are included in clocks;
raw BRAM capacity does not imply unlimited operand bandwidth.

The physical candidate is 4 filter groups x 3 kernel rows x 14 output rows =
168 PEs. Each PE has two filter contexts (P=2), two channel contexts (Q=2), and
two ping-pong load banks. Computation interleaves P, Q, filter tap and output x,
updating separate P-context psums. There is one signed INT8 product per enabled
PE per compute clock after pipeline fill. Registered address and operand stages separate next-sample generation,
scratchpad lookup and multiplication. Matching delayed commit addresses and
two final drain clocks preserve the result and one-product initiation.
Context counts and MAC increments are precomputed outside the MAC loop, so
performance instrumentation does not create a chained multiplier path.

Activation rows are identical across filter groups. Only group zero stores
them; its scratchpad output is broadcast to each filter group's operand
register. This removes three duplicate stores per row pair. Diagonal multicast
still reuses input rows among kernel/output-row pairs; stationary weight rows
are multicast across output rows on load. Each PE owns its weight contexts and
INT32 psums. Registered neighbor links perform kernel-row reduction before
bias, six-stage requantization, saturation and ReLU.

An independent loader fills the inactive bank while the active bank computes.
A bank becomes ready only when all Q channels/P filters have arrived; compute
releases it only after the final sample enters the operand pipe. Scoreboards
assert no load writes to a computing bank and compare actual enable counts.

| Local payload, 168 PEs / P=Q=2 | Bytes before primitive rounding |
|---|---:|
| Shared activation rows, 42 x 2 banks x 2 Q x 34 | 5,712 |
| PE weight rows, 168 x 2 banks x 2 P x 2 Q x 3 | 4,032 |
| PE psums, 168 x 2 P x 32 x INT32 | 43,008 |
| Registered row-neighbor links | 672 |
| Classifier input store | 1,024 |
| Ingress RAM payload | 2,048 |

RTL defaults retain a smaller tile-compatible configuration. The committed
board design and explicit `14 f4 p2 q2` arguments select the larger candidate.
Actual synthesis reports, rather than this byte table, establish fit.

## Hardware sequencing and boundaries

Fourteen 16-word descriptors supply operation, shape, source/destination and
parameter addresses. Hardware generates convolution tiles and padding, extra
quantization parameters for more than two filters, residual interleaving,
sum-pooling and full-INT32 classifier operands. Software uploads one compiled
model, then launches one DMA pair per image with persistent buffers. It does
not iterate layers or compile tiles per image.

Normal input is 16 command words + 768 packed pixel words; output is ten INT32
logits. No weights, intermediate feature maps or channel psums traverse DDR
between layers. The one-time model upload and 64-byte image command remain
transport traffic and are explicitly counted. Prefix debug commands re-execute
the graph up to one layer and export that tensor; the three-region store does
not retain all historical tensors.

This is a fixed regular RS schedule, not the analytical mapper's arbitrary
routing/mapping. In particular, 1x1 layers use only one of three kernel rows;
serial reduction/requantization, memory ports, tails and nonconvolution layers
reduce whole-network utilization. Predictions and counters are labelled with
their different scope and units.
