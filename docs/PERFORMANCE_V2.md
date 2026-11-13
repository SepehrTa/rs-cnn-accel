# Performance implementation work

The verified v1 design remains on main. Development is isolated on
`codex/performance-v2` so existing local Vivado session edits remain intact.

## Initial gap analysis and execution order

| Files | Gap / implementation work |
|---|---|
| `hw/rs_stream_core.sv`, `rs_pe.sv` | Alternating sample/MAC clocks; overlap sampling the next product with committing the previous product, then add banked PE contexts. |
| `host/protocol.py`, `hw/rs_accel_top.sv` | INT8 scalars occupy whole AXIS words; add explicit packed transport with signed byte/tail handling and legacy compatibility. |
| `hw/rs_axis_buffer.sv` | Existing ingress FIFO is not a resident tensor/model store. Add BRAM-backed model/tensor storage with real port/bank scheduling. |
| New resident compiler/sequencer | Host currently launches 293 tiled commands/image. Compile layer descriptors once, generate tile addresses/headers in RTL, and keep residual tensors on chip. |
| PE local stores/controller | No load/compute overlap or p/q folding. Add ping-pong channel loads, multiple stationary filter/channel contexts, and evaluate a 12x14 physical array within DSP/LUT/BRAM limits. |
| Host board driver / dataset runner | Add resident model loading, reusable buffers, complete 10,000-image board run, exact logits and measured cycles/access counters. |
| Mapper/report | Separate algorithm predictions from actual scheduled cycles, storage accesses, DMA bytes and host throughput. |

User priorities: (1) packed stream + initiation interval one; (2) resident global
buffer; (3) overlapped PE loading; (4) p/q folding and larger array; (5) complete
hardware layer sequencing; (6) full labelled board run and counter comparison.
Hardware sequencing is needed to use resident buffers, so its command compiler
and storage/address contract are developed alongside milestone 2.

## Gates

- Original software baseline: 11 NumPy checks, 8 protocol and 7 vector tests PASS.
- Preserve integer rounding/overflow/graph contract and actual-output dependency
  checks. Every transport/scheduler change requires directed and full-graph RTL
  comparisons before synthesis/board use.
- Use actual tensor liveness, exported parameter sizes, memory-port requirements
  and FPGA primitive rounding for the storage budget, not only raw byte totals.
- A 2x MAC initiation improvement and four-byte packing do not imply 2x total
  inference or exactly 4x total DMA: headers, INT32 values, tails and other phases
  remain. Publish measured end-to-end changes against retained v1 evidence.
- Only report 10,000-image hardware accuracy/throughput after that board run has
  actually completed. Current main's four-image measurements are not that run.

## First implementation evidence

- Export audit: 77,360 weight bytes + 4,108 numeric parameter bytes + 896
  descriptor bytes; exact peak live tensors 49,152 bytes (three 16 KiB banks).
- One-MAC-per-clock overlap passes the existing directed/full-network suite.
- Packed ABI-2 keeps a legacy tile mode and preserves INT32 header/bias/pool/FC
  values; signed bytes and independent segment tails are explicitly handled.
- Trained packed full graph: 114,762 exact values, 502,234 physical input words
  and 28,746 output words (2,123,920 stream bytes versus v1's 8,437,392).
- This first packed + MAC trace takes about 3.398 million wall cycles, versus
  v1's 3.802 million, not a 2x end-to-end core improvement. Load/output/control
  phases remain, motivating resident storage and overlapped/scheduled execution.
- Five packing tests pass; independent expected words include negative byte
  values and mixed INT32/INT8 classifier segments. Control/reset suite passes.

## Resident/context implementation (development, not yet board validation)

- Physical candidate: four filter groups × three kernel rows × fourteen output
  rows = 168 PEs (12×14); P=2 stationary filter contexts, Q=2 channel contexts.
  Independent ping-pong loaders fill the inactive PE bank during computation.
- Trained tiled packed RTL: 71 commands, 12,500,992 MACs, 1,752,551 wall clocks,
  110,886 overlapped load clocks. Every 114,762 intermediate/final value matches.
  PE activation writes are 2,920,960; weight writes remain 849,408.
- Resident ABI-3 stores the compiled model in 24,576×32-bit model BRAM and
  tensors in four byte lanes of 12,288 entries (48 KiB). Three 16 KiB tensor
  regions preserve residual liveness. Hardware descriptors generate all tile
  addresses and quantization parameters; no Python layer/tile loop is used.
- The actual compiled model is 20,607 words / 82,428 bytes including its header.
  One model upload precedes the dataset. Each inference is 784 input words
  (64-byte command + 3072-byte image) and ten INT32 output words.
- Resident trained RTL with a pipelined scratchpad-to-multiplier path passes
  all 14 layers / 114,762 exact values: 2,258,273 network clocks, 2,253,326
  core clocks, 37,171 model RAM reads, 160,010 activation RAM reads, 118,056
  activation write bytes. Additional runs cover repeated images without upload.
- The first packed-only 100 MHz build failed setup timing by 0.361 ns. It was
  not deployed. A further PE operand pipeline breaks its scratchpad critical
  path while preserving one product per clock; directed RTL passes afterward.
- Passive HP0 monitor counts accepted 64-bit read/write beats, WSTRB byte
  writes and address requests. Its directed RTL test checks handshake stalls,
  partial byte strobes, independent AXI-Lite AW/W arrival and counter reset.
  These are DMA interface transactions, not DRAM row commands or CPU traffic.
- The 168-PE resident build is undergoing real synthesis/implementation. No
  positive timing, resource-fit or full-dataset board result is claimed yet.

Remaining gates: finish resource/timing closure, repeated resident simulation,
board intermediate checks, complete 10,000-image exact-logit run, mapper versus
measured report, and refresh the cloneable GUI project/documentation.


Synthesis resource gate: the first 168-PE resident candidate inferred 36,996
LUTs, 11,721 FFs, 188 DSPs and 49 BRAM36-equivalent tiles in the accelerator
module alone. The raw byte-capacity lower bound (24+12 BRAMs) was optimistic:
Vivado rounds the model/tensor memory depths, inferring 32+16 BRAM36s, plus
ingress storage. More critically, 18,274 LUTs were used as memory against only
17,400 available SLICEM LUTs. The next candidate explicitly implements the tiny
PE weight stores as registers; DSP/BRAM contents and numeric behavior stay the
same. Whole-overlay routing/timing remains required.


The all-register weight candidate reduced SLICEM use to 16,930 LUTs but raised
accelerator total LUTs to 55,143, exceeding the device before DMA/control logic.
It was not deployed. The revision shares each activation-row store across all
four filter groups, retaining local operand registers and local weight stores.
Directed numerical/counter checks pass (45 commands, 10,524 exact outputs).
Full resident simulation/build remain active gates. The portable candidate
project opens/prepares in a clean source snapshot using Vivado 2025.2.


Shared-row candidate synthesis (Vivado 2025.2): accelerator module 29,011 LUTs,
11,746 FFs, 188 DSPs, 49 BRAM-equivalent tiles; 13,738 LUTs as memory (78.95%
of SLICEM LUT capacity). The complete PS7/DMA/monitor overlay is now routing.
The final shared-row resident simulation passes 28 layers / 229,524 exact values
on trained image 3 repeated twice, without re-uploading the model. Legacy,
control/reset and standalone PE suites pass after the storage change.


Placement of the shared-row candidate exposed two unpipelined paths around
20 ns: context-dependent MAC accounting inferred a three-DSP combinational
chain, and the channel-tail subtraction fed the next scratchpad address. This
candidate was not deployed. The next revision precomputes per-P-context MAC
increments with bounded shift/add cases, latches P/Q counts outside the MAC
loop, narrows context indices to configured widths, and registers scratchpad
addresses before the operand/product stages. Matching delayed psum commits and
two drain clocks preserve initiation interval one. Directed RTL/counters pass
with the new latency; the full resident regression and board build are running.


The address-pipelined candidate passes the complete trained tile trace:
12,500,992 MACs, 730,240 physical shared activation writes and 849,408 weight
writes, with all 114,762 tensor values exact. Its repeated resident trace passes
229,524 values / 28 layers / 25,001,984 MACs. Candidate accelerator synthesis:
28,410 LUTs, 13,666 FFs, 184 DSPs, 49 BRAM-equivalent tiles. Board routing remains
the deployment gate; these numbers are not a real-board performance result.


The next placed path was from resident filter-index arithmetic through cache
hit comparison and codec handshakes into core parameter write enables (18.6 ns).
The revision replaces convolution/classifier weight-address multiplication with
bounded 17-bit sequential pointers and per-filter channel bases. A nine-word
synchronous decoded FIFO breaks the sequencer/codec-to-core control path.
Tensor input addresses use the resident ABI's explicit 16-bit byte space.
Directed/reset checks and two complete resident executions pass afterward:
229,524 exact values; 25,001,984 MACs; 4,459,059 network clocks for two images.
No candidate with negative timing has been programmed onto the board.


A separate placed-path query confirmed the resident pool-size multiply had an
unregistered DSP cascade into state control (-5.581 ns in the earlier placed
candidate). The new revision calculates the bounded 8x16-bit product once in
layer setup and registers the last payload index before execution. Add/FC
payload limits are also precomputed. Two resident executions still pass all
229,524 values / 25,001,984 MACs. The active board candidate contains this fix.


The activation address/cache-compare chain was still 13 ns after placement.
A one-entry address-request pipeline now separates generation from cache lookup
and retirement. The generator can issue the next address on the same clock as
the previous request retires; it does not add a bubble per cached scalar. Last
payload status travels with the request, so the sequencer waits for retirement
before completing a tile. Directed/reset checks pass, and two full executions
pass 229,524 values / 25,001,984 MACs in 4,459,202 resident clocks.


In a separate immutable checkout, the reduction selector index is registered
while neighbor reduction runs, the selected link is captured separately, and
bias is added before requantization. This splits the remaining large-array mux
and adder path. The extra clock per convolution output costs about 86,016 clocks
per image and leaves the MAC initiation interval unchanged. Directed RTL and two
resident executions pass 229,524 exact values / 25,001,984 MACs in 4,631,239
network clocks for two images. The original candidate continues routing unchanged.


The reduction candidate completed route but failed setup timing; it was not
deployed. Packet codec geometry setup was the remaining critical path. A further
candidate registers patch height, activation count, and channel-size addition
in distinct setup phases, using a registered DSP multiply for the 8x8 count.
This adds only setup clocks per tile. Directed checks and two resident runs
pass all 229,524 values / 25,001,984 MACs in 4,631,500 network clocks for two
images. The timing-closed bitstream/board tests remain required.


Current checkpoint: the codec candidate closed 100 MHz timing at +0.034 ns
setup / +0.056 ns hold. Whole overlay: 29,777 LUTs, 18,264 FFs, 183 DSPs and
53 BRAM36-equivalent tiles. One actual resident inference passed: ten exact
logits, 2,315,746 network clocks, 27.15344 ms host latency, 392 HP0 read beats,
five write beats, 3,176 total HP0 bytes. The subsequent tiled benchmark did not
finish and Linux became unreachable. Remote serial/JTAG recovery did not
restore it. Full-dataset and full intermediate board validation remain pending;
no full-board accuracy/throughput result is claimed. See RESIDENT_CHECKPOINT.md.
