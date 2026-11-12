# RS-CNN Accelerator — Local Coding Agent Instructions

## Mission
Develop the existing `rs-cnn-accel` repository into a working, tested row-stationary INT8 CNN/ResNet inference accelerator targeting the PYNQ-Z2 (Zynq-7020). Preserve and extend existing work. Do not replace functioning modules merely to simplify the task.

## Start by auditing the repository
- Inspect the README, training/quantization/export pipeline, reference integer inference, mapper, simulator, tests, and existing FPGA/host code.
- Identify the exact network graph and quantization/rounding rules. Treat the existing integer golden model as the numerical reference, but independently test it.
- Inspect Git status before edits; create a dedicated feature branch when safe. Never discard local modifications.
- Produce a concise file-by-file gap analysis and prioritized execution plan. Then implement; do not stop at the plan.

## Hardware design requirements
- Implement genuine row-stationary data reuse: weight reuse, activation reuse, and local/neighbor partial-sum accumulation. Describe actual PE scheduling and communications; do not call a broadcast-MAC array row-stationary without evidence.
- Parameterize array dimensions, tile sizes, buffer capacities, and memory widths. Choose a feasible initial configuration for PYNQ-Z2 and make bigger sizes optional.
- Use synthesizable SystemVerilog; no unsynthesizable constructs in the hardware datapath.
- Implement or integrate a PE, PE array, scratchpad/banked global buffer, scheduler, address generation, control registers, memory/DMA interface, and output post-processing as needed.
- Support the operations required by the exported model: 3x3/1x1 convolution, signed INT8 multiplications with safe accumulation, bias, scale-aware requantization, ReLU, residual addition with possible scale mismatch, global average pooling, and classifier.
- Respect signed arithmetic, intermediate bit widths, rounding, overflow, saturation, valid/ready protocols, memory latency, and reset sequencing.
- Minimize unnecessary DDR traffic and report on-chip storage requirements. Avoid silently assuming unlimited bandwidth or BRAM ports.
- Add simulation-friendly interfaces/testbenches and Vivado TCL/project scripts and constraints appropriate to the actual target board and board clock.

## Verification and validation
- Run available Python/unit tests first and keep them passing.
- Add targeted directed and randomized tests for PE, data reuse, convolution tiles, boundary/stride/padding cases, quantization, and residual pathways.
- Compare intermediate tensors and final classification outputs against the exported integer golden model. Report exact mismatches and fix causes.
- Run the RTL simulation tools actually installed on this computer (e.g., Icarus/Verilator/xsim); use a simulator only if it supports the RTL and testbench used.
- If Vivado is installed, execute batch synthesis/implementation/timing analysis where feasible. Capture tool versions, utilization (LUT, FF, BRAM, DSP), timing, and warnings.
- If the PYNQ-Z2 is accessible, test using the board; otherwise provide a repeatable board-validation procedure. Never claim real-board success without running it.
- Distinguish estimated performance, simulation results, synthesis/timing reports, and real-board measurements.

## Software and host
- Preserve existing training, quantization, model export, and NumPy reference workflows.
- Implement a clear PYNQ host interface for model loading, layer configuration, execution, and output checks, as the installed image and DMA/AXI design permit.
- Add deterministic reproducible commands, requirements, and documentation.

## Operating practices
- Work only inside this repository and explicitly authorized build folders unless the user approves wider access.
- Ask for approval before deleting user files, changing system configuration, installing large packages, flashing hardware, or pushing to GitHub.
- Prefer small, reviewable commits and incremental vertical slices. Do not invent pass results; run checks and report their outputs.
- Do not require one monolithic pass: keep a live TODO/status document showing completed work, evidence, and remaining blockers.
- If tools are missing, check available alternatives, document the blocker, and implement/test the rest rather than falsely reporting completion.
- Prioritize a demonstrably working end-to-end inference path over unvalidated peak-PE-count claims.

## Completion criteria
1. Reproducible reference/export pipeline and passing software tests.
2. RTL module and integration tests for the implemented accelerator.
3. End-to-end simulation matching the integer reference for agreed test vectors.
4. Vivado build scripts and documented utilization/timing from actual runs, if tooling available.
5. PYNQ Python driver and reproducible integration instructions.
6. A final report listing all files changed, exact commands and outputs, known limitations, and the distinct status of FPGA deployment/board tests.
