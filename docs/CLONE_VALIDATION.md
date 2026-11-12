# Final main: fresh-clone verification

Date: 2026-11-12. Tested with Vivado 2025.2, SW build 6299465, on Windows.
Target: `xc7z020clg400-1`; top: `system_wrapper`; fabric clock: 100 MHz.

## What was checked

- Made an actual Git clone with no generated products, caches, vendor preset
  download, synthesis checkpoints or implementation results. Its directory
  name contains spaces to exercise project-relative paths.
- Opened the committed root `.xpr`; verified all referenced source inputs exist.
- Ran `scripts/prepare_vivado.tcl`: Verilog module-reference type, top module,
  checkpoint settings, module refresh, BD validation and IP product generation.
- Verified the seven top-level block-design cells: PS7, DMA, control and memory
  interconnects, reset, constant and accelerator.
- Ran all out-of-context vendor-IP synthesis runs, top-level synthesis,
  placement, routing, post-route physical optimization and bitstream generation.
- Opened the resulting routed design and checked setup/hold timing.

No Python, trained archive, connected board or separately installed PYNQ board
definition was used by this clone/open/build test. The preparation flow used
four Vivado threads and four synthesis jobs.

## Results and actual fixes

Initial default routing produced setup slack **-0.010 ns**, hold **+0.017 ns**.
The preparation script now enables post-route physical optimization with
`AggressiveExplore`. The same freshly synthesized/routed design then completed
that step and regenerated the bitstream with:

| Final fresh-clone build | Result |
|---|---:|
| Setup WNS / TNS | +0.006 ns / 0 ns |
| Hold WHS / THS | +0.020 ns / 0 ns |
| Failing setup / hold endpoints | 0 / 0 |
| LUTs / FFs | 9,060 / 6,665 |
| DSPs / BRAM36-equivalent tiles | 59 / 4 |
| Bitstream | 4,045,690 bytes; generated successfully |

These results describe this clean-clone implementation, not the earlier
deployed bitstream in BOARD_RESULTS.md. The sources are unchanged; implementation
optimization/placement can change utilization and timing. The new bitstream was
not programmed onto the board during this publication check. Actual board and
trained-model evidence remains in BOARD_RESULTS.md and its original reports.
Timing margin is small; inspect timing after every new implementation.

An initial standalone `synth_design -rtl` check failed because it bypassed the
project's out-of-context IP synthesis/linking flow. The validated flow uses
project synthesis/implementation runs, as the GUI's Generate Bitstream action
does. No missing RTL was hidden or replaced to make that check pass.

Raw final timing, utilization and run excerpts are retained under
`reports/final-clone/`. Existing vendor DDR preset and AXI adaptation warnings
remain visible in the logs; they were not suppressed.

## Reproduce

Clone the default `main` branch, open `rs-cnn-accel.xpr` in Vivado 2025.2, run
**Tools > Run Tcl Script > scripts/prepare_vivado.tcl**, and choose
**Generate Bitstream**. No paths from the original author's PC are needed.

Equivalent project-run commands in the Vivado Tcl console, from repository root:

```tcl
source scripts/prepare_vivado.tcl
launch_runs synth_1 -jobs 4
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
open_run impl_1
report_timing_summary
```

The previous `main` tip was retained as `archive/pre-final-2026-11-12`, and the
final project is published as a fast-forward without rewriting existing history.
The author's separate, locally edited Vivado session was preserved.
