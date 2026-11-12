# Second-PC resume audit and validation

This records the initial restore step. Actual board deployment and subsequent
trained-model work are recorded in [BOARD_RESULTS.md](BOARD_RESULTS.md).

2026-11-12. Restored GitHub branch `codex/rs-rtl-integration` at `bc6088c` into
`C:/Users/stabr/Documents/ChatGPT/FPGA/rs-cnn-accel`. Work continues on the local
branch `codex/resume-validation`. Initial working tree was clean.

## Audit and execution priorities

| Files | Finding and next action |
|---|---|
| `README.md`, `AGENTS.md`, `docs/*` | Existing implementation is substantially complete. Retain historical 2026.1 results; revalidate this PC separately. |
| `model/graph.py`, `nn_numpy.py`, `quantize.py` | Nine convolutions, three residuals, sum GAP, FC; single rounding with ties toward positive infinity. Export includes multiple images; use this existing contract. |
| `host/protocol.py`, `accelerator.py` | Packet compiler and DMA path exist. CLI selection previously permitted an invalid slice and narrowed input to INT8 before checking. Preflight archive/index before overlay loading. |
| `scripts/run_rtl.py` | Archive validation was fixed to image zero. Add selected-image support and separate nonzero-image output folders. XSim 2025.2 needs explicit default timescale. |
| `tests/test_vectors.py` | Add regressions for non-first images, invalid indices, corrupt/missing tensors, input range and pre-overlay failures; compare generated packets with independent scalar interpreter. |
| `hw/*`, `sim/*`, `mapper/*` | Preserve existing RTL/schedule; rerun baseline software and installed XSim suites before considering optimizations. |
| `.gitignore`, `build/` | Trained weights/data and generated bitstreams/HWH are excluded from Git; prior binaries are not restored by cloning. |

Priorities: (1) baseline tests/tool compatibility; (2) fix selected-image validation
and run a second full-network image; (3) obtain actual trained export/data;
(4) rebuild deployment artifacts and test on an accessible board with authorization;
(5) only then tune stream packing/residency and measure board performance.

## Changes

- `host/protocol.py`: `load_vector` checks selection, integer INT8 input, complete
  tensor batch shapes and exact golden recomputation for the selected image.
- `host/accelerator.py`: preflight before importing/loading Board; compare actual
  inference against recomputed tensors without narrowing invalid inputs first.
  `Accelerator.infer` API remains available.
- `scripts/run_rtl.py`: `--image`, argument checks, per-image folders, guaranteed
  vector-file closure, and explicit XSim `--timescale 1ns/1ps`.
- `tests/test_vectors.py`: seven regression tests including an independent scalar
  packet execution for the second selected image.
- README/progress/board documentation: restored environment, deployment dependency
  on the NumPy model and repeatable selected-image commands.

## Executed checks on this PC

Python 3.14.7; NumPy 2.5.4; Vivado/XSim 2025.2 (SW build 6299465).
Commands below run at repository root. For every RTL command append
`--vivado C:/AMDDesignTools/2025.2/Vivado`.

| Command | Observed result |
|---|---|
| `python tests/test_nn_numpy.py` | 11 PASS; ALL PASS; exit 0 |
| `python tests/test_rs.py` | 27 PASS; exact data and counts; exit 0 |
| `python -m unittest discover -s tests -p test_protocol.py -v` | 8 tests OK |
| `python -m unittest discover -s tests -p test_vectors.py -v` | 7 tests OK |
| `python scripts/run_rtl.py --suite pe` | 1,632 signed MAC checks, 544 neighbor reductions; PASS |
| `python scripts/run_rtl.py --suite control` | AW/W, stalls, fault, abort/reset recovery; PASS |
| `python scripts/run_rtl.py --suite directed` | 10,524 exact values, 94 packets; PASS |
| `python scripts/run_rtl.py --suite network` | 114,762 exact values, 293 packets, 3,802,331 wall cycles; PASS |
| `python scripts/run_rtl.py --suite network --archive build/two_image_int8.npz --image 1` | Second image: 114,762 exact values, 293 packets, 3,802,331 wall cycles; PASS |
| `python -m compileall -q host model mapper scripts tests` | exit 0 |
| `git diff --check` | exit 0 |

Original RTL, counters and first-image logits match the historical validation
report. Raw new simulation logs are under `build/sim/<suite>/*.stdout.txt`.
Existing tracked `docs/reports/` remains the original PC's evidence.
New XSim logs are retained separately under `docs/reports/resume-2025.2/`.

The second image has different logits from the first:
`[2957, -23454, -14843, -1970, 2777, 460, 7759, -3196, -20500, 16587]`.
Generate its two-image archive after the default network run with this Python
snippet from the repository root, then run the selected-image command above:

```python
from pathlib import Path
import sys
import numpy as np
sys.path[:0] = ['host', 'model']
from protocol import load_model
from nn_numpy import forward_int
source = Path('build/sim/network/synthetic_int8.npz')
q, graph = load_model(source)
with np.load(source, allow_pickle=False) as z:
    export = {k: z[k].copy() for k in z.files
              if not k.startswith('vec.') and k != 'test_logits'}
    first = z['vec.input'].copy()
second = np.random.default_rng(17).integers(-128, 128, first.shape, dtype=np.int32)
t = forward_int(q, graph, np.concatenate((first, second)))
export.update({f'vec.{n}': v.astype(np.int32) for n, v in t.items()})
export['vec.labels'] = np.array([-1, -1])
export['test_logits'] = t['logits'].astype(np.int32)
np.savez('build/two_image_int8.npz', **export)
```

Two observed failures were investigated: the original XSim 2025.2 run rejected
missing RTL timescales (fixed in the runner); broad unittest discovery imports
the legacy script `test_nn_numpy.py`, whose top-level `sys.exit` causes a loader
error despite its checks passing. Run that script directly and the two unittest
suites with the explicit patterns above. No test pass is inferred from the
failed broad discovery command.

## Remaining external work

No trained export or CIFAR data is present in this checkout. Synthetic tests
establish arithmetic consistency, not accuracy. No board was programmed and no
new synthesis/timing closure or board latency/power claim is made. Source and
reports are restored; generated deployment files need rebuilding or transfer
from the original PC. Keep matching bitstream/HWH pairs together.
