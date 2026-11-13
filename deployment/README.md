# Development checkpoint bundle

`resident-v2-checkpoint.zip` contains the source project, matching timing-closed
`rs_accel.bit` / `rs_accel.hwh`, trained INT8/float archives and retained logs.
Each stored file is verified against the included SHA256 manifest.

This is a **development checkpoint**, not a final release. One trained resident
inference passed on the actual PYNQ-Z2. The board became unreachable during a
subsequent tiled follow-up test; diagnosis and the full 10,000-image board run
are still pending. Read `docs/RESIDENT_CHECKPOINT.md` before board use.

Extract into a new directory. Open `rs-cnn-accel.xpr` in Vivado 2025.2 and run
`scripts/prepare_vivado.tcl`. The source project does not require training to
open. The included `.bit` / `.hwh` must remain together with matching basenames.
The complete CIFAR-10 dataset is not included.
