# Initial engineering and development validation — 2026-10-27

These are **single-seed validation results, not final test results or a new-method claim**. They demonstrate that the pipeline executes on full-size data and produces inspectable artifacts. No calibration/test performance has been opened for this development pilot.

## Scope and identity

- Configuration: `configs/pilot.yaml`, seed 17, 42 original fields, 192 encoded inputs.
- Training partition: 122,737 rows; validation: 26,302; reserved calibration: 26,302; reserved original test: 82,332.
- Data source: explicit, fixed-commit public mirror documented in [DATA.md](DATA.md). It has **not** been byte-matched against an independently obtained official copy.
- Prepared manifest SHA-256: `8f35efba5e03687c2a6c153c9fff8a7acb2fe70a27474099fcc83fa12da58af4`.
- Source SHA-256 recorded by every run: `1856f549b795ddc292b51454407697a4f66b421540e6b26bd258f220bacb035e`.
- Source snapshot began at commit `9b2fdb66604a2e5176b5c3a06c4cb202bb3f538d`; subsequent test/CI/documentation changes do not alter that experiment source fingerprint.
- Local environment: Windows 11, Python 3.12.4, approximately 32 GiB RAM, PyTorch 2.6.0+cu126, **RTX 4050 Laptop GPU**. Training used 4 CPU threads; MLP used CUDA. This is not an RTX 4090D laboratory measurement.

## Validation operating point

The threshold for each model was chosen using the **same validation partition**, at an empirical normal-FPR budget of 1%. These numbers are exploratory; independent calibration is still needed for final evaluation. AP uses scores over the full validation partition.

| Model | Attack recall | Normal FPR | AP | Macro-F1 | Serialized bundle bytes |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 77.7232% | 1.0000% | 0.992281 | 0.837837 | 8,192 |
| Random forest | 87.5824% | 0.9881% | 0.996205 | 0.904873 | 29,208,641 |
| Histogram gradient boosting | 88.7666% | 1.0000% | 0.996706 | 0.913144 | 792,248 |
| MLP (64, 32) | 85.0464% | 1.0000% | 0.995450 | 0.887247 | 68,236 |

Bundle size includes preprocessing and model state; it is not runtime RAM, GPU memory or parameter count. The MLP executed its 60-epoch budget and uses the best validation-loss checkpoint. One seed cannot establish a robust ranking or significant improvement.

## Preliminary CPU measurements

All four saved models were reloaded on CPU with one inference thread. Timing includes DataFrame preprocessing and prediction; excludes file loading, row slicing, thread configuration, networking and UI. Each model had 5 warmups per mode, 30 blocks of 100 individual rows (3,000 observations), and 30 repeated batches of 256 rows. Raw timings and RSS snapshots are retained locally.

| Model | Single-row p50 (ms) | Single-row p95 (ms) | Batch-256 throughput (rows/s) |
|---|---:|---:|---:|
| Logistic regression | 2.888 | 3.440 | 75,055 |
| Random forest | 5.521 | 6.673 | 30,861 |
| Histogram gradient boosting | 3.627 | 4.175 | 35,533 |
| MLP | 3.213 | 5.311 | 70,299 |

This was a shared interactive workstation with browser/development activity, not an isolated performance laboratory. These figures validate the measurement path and must be repeated under controlled conditions. Batch throughput must not be inverted and presented as single-row latency. RSS snapshots are not peak memory.

## Data quality findings

- Original training source: 175,341 rows and 101,040 distinct feature groups; 74,301 duplicate extra rows.
- Original testing source: 82,332 rows and 53,946 distinct feature groups; 28,386 duplicate extra rows.
- Across the two source files: 1,302 shared feature groups, affecting 10,279 training-source rows and 8,541 testing-source rows.
- Some identical feature groups have inconsistent attack categories or binary labels. The audit records them; the pipeline does not silently relabel or delete them.

Exact duplicates stay in the same train/validation/calibration partition. The original test file remains intact, so the official-style evaluation retains a known overlap limitation. A separately specified non-overlap sensitivity evaluation would be a different protocol and must be reported separately. Near-duplicate, temporal and cross-dataset leakage are not resolved by the exact-hash grouping.

## Engineering verification

- 111 local automated tests passed, including preprocessing isolation, ties at the FPR threshold, tampered artifacts, frozen-evaluation gating, unlabeled CSV prediction, and interrupted-run recovery.
- `ruff check src tests scripts` passed.
- CPU synthetic end-to-end suite: all four models trained; report generated with synthetic status visible.
- Full-data development pilot: all four models trained; all run artifact hashes verified; a second `--resume` invocation verified and reused every run.
- Chromium checks: model selection, error filtering, evidence inspection, 390-pixel responsive layout without horizontal overflow, and no console errors. The actual report contains mirror/validation labels and no synthetic IPs or timestamps.
- Windows and Ubuntu CI both passed for implementation/test commit `d2228b9`; see [the successful run](https://github.com/zhoumeng-creater/tracelight-ids/actions/runs/40426309451). This validation document was added afterward without changing the experiment source.
- PowerShell/Bash setup syntax and launcher failure handling were checked. The RTX 4090D setup scripts have **not** been executed on the laboratory machine.

Next: laboratory environment preflight; the 3-seed development suite; field-budget experiments; then a written model-selection/final-evaluation decision. The CLI never automatically opens final test performance at the end of training.
