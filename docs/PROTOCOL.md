# Evaluation and CPU benchmark protocol

This repository begins with a reproducible research baseline. A successful smoke
run establishes that its pipeline works; it does not establish a new detection
method, a deployment guarantee, or completion of the final research study.

## Four data partitions

1. **Train** fits preprocessing, category vocabularies, missing-value rules,
   scaling, feature selection and the classifier. Any resampling, stability
   selection or cross-validation must rebuild those operations inside the
   corresponding training fold.
2. **Validation** chooses model families, hyperparameters, feature budgets,
   early stopping and the predefined selection rule. Development comparisons may
   use provisional thresholds selected on validation; these are development
   estimates, not independent confirmation of an FPR guarantee.
3. **Calibration** is reserved until the final candidate configurations are
   selected. It freezes one operating threshold per fitted candidate. It must
   not subsequently select feature budgets, architectures or a lucky seed.
4. **Test** is evaluated only after models, candidate/seed lists, preprocessing,
   thresholds and reporting rules have been frozen. The runner must require an
   explicit final-evaluation gate and record the frozen artifact identity. The
   low-level metric function cannot determine which partition a caller passed;
   data provenance and the gate are runner responsibilities.

The calibration partition is for decision thresholds. It is not implicitly a
probability-calibration training set. Model outputs are attack scores unless
probability calibration has been separately designed and validated.

An official holdout should remain intact. Identify duplicate records and shared
flow/session/time groups where those metadata exist. Keep official-compatible
results distinct from a predeclared contamination sensitivity analysis. Missing
group metadata cannot be repaired by pretending shuffled row numbers are time.
Separately training on another dataset is a replication study, not direct
cross-domain generalization.

## Threshold definition and ties

An attack decision is `float64(score) >= threshold`. Scores must be finite and
within `[0, 1]`. Cast to float64 before comparing: a float32 scalar comparison can
round away a `nextafter` threshold and accidentally accept a tied score.

For `n` normal calibration records and FPR budget `alpha`, permit at most
`floor(alpha * n)` normal records to be flagged. Sort normal scores descending.
If `k = floor(alpha * n) < n`, set the threshold to the next representable float64
above the score at zero-based index `k`. This excludes all ties at the next
disallowed score. The actual count can be smaller than `k`. With `alpha = 1`,
threshold zero accepts all scores. Calibration without normal records is invalid.

A threshold above one is deliberately permitted: it rejects all scores and can
be necessary at zero false-positive budget when a normal record scores one.
Attack calibration scores do not determine the cutoff. Save the target budget,
normal denominator, permitted and observed false-positive counts, threshold,
calibration source and artifact identity.

The achieved FPR constraint is empirical on calibration only. Test FPR may exceed
the target and must be reported unchanged. Do not move a threshold after seeing
test labels. Interactive threshold controls belong to validation or a distinct
demonstration set. Test ROC/PR curves are descriptive and do not authorize
post-hoc operating-point selection.

## Metrics and uncertainty

Attack is positive, normal is negative. Report `TN, FP, FN, TP` and the underlying
denominators with attack recall `TP/(TP+FN)`, normal FPR `FP/(FP+TN)` and precision
`TP/(TP+FP)`. Undefined ratios are null, not invented zeroes. Macro F1 uses the
fixed labels normal and attack; it is null if either class F1 is undefined.
Average precision is the scikit-learn average-precision definition, not
trapezoidal PR integration. It is null with no true attack. ROC AUC is null unless
both ground-truth classes occur. A score-only metric does not depend on the
selected decision threshold.

If true attack-family labels exist, report the fraction of each family detected
as any attack. This is binary detection recall within a family, not correct
family classification. Normal records do not enter family-recall denominators.
Absent family labels remain explicitly missing.

The initial implementation provides two-sided Wilson 95% intervals for recall,
FPR and precision, together with numerator/denominator, a denominator-under-30
flag and a fewer-than-five-events-or-non-events flag. These are diagnostic flags,
not significance tests. Wilson intervals assume IID Bernoulli observations;
network flows often violate this assumption. They do not quantify training-seed
variance, do not constitute a deployment guarantee, and must not be called
group-aware confidence intervals. With reliable session/time metadata, a later
paired group bootstrap may compare models on shared records. It must declare
the resampling unit and effective group count.

Zero observed false positives does not imply zero population risk. Keep the
non-zero upper interval bound and sample count visible. Multiple training seeds
measure training randomness separately from test-sample uncertainty. Select a
demo model instance by a frozen validation rule, never the best test seed.

## CPU timing contract

`benchmark_predict` measures the supplied CPU prediction callable, which should
include preprocessing and model inference. Input parsing, DataFrame slicing,
thread setup, network requests and UI rendering are outside the measured scope.
It must be labelled preprocessing-plus-prediction, not pure-model or full-file
end-to-end latency. The runner must reject GPU inference for this benchmark and
configure estimator workers and PyTorch threads; the function cannot inspect an
opaque callable to prove which device it uses.

BLAS/OpenMP pools are bounded with threadpoolctl. Each mode receives unmeasured
warmup calls. Batch=1 timings really invoke one record per call, for up to the
configured number of evenly spaced records over repeated blocks. Save all
latencies, block medians, p50 and p95. Separately time a fixed batch capped by the
available row count, and compute throughput as all processed rows divided by the
sum of batch elapsed times. A batch's time divided by its row count is not
single-request latency. Repeated timing observations on the same input are not
independent traffic samples.

Process RSS before and after the benchmark is optional when psutil is available.
These snapshots are not peak memory and cannot be attributed entirely to the
model. Peak RSS, cold-start loading, complete file import and PCAP feature
extraction require separate instrumentation and must not be inferred from these
fields. A later GPU benchmark needs explicit synchronization and a declared
host/device transfer boundary; this CPU implementation does neither.

Freeze hardware/software, thread counts, input rows and batch sizes before
comparing candidates. Stop competing jobs where feasible and retain raw
measurements. Synthetic smoke data validates software behavior only. All final
claims require real audited data and the frozen evaluation procedure above.
