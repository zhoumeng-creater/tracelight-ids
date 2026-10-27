# UNSW-NB15 data contract

The supported input is the **official train/test CSV format**, not the four raw
UNSW-NB15 CSVs, PCAP traffic, or CICFlowMeter features. The
[UNSW dataset page](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
states 175,341 training rows and 82,332 testing rows. Its public SharePoint link
redirected to login during the 2026-10-27 check; this project does not log in or
silently substitute a mirror.

The split-file header has 45 columns: **42 predictor fields**, the source `id`,
and two labels (`attack_cat`, `label`). `proto`, `service`, and `state` are
categorical. All other predictor fields are numeric. The full ordered predictor
list is `ids.data.FEATURES`; only these fields may enter a model. `id`, labels,
`sample_id`, and `feature_hash` are metadata and must never become predictors.
The original raw dataset's 49-feature description is not this split schema.

## Obtain source files

Prefer local CSVs obtained from the official page. Academic use is permitted
under its stated conditions; retain the requested original dataset citations.
Dataset files are not included in the source repository.

The standard-library downloader accepts direct official links when available:

```console
python scripts/fetch_unsw.py --source official --output data/raw --train-url OFFICIAL_TRAIN_CSV_URL --test-url OFFICIAL_TEST_CSV_URL
```

An **explicitly selected, unverified third-party mirror** is also available:

```console
python scripts/fetch_unsw.py --source mirror --accept-unverified-mirror --output data/raw
```

This uses [Nir-J/ML-Projects](https://github.com/Nir-J/ML-Projects/tree/44bbd02b9e70f74dd915218d5eb702f125341ffc/UNSW-Network_Packet_Classification)
at fixed commit `44bbd02b9e70f74dd915218d5eb702f125341ffc`. It downloads only the
two split CSVs, caps each download at 64 MiB, checks the header, label consistency
and published row counts, and saves local SHA-256 hashes and source URLs in
`provenance.json`. **These checks do not establish byte identity with the official
source.** Set `data_kind="mirror"` during preparation. The script refuses to
overwrite existing sources. A failed official download never triggers a mirror
fallback.

## Python API

```python
from ids.data import (
    FEATURES, CATEGORICAL, load_unsw_csv, prepare_dataset,
    load_partition, make_synthetic_dataset,
)

manifest = prepare_dataset(
    "data/raw/UNSW_NB15_training-set.csv",
    "data/raw/UNSW_NB15_testing-set.csv",
    "data/prepared/run-01",
    seed=20261027,
    data_kind="official",  # or "mirror" or "synthetic"
)
train = load_partition("data/prepared/run-01", "train")
```

Exact signatures:

- `load_unsw_csv(path, require_labels=True) -> pandas.DataFrame`
- `prepare_dataset(train_csv, test_csv, output_dir, seed=20261027, *, data_kind="official") -> dict`
- `load_partition(prepared_dir, split) -> pandas.DataFrame`
- `make_synthetic_dataset(output_dir, n_train=600, n_test=120, seed=42) -> tuple[Path, Path]`

`official` and `mirror` require the published filenames and row counts so swapped
or truncated splits fail loudly. `official` records user-declared origin; it
does not imply an independent official-checksum comparison. Source metadata
contains basenames, never machine-specific absolute paths.

## Validation and missing values

All 42 predictor columns are mandatory. Unknown columns and duplicate headers
are rejected. Input column order may vary. Numeric text must parse as numbers;
positive or negative infinity is rejected. Blank and conventional numeric
missing tokens are retained as NaN, **not filled here**. Empty categorical values
are retained as NaN; unseen nonempty categories are valid inference inputs and
must be handled by the model's train-fitted encoder. Category strings are trimmed;
`-` remains a real category.

Labeled data must contain both labels, with binary `label` values and a known
attack category; only `Normal` may have label 0. The documented `Backdoors` alias
is normalized to the split-file spelling `Backdoor`. Original IDs, if supplied,
must be finite unique integers. Inference may omit both labels and the source ID:

```python
unlabeled = load_unsw_csv("inference.csv", require_labels=False)
X = unlabeled.loc[:, list(FEATURES)]
```

If either label is supplied, both are required and validated. A CSV's optional
ground truth never changes the permitted predictor fields.

## Identity and partitions

Each `sample_id` is `<first 12 hex digits of full source SHA-256>:<zero-based data row>`.
It is stable under file renaming and uniquely identifies a source row in the
prepared bundle. Full file hashes remain in the manifest, and prefix collisions
within an input pair are rejected. `feature_hash` is SHA-256 of canonical ordered
predictor values only; IDs and labels are excluded. Equivalent missing values
and signed zero are normalized. It detects exact semantic duplicate rows, not
approximate near-duplicates or shared sessions.

The original test CSV is retained independently. Only the original training CSV
is allocated to train/validation/calibration with **target ratios 70/15/15**.
The allocator treats every identical-feature group as indivisible, including
groups containing conflicting labels. A seeded, largest-group-first greedy
procedure balances attack-category support and total rows against the targets.
Exact proportions and class presence cannot be guaranteed with indivisible or
rare groups. Actual sizes, class counts and absent-class warnings are recorded.
This is not a temporal, host, or campaign holdout.

Duplicates and conflicting labels are reported in `audit.json` and retained.
Train/test overlap is audited without silently changing the published test set.
Train-only fitting is still essential: imputation, scaling, category vocabulary,
feature selection and model fitting must not consume validation, calibration or
test data. Hyperparameter decisions use validation; calibration/threshold fitting
uses calibration; final scoring uses test. Keeping all duplicate rows preserves
their multiplicity, which remains a stated limitation for statistical inference.

Preparation writes four Parquet files, `audit.json`, `manifest.json`, and
`manifest.sha256`. The manifest binds source hashes, feature schema, data kind,
seed, method, and each partition's hash and row count. `load_partition` verifies
the manifest checksum, schema, expected filename, partition checksum and row
count before returning data. Checksums detect accidental modification; they are
not a digital signature or proof against an attacker who can replace both files
and checksums. Existing prepared artifacts are never overwritten implicitly.

## Synthetic fixtures

```python
train_path, test_path = make_synthetic_dataset("data/synthetic")
manifest = prepare_dataset(train_path, test_path, "data/prepared/smoke", data_kind="synthetic")
```

Fixtures preserve the CSV schema and class-label relationship, include a
`synthetic_source.json` marker, and produce `synthetic` dataset IDs and manifests.
They exercise software only. Their scores are never intrusion-detection research
results, device acceptance evidence, or real-network performance.
