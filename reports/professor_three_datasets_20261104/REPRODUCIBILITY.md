# Reproducibility and preservation

Research branch: `research/architecture-variants-20261025`.
Preparation commit: `5f1b830d3aff3cce146fb5cce215d1e93283cac7`.
Original result anchor: `c286bcbc`; October31 report release: `2103a6e3`.

All three native model implementations and prior artifacts remain unchanged.
The first-session data policy, seeds, fifteen configurations and resource
ceilings were committed before the new comparison. See
`experiments/ednet_replacement_20261104/protocol.json` and `plan.json` in the
research repository, plus their source snapshots in the portable archive.

## Exact data reconstruction

Use an isolated copy of this research revision, not the live experiment
directories. Obtain the two actual public files listed in DATASET_AND_SPLITS.md,
verify their SHA256 and place them under
`.tools/ednet_replacement_20261104/raw/EdNet-KT1.zip` and `EdNet-Contents.zip`.
Extract ONLY `contents/questions.csv` to that raw directory. Check its hash.
With the recorded Python/NumPy/pandas environment, run from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.ednet_replacement_data
```

The deterministic population, split, processed CSV and feature sidecars must
match dataset_statistics.json and feature_specification.json. Scripts refuse
to silently replace completed preprocessing. On another machine the locator
paths in manifests must be explicitly relocated, leaving payload hashes,
scientific hyperparameters, selection and seeds unchanged. Absolute original
paths in provenance are locators, not prerequisites for reading result tables.

The published preprocessing does not tune or inspect model scores. Its row
integrity and actual label-perturbation checks are implemented in
`scripts/prepare_ednet_campaign_20261104.py:audit_data`; invoke the audit function
alone for a read-only scientific check (it writes audit output, not source data).
Do not regenerate the committed frozen manifest or invoke training on the live
completed campaign. No completed expensive experiment needs repetition here.

## Checkpoint reproduction and continuation

`checkpoint_references.csv/JSON` provides exact best.pt paths and SHA256 and
configuration copies for each measured seed. Original data/split fingerprints
are included. Each latest.pt additionally preserves Adam moments, contiguous
epoch history, early-stopping counters, Python/NumPy/Torch/CUDA RNG, loader
generator state, source/environment and elapsed runtime. Model checkpoints and
student-level predictions are backed up locally, not shipped in this reporting
archive. Refer to checkpoint_backups.json for new local recovery archives.

For independent CPU restoration at the recorded location:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.replay_ednet_campaign_20261104
```

It verifies hashes/configurations, exact seeded shuffle state and Adam counts,
restores native best checkpoints strictly and caches completed results. Existing
valid cache entries are hash-verified instead of repeating inference. Remove
no historical cache; use a separate explicit inference destination if a fresh
hardware reproduction is desired. Existing datasets' original CPU replay was
already completed; its saved predictions are independently recalculated here.

The recorded campaign launcher is deliberately single-use and refuses an
existing campaign status or experiment directory. Do NOT rerun it. Any future
replication must use separate run IDs, unambiguous source/config/data binding,
defined GPU budget, current shared allocation and admission, preserving prior
files. A partial run may only resume exact latest state within applicable
remaining budget; never reset it or silently reuse a new scientific protocol.

Core configuration for all replacement models: d64,2heads,dropout0.1,L100,b64,
Adam lr0.001,betas(0.9,0.999),eps1e-8,decay0,max50,patience5,strict validation AUC.
No AMP/TF32; deterministic training algorithms and recorded seeds. The original
trainer sets Torch CPU threads to4; CPU replay uses1. Source/environment hashes
and package versions are saved with each run. Exact floating-point portability
across hardware is not guaranteed; recorded CPU/GPU tolerances are explicit.

Test inference occurs only after validation selection on fixed configurations.
No test-guided tuning or checkpoint choice is allowed. Threshold0.5 is fixed.
Summary SD is sample SD over the five training seeds, not split uncertainty.

## Feature-matched control

Five separate SAKTQuestion configurations are under `prepared_controls/`.
The unchanged existing `scripts/feature_matched_sakt.py:SAKTQuestion` class is
adapted to the generic frozen EdNet sidecar by
`scripts/ednet_feature_matched_control.py:prepared_control`. Its CPU forward,
backward and refusal of test selection passed. This is prepared, not GPU-trained.
The core fifteen windows do not include the control; define its budget before
execution. Comparing identical item information across native encoders remains
necessary to isolate information effects from attention effects.

## Results regeneration and portable package

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.report_three_datasets_20261104
python scripts/verify_three_dataset_package.py /path/to/extracted/package
```

The report builder rechecks old checkpoints/prediction hashes and independently
recalculates old masked metrics without training or inference. It includes only
actual completed new replays and labels missing slots PENDING. Graphs show actual
values. The standalone package verifier uses the standard library to check
payload hashes, means/sample SD, matched deltas, CSV/JSON equality and checkpoint
config references. It does not fetch data or run models.

The reporting archive excludes raw datasets, student-level arrays, model weights,
credentials and transient download URLs. It includes report/CSV/JSON/graphs,
source/config snapshots, integrity evidence and explicit checkpoint references.
It is portable for reading and aggregation verification, not self-contained
training/inference. Git bundles are local, prerequisite-bound incremental backups;
their prerequisites are recorded and preserved. A remote push is only recorded
if actually verified. HiTSKT remains on hold and outside every action here.
