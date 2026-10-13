# Reproducing the corrected protocol

Run from `/workspace/LightSAKT2`. Python 3.10 and the installed PyTorch 2.4 development CUDA build were used; every run records full package versions and hardware in `results/raw/<id>/environment.json`. The GPU is an H100 80GB with a roughly 40GB MIG allocation; nvidia-smi reports driver 535.161.08. Raw files are excluded from Git; publisher references and exact SHA256 values are in `reports/*_statistics.json`.

```
python -m scripts.preprocess
python -m pytest -q
python -m scripts.make_schedule
python -m scripts.run --config configs/smoke.json
python -m scripts.run --config configs/first_full_run.json
python -m scripts.schedule
python -m scripts.summarize
python -m scripts.analyze
```

`make_schedule` creates the initial manifest and must not be rerun over an active schedule. `schedule` resumes interrupted epoch checkpoints and skips completed experiments. A failed run requires inspecting its log before `--retry-failed`; do not alter its existing config. Single-run continuation: `python -m scripts.run --config results/raw/<id>/config.json`.

Split seed is always 42, with 70/10/20 student assignments shared by every model. Training seeds are 42,123,2026,3407,7777; seed 42 is a new standardized choice, not a recovered historical seed. Python, NumPy, CPU/CUDA PyTorch, and DataLoader generators are seeded. A worker initializer is supplied; current loaders use zero workers. No label-derived normalization is used. The vocabulary is fitted on training-student identifiers and heldout unseen identifiers map to 0 (UNK). All target identities remain traceable to students and chronological indices.

ASSIST2009 collapses all simultaneous skill tags for one order_id into one sorted composite category to prevent repeated identical response labels becoming adjacent targets. Missing required fields and nonbinary labels are excluded using fixed rules, not statistics fitted to heldout labels. ASSIST2015 uses log_id as order and sequence_id as skill-builder ID. Algebra uses the available original training subset, not the official unlabelled test portion; KC(Default) composite strings are retained. First Transaction Time orders first-attempt response availability, with stable source-row ties. Sorting by Step Start Time can expose responses unavailable at the next step start (2,327 adjacent raw records in the audit); the response-event objective avoids that error.

Lengths: 100/100/200 for ASSIST2009/ASSIST2015/Algebra. All n-1 targets are covered using stride-L chunks and one shared history interaction at boundaries. State/context resets each chunk; this is not uninterrupted full-history inference. Sequence and split checksums bind runs to data. Raw source acquisition and licensing limitations are in DATA_PROVENANCE.md.

Adam: beta1 .9, beta2 .999, eps 1e-8, weight decay 0; maximum 50 epochs, patience 5, strict validation-AUC improvement. No threshold tuning: correctness threshold .5. Metrics aggregate all valid individual targets, never unweighted batch AUCs. BCE is evaluated on probabilities; DKT+ training objective additionally uses its original reconstruction and waviness coefficients. Undefined validation AUC fails explicitly, rather than selecting on test data.

Float32 only; AMP and TF32 disabled. Strict deterministic algorithms enabled. This build rejects CUDA cumsum, so the corrected kernel uses a work-efficient deterministic tree prefix scan on CUDA; CPU uses cumsum. Floating point addition order differs across devices: CPU/CUDA agreement is tested with atol=2e-6, rtol=2e-5; dense float64 math and gradient agreement with 1e-10. Bitwise repeatability is verified on the current GPU/software combination, not promised across versions or hardware.

Checkpoints: best.pt holds validation-selected weights; latest.pt holds optimizer, epoch, RNG states, DataLoader generator, selection metadata and history. Writes use temporary files and atomic replacement. An interruption inside an epoch replays that epoch from the previous completed checkpoint. Source fingerprint mismatches block resume. Exact CPU interrupted-resume equivalence is tested; all model families have executed CUDA repeatability/backward checks.

Tuning budget: 36 LightSAKT configurations per dataset and 36 for each baseline variant; PAM and MHA are separate. GKT batch 8 versus 64 for other models is disclosed as a memory constraint. Max epochs and early stopping are the same, but wall-clock budgets are not guaranteed equal. Historical simplified GKT lacks canonical gates and batch normalization and must carry that label. Tuning sees validation only; seed-42 final evaluation reuses the chosen tuning checkpoint after freezing selection. Four other seeds train from scratch. Default ablation reuses final LightSAKT runs, with four distinct component-removal variants; no_regularization is user-approved as equivalent to no_dropout and reuses its evidence.

CSV tables are generated from run JSON files. Empty numerical cells are pending results, never zeros. Student split sampling uncertainty is not captured by seed confidence intervals. Five seeds are too few for strong Wilcoxon claims: two-sided exact minimum p is .0625 before multiple-comparison correction for five nonzero pairs.

The scheduler is a process on this machine, not a hosted service. It can survive a terminal disconnect when launched with nohup, but does not restart after machine loss. Latest checkpoints and the durable workspace allow manual continuation. Do not claim automatic future completion after an environment shutdown.

Runtime limitation: training_seconds aggregates checkpointed epoch segments and resumed completed work. Compute discarded by an interruption within an epoch is not included in that aggregate; failed/interrupted logs are retained. Controlled efficiency timing is separate and must be run after training contention stops.
