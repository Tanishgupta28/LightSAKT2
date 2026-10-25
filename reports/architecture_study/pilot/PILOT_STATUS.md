# ASSIST2009 seed-42 architecture pilot

The depth-stage launch was attempted and **blocked before PyTorch/CUDA import**
by the persistent user training hold. Both the original checkout and the research
worktree still contain `experiments/USER_TRAINING_HOLD.json`. The current request
allows training conditional on an already lifted hold and GPU authorization; it
does not lift that hold. Neither marker was removed or modified. No GPU
authorization record for this pilot has been supplied or inferred from free
memory. No current GPU probe, training run, scheduling action, process signal,
automatic retry or test evaluation was performed.

The executable launcher is `scripts/architecture_pilot.py`. Its default depth
stage runs a fresh original LightSAKT baseline followed by the two-layer linear
variant. It cannot dispatch the 45-run study. Each invocation permits at most
two sequential runs on one dataset and one seed. A later attention stage runs
causal softmax and recency softmax, after the depth pair is complete and baseline
retraining is within the prespecified tolerance.

ASSIST2009, seed 42 was chosen before examining new variant outcomes: it uses the
smallest selected embedding width (64), L=100 and two heads, and has the shortest
recorded selected baseline training time. This gives a low-cost depth screening
case with 198,942 training interactions and 27,456 validation next-response
targets. It does not establish that improvements will transfer to ASSIST2015 or
Algebra2005. Dataset choice was not based on their test scores.

Actual validation-only inference was completed on CPU using the immutable saved
seed-42 baseline checkpoint. This establishes a verified reference, not a newly
trained baseline or a measured architecture improvement:

|Measurement|CPU checkpoint replay|Saved validation record|
|---|---:|---:|
|Validation AUC|0.750803522191|0.750803537111|
|Validation F1|0.814805489110|0.814805489110|
|Validation BCE|0.542363721270|0.542363721265|
|Validation targets|27,456|27,456|
|Selected epoch|24|24|

AUC absolute difference is 1.49e-8, below the 2e-6 replay tolerance. The replay
took 1.00 seconds inside the measured inference/data-loading scope, excluding
interpreter and library startup. Process lifetime peak RSS was 764,387,328 bytes
(approximately 729 MiB); this is CPU process memory, not training/GPU memory.
Exact versions, source/data/split/checkpoint hashes and metrics are in
`baseline_validation_replay.json`. No test loader or saved test predictions were
accessed. The common CSV loader reads the checksum-pinned processed file and
selects only the validation students for inference.

The fresh comparison is still pending:

|Pilot arm|Status|New validation AUC|
|---|---|---|
|Original one-layer, fresh retraining|Blocked by hold|Unmeasured|
|Two-layer linear|Blocked by hold|Unmeasured|
|Causal softmax|Deferred until depth stage completes|Unmeasured|
|Recency-biased softmax|Deferred until depth stage completes|Unmeasured|

`comparison.json` deliberately leaves these cells null. It never subtracts a new
validation score from an old test score. New deltas will be measured against the
fresh same-seed one-layer validation result. Repeated initialization under the
same seed, independent first/second blocks, masked targets, dropout, split seed
42 and student-disjoint 70/10/20 assignments remain fixed. The selected protocol
uses batch 64, Adam lr=0.001, betas=(0.9,0.999), epsilon=1e-8, zero weight decay,
float32, no AMP/TF32, at most 50 epochs per arm and validation-AUC patience 5.
Both depth arms receive the same budget; no early truncated baseline is compared
with a full-budget variant.

For this screening, a validation AUC increase of at least 0.001 is a promotion
signal, conditional on correct execution, memory bounds and a fresh baseline
within 0.001 of its saved validation result. That retraining tolerance is distinct
from the tighter checkpoint-inference tolerance above. A single-seed increase
does not establish statistical significance. Qualifying two-layer results should
receive the original five-seed evaluation on ASSIST2009, ASSIST2015 and
Algebra2005. Softmax and recency should first complete the separate attention
stage, then follow the same promotion rule. A negative depth result is still a
valid outcome and does not prevent screening the attention alternatives.

Shared-GPU execution requires an existing, scoped authorization identifying the
device, sharing terms, permitted stage and valid time window. The example JSON
under `experiments/architecture_study/pilot/` is explicitly unverified and cannot
authorize execution. The launcher requires the exact authorized UUID in
`CUDA_VISIBLE_DEVICES`, checks both holds, and acquires the original project's
nonblocking admission lock plus a pilot lock. It exits instead of waiting for or
displacing another LightSAKT2 worker. It caps its own PyTorch allocation at 4 GiB,
requires 5 GiB free before starting, and preserves a 1 GiB free-memory floor.
Hold and 15-minute invocation-budget checks occur before each batch; resource
and authorization checks occur at most every two seconds. A breach stops only
the pilot worker. It does not reduce batch size, alter precision, signal other
jobs, or modify their allocations. This bounds memory use and execution scope;
allowed compute sharing must come from the actual authorization.

The common trainer gained an optional observer; its default optimization and
evaluation path is unchanged. New pilot histories persist each epoch's validation
AUC/F1/BCE, training BCE, runtime, CUDA allocated/reserved peaks and process RSS.
The epoch CUDA peak is reset each epoch, and the final result retains the maximum
across all saved epochs. RSS is explicitly labelled as a lifetime high-water
mark. Epoch CSVs are exported on completion or cooperative interruption.
`historical_baseline_epochs.csv` exports all 29 existing baseline epochs as
historical evidence; its unavailable per-epoch memory values stay empty.

Pilot `best.pt` and `latest.pt` are written only under the ignored
`pilot-workspace/checkpoints/<new pilot run ID>/`. Latest state retains optimizer,
Python/NumPy/Torch/CUDA RNG, loader RNG, patience and complete-epoch history.
Hold/resource interruption retains the last complete epoch, discarding any
partial epoch on resume. Source or instrumentation changes reject mixed resumes;
renewing an otherwise compatible allocation is audited without changing model
updates. No legacy checkpoint is resumed or overwritten.

CPU verification: **115 passed, 16 skipped**, with the historical archive test
deselected because its ignored binaries are only in the original checkout and
were verified in the earlier stage. The added tests cover both hold locations
before Torch import, malformed/expired/out-of-scope authorizations, nonblocking
lock conflicts, isolated input links, epoch exports, full versus interrupted
resume, unchanged parameters/validation under instrumentation, resource-stop
checkpoint preservation, and fresh validation-pair comparison. GPU behavior has
not been claimed as tested while the hold is active. Evidence: `cpu_tests.xml`.

`preflight.json` verifies all three existing processed/split hashes, student
disjointness, 15 selected checkpoint references and 42 original content hashes.
All match; the original hold, manifest, baseline architecture and results remain
unchanged. The observer exists only on the isolated research branch; the original
checkout's trainer is unchanged as well. No old experiment/recovery state was
reset, re-labelled or continued.

Once the existing hold has actually been lifted through an explicit project
decision and GPU execution is authorized, the bounded launch is:

```bash
cd /tmp/lightsakt-architecture-20261025
CUDA_VISIBLE_DEVICES='<authorized GPU or MIG UUID>' python -m scripts.architecture_pilot run --stage depth --authorization /path/to/existing-authorization.json
```

The same command with `--stage attention` is separate and checks completion of
the depth pair first. CPU-only replay remains available through
`CUDA_VISIBLE_DEVICES='' python -m scripts.architecture_pilot reference`.
No background process is waiting for authorization and nothing will start
automatically. The remaining blocker is the active hold; GPU execution permission
must also be established before a later launch. A new two-layer AUC cannot be
reported until that condition permits actual training.
