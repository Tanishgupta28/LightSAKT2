# LightSAKT architecture study — prepared 25 October 2026

LightSAKT has **one encoder block**, consisting of one multi-head attention
operation followed by one feed-forward network (FFN). This is true in both
`original_snapshot/train/models/light_sakt.py` and the corrected baseline in
`src/models/light_sakt.py`. Neither contains a loop or list of stacked encoder
blocks. Multiple heads, Q/K/V projections, and prefix-sum operations are internal
parts of this single block. The existing five-seed corrected baseline is the
comparison target. Historical snapshot results remain preserved and unverified;
its previously documented attention leakage is not restored.

The original checkout remains on `research-rebuild` at `a6bdc6c0`. Work is isolated
on `research/architecture-variants-20261025` in
`/tmp/lightsakt-architecture-20261025`. Existing uncommitted reports, experiment
state, checkpoints, and the active user training hold are retained. Nothing was
submitted to the existing schedule, no process was signalled, no CUDA computation
was performed, and no real train/validation/test dataset was loaded for this study.

An interaction at position t encodes the already observed skill and response;
the query is the skill for position t+1, whose response is the target. In detail:

|Component|Confirmed baseline behavior|
|---|---|
|Embeddings|M has 2K×d interaction parameters; E has K×d next-skill parameters; learned P has L×d positional parameters, added to M|
|Attention|Q=E(next skill), K=V=M(observed skill, response)+P; affine d×d Q/K/V/output projections; heads concatenate back to d|
|Kernel|φ(x)=ELU(x)+1; normalized causal prefix attention, including exactly valid j≤t keys in numerator and denominator, with epsilon 1e-8|
|Residual and first normalization|S=LayerNorm(Dropout(attention output)+M+P+E)|
|FFN and second normalization|d→d/2→d, ReLU, dropout after ReLU and after final projection; F=LayerNorm(FFN(S)+S)|
|Normalization|Two affine post-residual LayerNorm modules, default epsilon 1e-5|
|Prediction|One d→1 affine projection, sigmoid, padding masked to zero; masked binary cross entropy|
|Dropout|One attention-output dropout and two FFN dropouts. The kernel attention module defines a dropout object but does not apply probability dropout. No embedding dropout|
|Depth versus heads|One block; default 2 heads. Selected ASSIST2009 uses 2, selected ASSIST2015 and Algebra2005 use 4|

The kernel's prefix KV and K computations do not add model depth. Its attention
work scales as O(BHLd_head²), excluding projections, and its current differentiable
implementation materializes prefix KV states. Linear scaling with sequence length
does not by itself establish a runtime or memory advantage at L=100 or 200.

The existing dataset settings are confirmed below. K includes the training-fitted
UNK identifier. These are the project's documented local dataset variants, not
new publisher downloads; Algebra uses the available labelled training subset.
Exact identifiers, student counts, hashes and baseline references are pinned in
`experiments/architecture_study/plan.json`.

|Dataset|Clean interactions|Students train/validation/test|K|L|d|Heads|Dropout|Learning rate|
|---|---:|---|---:|---:|---:|---:|---:|---:|
|ASSIST2009|274,590|2,905 / 415 / 831|137|100|64|2|0.1|0.001|
|ASSIST2015|683,801|13,888 / 1,984 / 3,968|101|100|128|4|0.1|0.0005|
|Algebra2005|607,025|401 / 57 / 116|388|200|128|4|0.2|0.001|

ASSIST2009 uses skill-name composites with simultaneous tags collapsed;
ASSIST2015 uses sequence_id from the four-column skill-builder release;
Algebra uses full KC(Default) composites ordered by First Transaction Time.
Splits are student-disjoint 70/10/20 with split seed 42. Vocabulary is fitted only
on training students. Stride-L chunks share one boundary interaction to retain
all n−1 next-response targets; context resets between chunks.

The stored baseline results below are mean ± sample SD over seeds
42, 123, 2026, 3407, 7777. Validation values come from the selected checkpoint of
each seed. Test values are read from previously published result tables solely
as historical context; this study neither reran test inference nor used those
test scores for architecture selection.

|Dataset|Baseline validation AUC|Previously reported baseline test AUC|
|---|---|---|
|ASSIST2009|0.749454 ± 0.001570|0.751056 ± 0.000766|
|ASSIST2015|0.729922 ± 0.000819|0.726088 ± 0.000315|
|Algebra2005|0.807548 ± 0.000766|0.810590 ± 0.000422|

The baseline settings were selected from 36 validation-only grid configurations
per dataset. Preserve these selected settings for the first architecture study;
it measures architectural effects conditional on those settings. If subsequent
tuning is needed, predefine an equal validation search budget for every arm,
including the baseline, and report that as a separate experiment.

The separate `LightSAKTResearch` factory entry implements the following arms.
The existing `LightSAKT` constructor, source, weights, and state layout are
unchanged. `num_layers` accepts positive integers; `attention_type` accepts
`linear`, `softmax`, or `recency_softmax`. The prepared study varies one factor
at a time, although other depth/attention combinations can be configured later.

|Arm|Layers|Attention|New runs|
|---|---:|---|---:|
|original_one_layer_linear|1|Existing causal ELU+1 baseline|0; 15 completed references|
|two_layer_linear|2|Same causal ELU+1 attention|15|
|one_layer_softmax|1|Causal scaled dot-product softmax|15|
|one_layer_recency_softmax|1|Same softmax plus a learned nonnegative distance penalty per head|15|

The two-layer model shares embeddings and the prediction head and has independent
attention projections, FFNs and two norms per block. The first block is exactly
the baseline block. Each later block receives the preceding hidden states as
K/V, retains E(next skill) as Q, and applies the same residual/norm/FFN pattern.
Padding is zeroed between blocks. At position t, every hidden state read belongs
to j≤t and depends on responses only through j, so stacking cannot introduce the
target response. This is a stack of SAKT-style query-to-history blocks, rather
than an added recurrent refinement with shared parameters. Optional attention
visualization returns the final block's weights.

The one-layer linear research configuration has exactly the baseline's state
keys, initial values, and forward path, including dropout under the same RNG
state. Additional layers initialize after baseline components, preserving first
block initialization. Alternative attention retains the same initial Q/K/V and
output projections. All arms keep attention-output and FFN dropout unchanged;
attention probability dropout remains disabled so it is not a confounding change.

My first recommendation is the **one-layer softmax control**: it isolates the
attention kernel with identical projection shapes and parameter count. More
selective relevance weighting is a plausible AUC hypothesis at these short
sequence lengths. [simpleKT](https://arxiv.org/abs/2302.06881) demonstrates that
ordinary dot-product attention can be competitive in KT, although its embeddings
and other design choices differ; its reported gains cannot be attributed to
this transplant.

The next recommendation is **recency-biased softmax**, compared directly with
plain softmax. It subtracts softplus(a_head)×(t−j) from allowed causal logits;
initial decay is 1/32 per interaction, with 2 or 4 extra learned scalars.
This tests whether a forgetting prior helps.
[AKT](https://arxiv.org/abs/2007.12324) motivates exponential decay and
context-aware relative distance. [ALiBi](https://arxiv.org/abs/2108.12409)
motivates linear distance penalties, and
[stableKT](https://www.ijcai.org/proceedings/2024/0654) studies length
generalization in KT. Our option retains learned absolute positions and uses
learned slopes; it is an isolated bias experiment, not a reproduction of any of
these full models. Positional-capacity extrapolation is outside this study.

If recency helps, a later **decayed linear-attention** arm could use
S_t=λS_(t−1)+φ(k_t)v_tᵀ and z_t=λz_(t−1)+φ(k_t), with a positive learned λ<1.
This would retain linear work in L, but needs a separate deterministic scan,
numerical-stability checks and runtime profiling. It is recommended, not
implemented here. [Performer FAVOR+](https://arxiv.org/abs/2009.14794) is a lower
priority option if longer contexts become necessary: random-feature
approximation introduces a new feature-budget/RNG factor when exact softmax is
already practical at L≤200. Local or top-k attention is also deferred because
it adds a window/k search and can discard older relevant skills. These priorities
are research judgments; no variant AUC improvement has yet been measured.

All 45 proposed runs inherit batch size 64; Adam with betas (0.9,0.999), epsilon
1e-8, zero weight decay; the dataset's selected learning rate; maximum 50 epochs;
patience 5; float32 with AMP and TF32 disabled; and the five original seeds.
The shared trainer, loader order, loss, globally pooled metrics and strict
validation-AUC checkpoint rule remain unchanged. No new gradient clipping,
scheduler, augmentation or optimizer tuning is introduced.

The minimum requested comparison is 30 new runs: two-layer linear and one-layer
softmax across three datasets and five seeds. The recency arm adds 15. Seed 42
can be a first diagnostic pass (6 required or 9 with recency), but all five seeds
must finish before ranking a prespecified arm. Select per dataset using mean
validation AUC, equivalently mean paired delta against that dataset's baseline;
resolve exact ties by fewer parameters, then lexical arm name. Report paired
seed deltas and sample SD, never the best seed. Five seeds on one student split
measure initialization variability, not uncertainty over new student cohorts.

Every new config has `evaluate_test=false`. Before later execution, verify
processed/split hashes against the plan as well as the standard trainer's
statistics checks. Keep test access locked while choosing architecture and
settings. Freeze that decision before any separately authorized final test
evaluation; then report all prespecified seeds, including negative outcomes.
The existing baseline test has already been exposed historically, so this is
continued development on a fixed benchmark, not a claim of a never-seen new test.

`validation_comparison.csv` currently contains completed baseline validation
references and pending cells for every new arm. The comparison script requires
all seeds before displaying an arm mean, verifies matching configs/data hashes,
and rejects new results containing test metrics. New performance claims also
require checkpoint replay and artifact verification after actual training.

CPU validation passed **95 tests** with **16 skips** (15 CUDA checks and the real
processed-dataset test unavailable in the isolated worktree). The historical
archive checksum test was run separately in the original checkout and passed;
only that test was deselected in the worktree run. Evidence is in `cpu_tests.xml`
and `original_snapshot_tests.xml`. Tests cover dense softmax output/gradient
references, future skill/response/query perturbation, invalid indices, internal
padding, all-padding examples, batch independence, visualization invariance,
independent sequential blocks, finite gradients/Adam updates, serialization, and
exact synthetic resume. Guarded synthetic training asserts that no test loader
is constructed and that checkpoint selection uses only validation AUC.

A separate bounded CPU profile checked all four arms at the three real
configuration widths, using synthetic batch 8, one thread, one warmup and three
measured updates. `cpu_profile.json` records the samples and versions. Its
timings include forward, BCE, backward, finite-gradient checks and Adam; they
are diagnostic timings, not isolated production throughput measurements.

|Dataset|Baseline parameters|2-layer parameters|Softmax parameters|Recency parameters|CPU step ms: baseline / 2-layer / softmax / recency|
|---|---:|---:|---:|---:|---|
|ASSIST2009|53,857|74,945|53,857|53,859|26.2 / 47.9 / 6.2 / 6.4|
|ASSIST2015|134,849|217,985|134,849|134,853|52.0 / 95.8 / 11.2 / 11.1|
|Algebra2005|257,857|340,993|257,857|257,861|121.9 / 220.3 / 27.4 / 28.2|

Recorded baseline GPU training averaged approximately 24.8, 74.5 and 32.9 seconds
per selected run, respectively, on an H100 3g.40gb MIG in the historical
environment. Actual selected baseline epoch counts were 25–34, 16–23 and 28–38.
Peak allocated memory was approximately 255, 441 and 818 MiB. These records are
read from completed experiments; no current GPU allocation was queried or used.

`training_estimates.json` uses the saved epoch timings, following seed 42's
physical selected-grid history where its logical final run has no history file.
It applies explicit, unmeasured runtime-factor scenarios relative to the
baseline: 1.5–3× for two-layer linear, 0.5–2× for softmax, 0.5–2.5× for recency
softmax. At comparable early-stop counts, the full 45-run study is approximately
**28–83 serial GPU minutes**; at 50 epochs for every run, approximately
**61–184 minutes**. Startup, dataset loading, verification, final heldout
inference, queueing and contention are additional. CPU speed ratios are not
used as GPU speed estimates. A later bounded profile on an authorized dedicated
allocation should measure actual peaks and step times; provision at least
4 GiB initially. That is a planning allowance, not a measured variant peak.

The training hold remains active. The plan and configs are separate from
`experiments/manifest.json` and cannot launch themselves. The following commands
only prepare reports or run CPU checks; they do not release the hold or train
on the real data:

```bash
cd /tmp/lightsakt-architecture-20261025
python -m scripts.architecture_study
python -m scripts.compare_architecture_study
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.benchmark_architecture_study
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/test_architecture_study.py tests/test_correctness.py tests/test_training.py tests/test_user_training_hold.py -k 'not original_snapshot_is_unchanged' --basetemp=/tmp/lightsakt-architecture-pytest-reproduce
```

Original-checkout preservation evidence is in `preservation_before.json` and
`preservation_verification.json`: selected-checkpoint/content hashes, the full
checkpoint file-metadata inventory, unchanged hold and manifest, and checks
that original model/attention/trainer/data source bytes remain unchanged. No
baseline artifact is copied back, overwritten or re-labelled by this study.
