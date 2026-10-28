# Evidence-led question representation pilot — 28 October 2026

**Promising partial endpoint; training interrupted by resource guard.**
Best saved validation AUC0.778424995882,delta+0.027621458771 over original
(+2.762146 percentage points). Validation F10.823962902619,
accuracy0.744208916084,BCE0.519885034365,27,456targets.
This is not a completed early-stopped or50epoch-budget experiment. Best/latest
are epoch18,bad_epochs0. The host-memory guard stopped the invocation before
completion;no retry/resume followed and the exact grant is closed_interrupted.
No completed-run metrics.json was created or fabricated.

| Model/input variant | Validation AUC | F1 | Epochs completed | Status |
|---|---:|---:|---:|---|
| Original one-layer linear | 0.750803537111 | 0.814805489110 | 29 | completed |
| Two-layer linear | 0.750790932280 | 0.816262117682 | 29 | completed |
| Causal softmax | 0.745639378878 | 0.812546863284 | 21 | completed |
| Recency softmax | 0.748237788473 | 0.813433394001 | 21 | completed |
| Gated linear | 0.750645233679 | 0.815458840084 | 29 | completed |
| Question-difficulty representation | 0.778424995882 | 0.823962902619 | 18 | interrupted, provisional |

## Why this experiment was selected

See reports/diagnostics_20261028/DIAGNOSIS_AND_SELECTION.md and aggregate CSV/JSON.
Baseline evaluation-mode train AUC0.765195 versus validation0.750804 implies a
modest gap. Existing36configuration grid already covered width32/64/128,
heads2/4,dropout0.1/0.2,lr0.0005/0.001/0.005;none beat selected baseline.
Removing attention drops checkpoint AUC to0.688005,so historical interaction
learning is active. Failed depth/attention/gating changes do not establish that
more capacity is the missing ingredient.

The actual q input is a composite skill,not individual problem ID. Within-skill
exercise difficulty is missing. Most errors are not confined to rare/unseen
skills;frequent-problem and frequent-skill groups also have poor discrimination.
Correct-class prevalence66.65%,negative recall42.82%,uncertain0.4..0.6 predictions
18.80% explain why F1 alone is insufficient. History/frequency strata differ in
cohort/skill/class composition;diagnostics are not causal estimates of length
or rarity effects. Full learning curves and skill-level error aggregates are saved.

A training-fitted frozen-baseline+question-feature probe gave AUC0.780203;
training-student-subset fits were stable. The weaker independent practice-count
probe gave0.751670 and was deferred. All probes reused the existing validation
split and baseline train logits were in-sample;they guided this one pilot and
were not counted as independent experimental confirmation.

## The one intended variable

Add alpha*[logit(problem_rate)-logit(skill_rate)] to original LightSAKT logit,
with one new scalar initialized0. Training rows exclude ALL labels from their
own student when computing BOTH rates;validation rows use only training labels.
Skill smoothing(+1)/(n+2),question smoothing10 observations toward skill rate,
fixed before the run with no sweep. Unseen questions fall back to skill rate.
Selected alpha=0.790180266. Features match diagnostic
values within1e-6. CPU tests confirm validation-label invariance and own-student
exclusion. Raw IDs joined only via existing train/validation student/order keys;
raw correctness was not needed for the join and no test adapter is implemented.

This uses additional known problem-ID information and training-derived per-problem
statistics;one new neural scalar does not mean only one extra statistical degree
of freedom. Therefore the gain supports
an input-representation hypothesis,NOT superiority of the original attention
kernel over matched-input competitors. Original attention,one block,embeddings,
heads,dropout,residuals/FFN/head stay the same;original model source preserved.
Dedicated six-field dataset/loss adapter restores bindings on exit;original
training engine source unchanged. Baseline init/RNG/initial predictions match.

Same ASSIST2009 splits/vocabulary,seed42,L100,d64,2heads,dropout0.1,batch64,
Adam lr0.001,betas(0.9,0.999),epsilon1e-8,weight decay0,float32/no AMP/TF32,
maximum50epochs,patience5,strict validation-AUC checkpoint rule. Only18 epochs
were reached. Test inference,predictions and selection remain untouched.
93 CPU tests passed before launch,including causality/padding/finite gradients,
feature leakage exclusions,reporting,exact resume and test-dataset refusal.

## Verification, resources and interruption

Independent CPU replay PASS: AUC absolute difference
2.08886883346e-08;F1/accuracy/precision/recall exactly match.
Best and latest models/Adam states finite;config/data/feature/source/grant hashes
and full optimizer/RNG/loader/patience/history state checked. All180 protected
prior file hashes unchanged. No optimizer update during verification.
Reproduce with `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.verify_question_pilot --partial`.

Saved training elapsed214.204s at epoch18;
invocation start-to-stop223.965s.
Maximum recorded completed-epoch GPU allocated
254.627MiB,
reserved368.000MiB.
These are saved epoch peaks,not a reconstructed unsaved-epoch peak. GPU4GiB
and900s caps were respected. Stop reason: `PilotPause: Host memory headroom floor breached`. Numeric failing
host sample was not written;guard reports below the required2GiB host headroom.
Successful telemetry/after-exit snapshot preserved without inventing ownership
or an exact failed value. Existing peer GPU/HiTSKT workloads were not changed.
No policy relaxation or automatic retry after the breach.

Execution commit `b24e154a`.
Run `ASSIST2009-LightSAKT-pilot-one_layer_question_difficulty-42-v1`.
Best SHA256 `80a2ccc0b7f81d99f459bec3a1b8094f934d4d0df9b89123f78939643323eb03`.
Latest SHA256 `e134577a436c125274e00629b510739482b77b443b3db8d49d7c79092aca2c24`.
Feature SHA256 `6231fc5b51317ab921be89647b5101f654dbf8fdb5d841027073df0f29e1e464`.
Training source SHA256 `cf083a73dc0dce9a875c296b4450c7b6f591bb2df6124566dc50078f3113adc0`.
Checkpoints remain under pilot-workspace/checkpoints/<run_id>/.
Persistent read-back-verified archive `/workspace/LightSAKT2/backups/question_pilot_20261028/interrupted_epoch18_checkpoints_features_records.tar.gz` includes both checkpoints,
raw interruption records,config,executed grant AND the required local feature
sidecar;hashes in recovery_archive_20261028.json. Binaries/features/datasets are
not stored in memory or Git. Training/validation curves exported as PNG/PDF.

## Next action and scope

First finish this SAME frozen run from latest.pt epoch18 under newly authorized
bounded execution and restored resource headroom. Recheck exact source,adapter,
config/data/feature hashes and preserve interruption history. Do not reset weights,
change smoothing/optimizer or label the current run completed. No resume under
the closed grant and no new independent pilot in this task.

Then recommend paired seeds42,123,2026,3407,7777 on ASSIST2009. Include a SAKT
control using the same known-question feature before claiming model superiority.
Algebra2005 has Problem Name/Step Name metadata,requiring verified semantics and
mapping before an analogous frozen comparison. The current ASSIST2015 release
has only user_id,log_id,sequence_id,correct;there is NO separate problem ID,so
this exact feature cannot be replicated there without independently verified
additional metadata. Do not silently reinterpret sequence_id or change datasets.
Practice-count residual is a deferred independent feature candidate when item
identity is unavailable,not an equivalent replication of this pilot.

All gains are conditional on one seed and a reused validation split. Bootstrap
intervals are descriptive,not fresh confirmation. No general improvement or
final test claim. Missing Git transport auth remains;local commits/bundles are
preserved without force push or credentials exposure.

Professor update: Diagnostics identified missing question-level difficulty as
a stronger bottleneck than encoder depth or attention type. A one-scalar,
training-derived difficulty correction reached validation AUC0.778425 versus
0.750804(+0.027621),with independent checkpoint replay. The shared-host memory
guard interrupted training after18epochs,so the pilot is provisional and must
be completed before seed replication and matched-input competitor comparisons.
