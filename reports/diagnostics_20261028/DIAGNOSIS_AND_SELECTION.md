# Evidence-led LightSAKT diagnosis — 28 October 2026

## Established findings

All five prior seed42 pilots were verified through configuration/terminal records
and checkpoint hashes. Their complete learning curves are in architecture_curves.csv;
original artifacts were not changed. The original one-layer linear model remains
best among those architectures. No completed training was rerun.

CPU evaluation of the original selected checkpoint: train AUC0.765195061/BCE0.530058344
versus validation AUC0.750803522/BCE0.542363721. GPU-recorded validation AUC0.750803537
is reproduced within1.5e-8. There is modest generalization error,not evidence of
catastrophic overfit. Online training BCE decreased0.630594 to0.535854 by selected
epoch24;validation BCE improved to0.542364. By epoch29 online BCE0.532162 while
validation BCE worsened0.543553,consistent with early stopping being useful.
Online dropout-active BCE and eval-mode train BCE are distinct measurements.

All36 original grid configurations were reviewed. Width32 best AUC0.747109 versus
64 best0.750804 and128 best0.750732. Two/four heads,dropout0.1/0.2,learning rates
0.0005/0.001/0.005 were already tried. No repeat of these settings is warranted.
Marginal means in diagnostics.json are descriptive,not isolated causal effects.
Depth,softmax,recency and gating tests also provide no baseline-relative gain.
Capacity/ordinary optimizer settings are not the strongest current lead.

## Where predictions fail

Validation target positive rate66.6521%;negative recall42.8244% versus positive
recall88.4153% at0.5.18.7973% of predictions lie in0.4..0.6;0.6046% are confidently
wrong (>=0.9 on incorrect or<=0.1 on correct). High F1 partly reflects class
imbalance. Reweighting BCE is not selected:changing threshold cannot improve AUC,
and the ranking/representation issue has a more direct candidate.

The first position in each context has AUC0.718817 (567targets);contexts51..100
AUC0.735890 (10,020targets),versus positions6..10 AUC0.768134. Student-history>200
has AUC0.723853 (8,177targets). These strata differ in skills/students/class mix;
not causal evidence that longer history itself hurts. Context resets at100,so
full student-history length differs from attention-context length.

Rare skills with<=100 training interactions account for only166 validation
targets and do not explain the bulk of errors. Most-frequent-skill stratum has
AUC0.709488 over4,332targets. Problem IDs never seen in training account for197
validation targets;their BCE0.600704. Problems with>100 training occurrences
have BCE0.664333 on627targets,so difficulty is not solely a cold-start issue.
Group AUCs are descriptive and should not be ranked without sample/class context.

Examples with>=50 validation targets and poor BCE include Translations
(64targets,BCE0.741724),Mean(56,0.724343),Mean~~Stem and Leaf Plot(123,0.709417)
and Write Linear Equation from Graph(836,0.669445). Aggregate skills and counts,
not student-level records,are in validation_skill_errors.csv.

The processed model q is composite SKILL,not problem_id. Raw problem metadata
was joined one-to-one via existing permitted student/order keys,without reading
raw correctness for validation or analyzing the test partition. Thus individual
exercise differences within a skill are unavailable to all five prior models.

## Does history matter?

Yes. Zeroing only the selected model's attention branch gives validation
AUC0.688005 versus0.750804. Flipping observed response bits gives0.418812.
These are checkpoint perturbations,not retrained ablations or performance claims.
A whole-chunk response shuffle gives0.728202,but can move future responses into
earlier positions;it is deliberately noncausal corruption and must NOT be treated
as a valid alternate prediction protocol. Zero-attention and bit-flip checks
already establish strong historical-response sensitivity without that ambiguity.

Attention at context51..100 has about62.3 effective keys;target-skill mass0.1847
versus uniform expectation0.1502. It is broad with modest same-skill preference.
This may limit extraction of explicit mastery/count information,but does not
prove an attention defect. Joint skill-response embedding geometry has weak
cross-skill alignment;that observation alone does not establish a useful change.

## Two independent candidate directions

1. **Selected: known-question difficulty correction.** A train-fitted logistic
probe combining baseline logits with problem-vs-skill log-odds gives validation
AUC0.780202540,+0.029399018. Full current training-student histories are removed
from BOTH rates for each training feature;validation uses training labels only.
10 pseudo-observations toward skill rate were fixed,no smoothing search. Fits
on even/odd training-student subsets give AUC0.780208365/0.780200260. Exploratory
paired student-bootstrap300 delta interval [0.021937716885612545, 0.037223919921824486].
This interval conditions on the selected checkpoint/features and is not an
independent confirmation after model selection.

2. **Deferred: explicit same-skill success/failure readout.** A training-fitted
baseline-logit plus log1p within-context correct/incorrect counts probe gives
AUC0.751670066,+0.000866543. Even/odd training-student fits remain positive;
exploratory bootstrap interval [0.00035167305520300286, 0.0013787344203740335].
This is weaker than the question feature and is not the authorized GPU pilot.
The probe is not a new neural result or a complete PFA reproduction.

Neither probe uses validation labels to fit coefficients/features. Baseline train
logits are nevertheless in-sample;all analysis reused the existing validation
split and is exploratory. No hidden test partition has been accessed or used.

Selected scientific variable: add ONE learned scale on a fixed known-problem
relative-difficulty feature to the original logit. Start scale0 for baseline
identity;keep attention,depth,embedding width,heads,dropout,Adam,split,budget and
checkpoint selection fixed. This requires an optional input adapter,not a change
to original processed data. Inference feature depends on known item identity and
training-derived statistics,not a validation response. No test adapter is supplied.
This is a richer-input representation comparison;matched-input SAKT and other
controls will be needed before claims about model superiority. Historical selected
SAKT seed42 validation AUC0.749442606 uses the older skill-only input.

Literature supports the direction without proving this transplant: [AKT](https://arxiv.org/abs/2007.12324) explicitly models within-concept question
differences;[PFA](https://files.eric.ed.gov/fulltext/ED506305.pdf) separates prior
correct/incorrect counts for skill mastery. Our normalized empirical difficulty
feature and two-parameter diagnostic count probe are different models.

All input/protocol/config hashes,feature construction,CPU tests,authorization
and final pilot outcomes are recorded under reports/architecture_study/
question_pilot_20261028/. No attention/kernel or optimization hypothesis is
promoted from diagnostic correlations alone.
