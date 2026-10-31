# Professor-ready research summary — 31 October 2026

We verified the LightSAKT question-difficulty results across five random seeds
on ASSIST2009 and Algebra2005. The seed-42 ASSIST2009 validation AUC change
0.7508 -> 0.7812 reproduces from saved artifacts. Mean held-out test AUC improves
from0.751056 to0.779359 on ASSIST2009 (+0.028303) and from0.810590 to0.836871 on
Algebra2005 (+0.026281). All five matched seeds improve validation and test AUC
on both datasets, versus both original LightSAKT and SAKT. Mean test F1 and
accuracy also improve. Tables report sample standard deviations and all seeds.

ASSIST2015's official original CSV exactly matches the preserved four-column
release. It contains interaction IDs and skill-builder/problem-set IDs, but no
independent question ID or recovered item crosswalk. Its five difficulty-model
results are unavailable under the current protocol; baseline results remain valid.

Leakage and checkpoint/configuration/split/masking/metric checks passed for the
stated offline student-disjoint protocol. No GPU experiments were repeated.
The modified models receive richer item information; a feature-matched SAKT
control remains necessary to isolate that contribution. Complete training-corpus
statistics do not demonstrate forward-time deployment performance.

These findings are appropriate to share as verified research progress, with the
ASSIST2015 gap and limitations stated. They do not yet justify publication
readiness, three-dataset superiority, or an attention-architecture advantage.
