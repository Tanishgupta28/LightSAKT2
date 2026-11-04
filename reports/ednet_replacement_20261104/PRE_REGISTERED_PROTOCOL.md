# Replacement-dataset protocol — 4 November 2026

EdNet-KT1 is selected over Junyi because its official question table supplies
stable question_id, correct_answer and tags matched to recorded user answers.
The public Junyi schema describes exercise names and some problem templates;
these are not established as individual generated-question identities. A Junyi
exercise-level study would need that separate label. No identifier is invented.

The EdNet archive and Contents are genuine publisher-linked files. Use5000student
files selected before labels with fixed seed20261104; keep each authentic user
and question ID. The target is the first presented item in each solving session
(timestamp then raw-row tie break),selected before inspecting response validity.
Other questions from the same bundle are excluded so their responses cannot
become the current item's history. This is an explicitly reduced EdNet-KT1
population,not a full-KT1 or official-leaderboard claim. Earlier-session history
remains causal; full training-corpus rate fitting retains the established offline
student-disjoint interpretation,not chronological deployment validity.

All three models receive identical 70/10/20 student partitions(seed42),length100
windows/all n-1 targets,training-only sorted composite-tag vocabulary,batch64,
Adam lr0.001/betas0.9,0.999/eps1e-8/decay0,d64,2heads,dropout0.1,max50epochs,
patience5 and validation-AUC checkpoint selection. These common compact settings
are frozen before any new model validation/test scores. Old datasets keep their
existing selected settings;there is no claimed matched tuning search across
all datasets. Difficulty retains its original one-scalar correction and smoothing10.
Five matched training seeds42,123,2026,3407,7777;no model changes.

Fifteen serial core invocations,each<=900seconds and4GiB allocator,total<=13500
window-seconds(225minute ceiling),with unchanged shared GPU/host floors,pacing,
step limit and fresh admission. Exact prior shared-MIG UUID remains available;
current user authorizes the new missing experiments. Preserve old general hold
bytes and bind explicit current scope to this separate dataset/config manifest.
Stop on unsafe conditions without automatic retries. Test inference stays out
of training and occurs only after all configs are frozen and each checkpoint is
validation-selected. No HiTSKT access or operation. Feature-matched SAKT control
is preparation only;core results are prioritized.
