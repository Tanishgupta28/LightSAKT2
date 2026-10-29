# Exact next checkpoint — authorize a defined replication budget

Seed42 question-difficulty run is COMPLETE. Do not restart/resume/retrain it.
Retain old epoch18 interruption record as history,not current state. To verify
current result on CPU:python -m scripts.verify_question_completion_20261029;
state continuity:python -m scripts.verify_question_state_continuity.

Prepared packet `/workspace/LightSAKT2-architecture/experiments/question_replication_20261029/plan.json` contains10configs across five seeds42/123/2026/3407/7777
and two feature-matched families. REUSE completed LightSAKT seed42;only4new
LightSAKT runs+5SAKT controls remain. Exact same feature SHA256 and split in all.
SAKT retains its previously selected lr0.0005/4heads/dropout0.2;LightSAKT
lr0.001/2heads/dropout0.1. No new tuning. Control scripts/feature_matched_sakt.py
is CPU validated and bound by hash in configs. No real-data controls ran.

Campaign requires SEPARATE explicit budget approval and a seed-general budget-
enforcing launcher (current pilot is deliberately seed42-only). Proposed maximum:
9new sequential invocations,<=900s and4GiB per invocation,total<=8100GPU-seconds
(135minutes);existing shared GPU/host floors,pacing,step limits remain. No automatic
extra windows/retries. This is a ceiling,not a throughput estimate or permission.
Do not launch based on prepared files or unused prior scope.

Report paired validation deltas,mean/sample SD over all seeds;never bestseed.
Five seeds assess initialization variance,not new student cohorts. Keep test
locked. Global train-fitted difficulty is valid offline,not a forward-time claim;
a chronological protocol would be a new separately frozen investigation,not a
silent change. Algebra2005 needs verified item/step mapping;ASSIST2015 current
release lacks problem IDs. No silent schema substitution. No force push or
credentials exposure;preserve local checkpoints,Git bundles and old artifacts.
