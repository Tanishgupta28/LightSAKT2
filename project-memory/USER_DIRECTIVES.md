# User directives

## Latest clarification — allocation first, 20 October2026

User has no exclusive MIG assignment evidence. Preserve all checkpoints/results;
do not stop LightSAKT2 solely for GPU release before HiTSKT allocation is confirmed.
Project ownership is not permission to displace other reservations. No CUDA or
HiTSKT training without verified authorization. Do not restart anything or alter
the existing hold. Current clarification requests administrator guidance only.

## Earlier handoff authorization — historical order

20 October 2026 UTC — explicit user authorization supersedes earlier protection
against stopping only the user's verified LightSAKT2 jobs. Inspect ownership,
preserve all checkpoints/results/configs/logs, disable future automatic launches,
gracefully stop verified training if active. Never kill unrelated jobs/services,
or force-kill during artifact writing. Ask if ownership or safe preservation
cannot be established. All actual workers were already interrupted before this
turn, so no signals were necessary or sent.

Keep `experiments/USER_TRAINING_HOLD.json` active. No future LightSAKT2 resumption
or automatic recovery without a later explicit user decision. Preserve pending
configurations and both interrupted statuses; do not mark unfinished work complete.

User conditionally authorizes HiTSKT FAVOR+ staged CUDA checks then <=5 ASSIST2017
train-only updates after genuine exclusive allocation and correctness PASS.
Stopping LightSAKT2/free memory does not prove allocation. No unrelated workload
termination, allocation change, full training, test evaluation, DKT/EdNet
resumption/reset or historical ELU+1 modification is authorized.

Commit/push verified changes to the appropriate existing research branches;
no merging, force-push or history rewriting. Leave unrelated dirty work intact.
