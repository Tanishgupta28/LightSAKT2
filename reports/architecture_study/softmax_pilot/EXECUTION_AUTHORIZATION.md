# Approved bounded causal-softmax experiment — 25 October 2026

The user explicitly authorizes: “this bounded causal-softmax pilot, provided the
shared-GPU admission checks pass.” The scope is fresh one-layer causal softmax
on ASSIST2009, seed 42, with the already frozen protocol, validation-AUC selection,
4 GiB allocator cap and 15-minute invocation limit. The user requires precision,
recall, accuracy, F1, AUC and BCE per epoch, preserving the final test and prior
research artifacts. No recency run or multi-seed campaign is included.

Existing intentional shared-H100 permission is documented at
`/workspace/capstone/reports/favor_plus_v1/shared_gpu_v1/USER_SHARED_AUTHORIZATION.md`.
The new instruction supplies the current LightSAKT execution scope. No exclusive
ownership, scheduler lease, host PID attribution, or right to stop peers is
claimed. The operator window in the execution JSON limits this invocation;
it is not an externally verified reservation expiry.

Use the same exposed research MIG:
`MIG-d2fcfd79-720e-5218-82c9-0a132be68920`. A finite read-only inspection passed
shared admission with about 29.3 GiB free and an existing compute PID recorded as
unattributed. MIG utilization is unsupported, not zero; retain the authorized
headroom/cap/pacing fallback. Recheck admission immediately before CUDA.

Keep the original general holds present and byte-identical. Release only the
exact `one_layer_softmax` configuration for stage `softmax`, through an expiring
exception bound to both hold-marker hashes and the immutable configuration hash.
General scheduling, recovery, the recency variant and other seeds remain held.
Preserve 4 GiB allocator / 10 GiB admission-free / 6 GiB runtime-free / 4 GiB host
RSS / 2 GiB host-headroom limits, 0.1-second inter-batch pacing, the 30-second
step stop, and 900-second invocation stop. No resizing or automatic retry.

The model implementation, original trainer source and existing metric definitions
are unchanged. A dedicated-process reporting context adds precision/recall at
the fixed 0.5 threshold and online training metrics. CPU checks verify identical
model updates and exact resume with and without this reporting. Online training
scores are explicitly distinguished from validation-mode epoch-end scores.
Prior depth reports and checkpoints stay immutable; new reports use a separate
softmax directory. Stop after this one approved run and close the grant.
