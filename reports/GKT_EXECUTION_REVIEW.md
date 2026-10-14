# Additional GKT execution review

The complete 957-entry/912-distinct-training schedule remains active. Neither model definitions nor training mathematics were changed; the protected source fingerprint remains `0a04dc026130226cf58cd1fad8e9f9a696d91088097b2190cdf9e1bcd1fa48f6`. The 36-candidate tuning budget per variant/dataset, fixed student splits, five selected seeds, Adam, batch sizes, precision and validation-only selection remain intact.

## New executed probe

The probe ran exclusively between completed research runs, with no signals to healthy training. It used genuine train batches for both GKT variants on all three datasets, plus the widest Algebra PAM/MHA settings: eight configurations, each measured with four, one and two CPU dispatch threads (24 cases). Each case executed two warm-up and three timed FP32 Adam steps. Initialization, CPU RNG, step probabilities, gradients, parameter updates and CUDA RNG were compared through exact tensor-byte digests at every step.

All cases passed exact equivalence. One-thread speedups ranged from **0.8312 to 1.0632** versus the original four threads; two-thread speedups ranged from **0.7603 to 1.0297**. There was no consistent gain across configurations, and the short timing window cannot establish sustained epoch throughput. The original four-thread policy stays active; no improvement is credited to the ETA. Raw per-step measurements and proofs are in `reports/profiling/gkt_threads/measurements.json` and `verification.json`.

## Existing accepted and rejected measures

Exact SHA-verified sequence caches continue to avoid repeated CSV grouping. Two independent MHA workers retain their previously verified 1.51–1.61 throughput improvement and kernel-owned admission slots; PAM stays exclusive because wide PAM measured 0.90 with two workers. Existing profiles identify recurrent node updates, many CUDA launches and host dispatch as the main GKT costs. Cached data collation/transfers are small relative to that computation.

Batch-size changes, sequence trimming/bucketing and mixed precision would alter the established update/RNG trajectory. BF16 previously failed strict probability/gradient equivalence. Compilation and CUDA graphs lack complete dropout/RNG/checkpoint-resume proof and a measured sustained benefit. They remain disabled. No unverified kernel/architecture shortcut was adopted.

## Additional safe overhead and recovery improvements

Publication plots now use stable SVG IDs and omit changing creation timestamps. Identical observations produce identical PNG/SVG/PDF bytes, and unchanged files are not rewritten to NFS or Git. An executed test checks all three formats preserve bytes and modification times, while changed numerical observations update them. This reduces reporting/storage churn; no GPU-hour speedup is claimed.

The new CPU process supervisor restores missing services/controllers only when their processes and lock owners have gone. It defers to surviving GPU children, checks frozen versions/source/data/checkpoints, and starts the original correctness-gated scheduler from latest state. Long GKT jobs receive offsite recovery after the first completed epoch and whenever a newer epoch exists beyond a five-minute minimum interval, with a 60-second poll. This reduces potential interruption losses relative to hourly snapshots; it does not change gradients or create mid-epoch state.

True automatic recovery after container replacement remains unavailable because provider startup/controller access is denied. The tested manual bootstrap and checksum-verified data/archive restoration commands are in `reports/RECOVERY_PROCEDURE.md`. No healthy GPU process was stopped or restarted to configure this work.

## Preserved scientific findings

LightSAKT's original architecture and all 108 grids/75 ablation slots are retained. Its original-device probabilities were verified independently. The separate efficiency evidence continues to show LightSAKT slower in all 60 matched latency comparisons, with lower peak allocated memory in only 7/60. Theoretical linear sequence-length work, measured latency/memory and parameter counts stay separate; these new GKT engineering probes do not replace or modify that benchmark.
