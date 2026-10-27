# One bounded recency-softmax run — 27 October 2026

User explicitly authorizes exactly one ASSIST2009 seed42 one-layer recency-biased
softmax run under existing shared-GPU permissions.4GiB allocator cap,900second
invocation limit, same frozen settings/early stopping/checkpoint selection.
Validation only; no final test evaluation, new tuning, peer signals or retry.
Single recency stage releases only exact current config/hold hashes; the general
holds remain active. Existing shared policy retains10GiB admission/6GiB runtime
free floors,4GiB RSS/2GiB headroom,0.1s pacing,30s step stop and the documented
unsupported-MIG-utilization fallback. Operator window is not an exclusive lease.
Previous history and Git authentication blocker remain preserved.
