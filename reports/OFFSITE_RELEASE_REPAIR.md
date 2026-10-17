# Completed-run backup recovery — 17 October 2026

The completed-run backup worker failed because two unpublished private GitHub
release drafts had the same `research-recovery-20261013` tag: IDs `411825133` and
`414881329`. The previous lookup required exactly one matching release. The
verified asset ledger bound existing evidence to release `411825133`.

The corrected lookup retains that binding and preserves both drafts. Uploads
address the chosen release ID directly, avoiding ambiguous tag-based uploads.
Existing immutable assets are reused; new archives still require successful
download and SHA256 verification. Public repositories, published recovery
containers and corrupt downloads remain rejected.

Only the failing CPU backup worker was replaced. Healthy GPU training, its
scheduler, locked data splits, model source, gradients and completed experiments
were untouched. The independent active-checkpoint worker remained running.

The full CPU suite passed **132 tests**, with **15 CUDA checks skipped** during
active training. Evidence: `reports/test_runs/20261017T063340-recovery-release-cpu.log`.
New regression tests cover duplicate drafts, existing-ledger selection and
release-ID uploads. Real uploads and downloaded checksums verified the repaired
path; completion and backlog evidence are in `reports/offsite_release_repair.json`
and the authoritative immutable asset ledger `reports/offsite_backup.json`.

The lookup failure is retained in the repair record and Git history. The live
failure file can also report transient upload/download failures; the worker
retries them without marking unverified assets complete. Local completed archives,
resumable active checkpoints and the manual recovery procedure remain available.
This repair does not configure automatic platform recreation after an environment
shutdown.
