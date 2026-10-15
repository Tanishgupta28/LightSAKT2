# SAKT matched-grid continuation

All 108 SAKT grid configurations and 15 selected final-seed slots already completed
in the original 957-entry schedule. The reference manifest is
`experiments/sakt_matched_grid_manifest.json`. No training jobs were added or restarted.

All 108 validation AUCs were independently reproduced on CPU. A separate strict
CPU full-metric check encountered one threshold-sensitive accuracy discrepancy
equivalent to one net correct prediction in 94,262 targets. Its failure remains in
`reports/sakt_grid_cpu_validation_failure.json`; tolerances and original experiment
metrics are unchanged. The complete original-device validation check waits for the
existing exclusive GPU admission lock, allowing the healthy GKT run to finish.

Two detached processes are configured: `scripts.sakt_matched_grid verify --device cuda`
and the CPU-only `scripts.publish_sakt_grid`. Their live state and PIDs are recorded in
`reports/sakt_grid_verification_execution.json` and
`reports/sakt_grid_publication_execution.json`. Publication waits for all 108 original-device
checks, then regenerates tables and plots, verifies protected artifacts, uploads and
downloads a private evidence archive, and commits/pushes the milestone. A separate
preliminary archive preserves current AUC-verified outputs and explicit prediction masks;
its verification record is `reports/sakt_grid_pending_offsite_backup.json`.

The following commands resume these analysis stages after an interruption. They take
their own locks and retain verified checkpoint records. Inspect the live state first;
do not start duplicate analysis workers. No command below starts model training.

```bash
cd /workspace/LightSAKT2
nohup env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u -m scripts.sakt_matched_grid verify --device cuda >> reports/sakt_grid_gpu_validation.log 2>&1 < /dev/null &
nohup env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u -m scripts.publish_sakt_grid >> reports/sakt_grid_publication.log 2>&1 < /dev/null &
```

The existing training recovery system and manual procedures remain unchanged; see
`reports/RECOVERY_PROCEDURE.md`. Detached processes survive a terminal/session disconnect
while this environment remains alive. Automatic recreation after full environment
shutdown is not configured or claimed. Preserve and restore the private offsite
checkpoint archives and Git history if the workspace becomes unavailable.

To reproduce only the preliminary archive without touching GPU training:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.preserve_sakt_preview
```

CSV/Markdown/LaTeX tables and figures may be used with their explicit verification
status. Do not mark the additional workflow fully verified until the original-device
proof and final publication/offsite records confirm completion. A verification failure
must be investigated; it must not be hidden by changing tolerances or result values.
