# Checkpoint recovery and platform limitations

Development and restoration use the existing `/workspace/LightSAKT2` root. Healthy training is never signalled by the recovery supervisor. Completed run configurations, metrics, weights and predictions are never reset.

## What is automatic

The detached CPU supervisor (`scripts.recover --watch`) checks project processes every 30 seconds. It restores missing CPU reporting/backup services. If the scheduler and coordinator have died, it waits for any independently surviving training or GPU-analysis child and GPU admission lock to finish. It then verifies the frozen source fingerprint, Python/package versions, dataset/split checksums, completed artifacts, and resumable best/latest states before launching the original scheduler. The scheduler's exclusive CPU/CUDA test gate still executes before training resumes. Optimizer, epoch, early-stopping state, model weights, Python/NumPy/PyTorch/CUDA RNG and DataLoader RNG come from **latest.pt**, rather than restarting from best.pt.

There is a 120-second retry cooldown and a three-attempt limit without completed-run progress. CUDA unavailability, version mismatch, missing/corrupt evidence or live lock owners prevent unsafe startup. A stalled but living training process is reported rather than killed automatically. Original GPU admission still limits MHA to two workers and keeps PAM/other models exclusive.

## What is not automatic

A supervisor inside a terminated container cannot restart itself. PID 1 is Jupyter, systemd/cron/supervisord are unavailable, and the mounted Kubernetes service account denies current-pod access, exec, deployment changes and CronJob creation. No supported provider startup hook was found or installed. Evidence: `reports/recovery/infrastructure.json`.

**Automatic restart after environment shutdown is not configured.** Kubernetes restarting Jupyter would not, by itself, invoke this project. The staged provider entrypoint is `bash /workspace/LightSAKT2/scripts/resume_research.sh`, but an authorized platform administrator would have to configure it externally. The current healthy environment was not restarted to test that possibility. `/workspace` is NFS; retention after deprovisioning is unverified.

## Manual recovery when the NFS project survives

Use the original supplied environment image. Required Python/library versions are frozen in `configs/recovery.json`; notably Torch is `2.4.0a0+f70bd71a48.nv24.6`. Installing unversioned generic requirements does not reproduce that build. Preflight blocks mismatched versions.

```bash
cd /workspace/LightSAKT2
python -m scripts.recover --check
bash scripts/resume_research.sh
python -m scripts.progress
```

The bootstrap is idempotent. If healthy controllers/workers already exist, they remain running. If they are absent, its detached supervisor starts the original scheduler from valid latest checkpoints, along with missing CPU services. Inspect `reports/recovery/supervisor.json`, `reports/recovery/schedule.log`, `reports/recovery/preflight.json`, and `reports/EXECUTION_HEALTH.md`.

## Manual recovery when artifact files are missing

First restore the **pushed research-rebuild source branch into the same authorized root**, or recover it from a verified Git bundle. Do not clone over an existing repository or overwrite unrelated files. Restore the supplied environment and GitHub authentication. No credentials belong in this repository. With no live project GPU/controller process, run:

```bash
cd /workspace/LightSAKT2
python -m scripts.restore_artifact --asset original_and_raw_20261013.tar.gz --prefix data/raw/
python -m scripts.rebuild_missing_data --install-missing
python -m scripts.restore_artifact --all-completed
python -m scripts.restore_artifact --latest-active
python -m scripts.recover --check
bash scripts/resume_research.sh
python -m scripts.progress
```

Downloads use asset IDs and SHA256 references in the pushed `reports/offsite_backup.json` and `reports/active_recovery_index.json`. The archive restorer checks hashes, rejects path traversal/links, and writes only absent paths. Differing existing evidence is a hard failure, rather than permission to overwrite it. Identical restored files are skipped. Active restoration chooses the highest verified captured epoch per unfinished run, restores checkpoint bytes only, and preserves Git-carried history/status as immutable historical copies. Completed runs never fall back to older active snapshots.

If a checkpoint from after the latest offsite capture was lost, those unavailable unfinished epochs cannot be recovered. The engine resumes the last available completed epoch; it does not claim later progress survived. Missing completed-run evidence blocks startup until recovery is resolved. Dependency/package or archive failures remain explicit.

## Backup frequency and limits

Local best/latest state is saved atomically every epoch. Expensive GKT runs now receive an offsite snapshot after their first completed epoch, then when a new epoch exists and at least five minutes have elapsed, with a 60-second poll. Other long jobs retain the 15-minute initial threshold/hourly snapshot interval. Completed-run backups continue independently.

Snapshots require coherent best/latest epochs, include optimizer/RNG/history/config, and are downloaded and SHA256 checked from private daily draft release containers. This is **not a five-minute loss guarantee**: no partial-epoch checkpoint exists, and epoch duration, polling/upload time and provider outages affect the loss window. No snapshot deletes an earlier archive. The daily two-worker minimum-interval policy stays below the existing 1,000-assets-per-release limit under continuous operation.

## Executed verification

- A real isolated toy worker was killed with SIGKILL immediately after an atomic epoch checkpoint. Public recovery preflight accepted the state; resumed model tensors, Adam tensors, RNG, selected validation epoch and validation metrics matched uninterrupted execution exactly. Reopening the completed run changed no evidence bytes.
- Real ASSIST2009, ASSIST2015 and Algebra2005 raw files were reprocessed in staging. All three processed-data and student-split hashes matched the locked originals exactly; production data was not overwritten. Evidence: `reports/recovery/preprocessing_reconstruction.json`.
- Archive tests verified idempotent restoration, checksum rejection, conflict rejection and path-traversal rejection before writes.
- The live bootstrap was invoked twice and retained its existing supervisor; no healthy GPU training was stopped/restarted. The CPU recovery services alone were refreshed to load version guards and improved backup cadence.

These tests verify checkpoint/process/data recovery. They do not claim a real environment shutdown or provider startup hook was tested. Current full-suite execution is recorded in `reports/latest_cpu_test_run.json`.
