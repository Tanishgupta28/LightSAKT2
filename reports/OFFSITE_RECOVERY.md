# Private offsite recovery

This private draft release is an artifact container for an incomplete research schedule. It is not a research publication or a claim that all experiments finished. Original/raw archives, verified completed-run checkpoints/configurations/predictions, supplemental reused-result archives and an initial full Git history bundle are copied independently of the workspace. Uploads use unique names; existing artifacts are never overwritten or deleted. Every new asset is downloaded and SHA256 checked before being marked verified in `reports/offsite_backup.json`.

The detached CPU offsite worker checks the existing repository remains private and the container remains a draft, then synchronizes newly available completed archives. Git commits continue to be pushed separately; all local milestone bundles remain under `backups/`. A failure keeps local backups and is retried, without signalling or restarting GPU training. Running jobs acquire an offsite archive when completed; their latest atomic checkpoints remain on NFS during training.

Recovery in an authorized project root:

```
gh release download research-recovery-20261013 --repo Tanishgupta28/LightSAKT2 --dir backups/downloaded_recovery
```

Compare each downloaded file with the SHA256 entries before extracting original/raw files and the desired run archive. Git history can be restored from the downloaded bundle or current `research-rebuild` branch. Restore only into an authorized empty recovery location, and use the saved configurations/split manifests to regenerate processed data. Inspect run statuses before resuming; completed runs must remain completed. The recovery command does not start a scheduler.

Release assets are outside ordinary Git history. The provider documents a 2 GiB per-file limit and 1,000 assets per release: [GitHub release documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases). The worker enforces both limits. Remote persistence protects the uploaded completed evidence; it does not keep the GPU process running after environment shutdown.
