# Next checkpoint

Preservation/hold complete; stop operational work. Leave
`experiments/USER_TRAINING_HOLD.json` present, all scientific artifacts and older
dirty state intact. Publish scoped controls/evidence/memory to `research-rebuild`
only; do not merge or stage old unrelated modifications.

Next bounded task is external allocation clarification for HiTSKT, handled in
`/workspace/capstone/project-memory/`. Do not poll automatically, signal unrelated
GPU owners, initialize CUDA or resume LightSAKT2.

Any future user-authorized LightSAKT2 recovery must first recheck ownership and
checkpoint/data/config/source hashes. Epoch2 GKT (`...e01b589e46cc`) is validated;
waiting GKT (`...e1c69a92d6b5`) has no epoch. Consult
`reports/priority_handoff_20261020/interrupted_checkpoint_and_archive.json` and
the still-local original checkpoints. Only then make an explicitly reviewed hold
release decision. Do not use stale PID files or reset research state to Git HEAD.
