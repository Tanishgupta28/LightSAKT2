# Reproduce the assets without training

From this asset directory, use the versions in `verification/python_environment.json` and install `python_scripts/requirements-plotting.txt` in a separate environment if needed:

```bash
python -B python_scripts/generate_all_paper_graphs.py
python -B python_scripts/generate_paper_tables.py
python -B python_scripts/verify_assets.py --reproduce
```

These programs read the included genuine CSV/JSON extracts. They need no GPU, raw dataset or checkpoint. DejaVu Sans fonts, SVG hash salt and PDF/SVG timestamps are fixed. In the recorded environment, running the graph programs again reproduces PNG/PDF/SVG bytes exactly. Different software or fonts can change bytes; numerical values must still match. Each graph has input hashes and its caption in a `.provenance.json` sidecar.

To repeat the full audit against separately preserved source artifacts, run from the original research checkout:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B python_scripts/collect_verified_sources.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B python_scripts/verify_leakage.py
python -B python_scripts/verify_assets.py --sources --reproduce
```

Adjust the script path if invoking from the repository root. These commands require the exact original datasets, checkpoints and saved probability files referenced by SHA256. They do not train or write into historical result directories. The separate `replay_reusable_GKT.py` performed CPU evaluation of exactly two frozen checkpoints and saved real probabilities in the ignored local `.tools` directory. Its cache prevents duplicate evaluation. No missing-model GPU campaign is implemented or implicitly authorized here.

PyMuPDF and pypdf were used to inspect the original paper. They are not needed to draw graphs. The original PDF is referenced by hash, not included or rewritten. `generate_documentation.py` provided the initial mapping and report drafts; the final Markdown was edited for readability. Numerical tables and graph programs remain reproducible from the supplied extracts.

Run `python -B python_scripts/package_assets.py --destination /your/export/directory` to create and verify a portable ZIP. The archive contains all completed assets, input extracts, provenance and instructions. It excludes raw datasets, checkpoints and full prediction arrays. Small anonymized prediction examples contain actual observations and probabilities, with no original student IDs.

After any future authorized training, extend/version the result inventory and verify new checkpoints, labels, masks and probabilities before adding measured rows. Never fill missing metrics or select configurations using test results.
