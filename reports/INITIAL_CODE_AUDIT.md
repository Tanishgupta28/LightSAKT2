# Initial source audit

Root: /workspace/LightSAKT2, initially empty; no prior Git history. Source inventory and SHA256 records are in drive_inventory.json and original_checksums.json. All 1,613 inventoried files are preserved with matching byte sizes. Source .py/.json/.ipynb and PDF files were fetched through read-only Drive calls; many historical artifacts were copied from size-matched local files, which is not proof of Drive byte identity. No original Drive writes were made. Raw ASSIST CSVs and Algebra training TSV were absent from the shared folder and obtained from local reference material, with provenance to be checked separately.

|File/function|Problem and effect|Correction/classification|Architecture change?|Executed verification required|
|---|---|---|---|---|
|models/light_sakt.py LinearAttention.forward|Global KV numerator contains future responses; invalid ranking metrics possible|Causal prefix KV and key sums; mathematical correctness|No|Future perturbation and dense kernel reference|
|same|Denominator omits query contraction and sums across causal rows; not normalized kernel attention|Same boundary and scalar query-key normalization; mathematical correctness|No|Normalized weights and dense reference|
|same visualization branch|Softmax inference replaces ELU kernel inference|Compute explicit kernel weights separately; evaluation correction|No|Prediction equality with visualization|
|Light_SAKT.forward|No padding mask, unsliced P, CPU mask|Sanitize invalid indices; mask projected keys/values; slice P; masking correction|No|Padding replacement/extension, CPU/CUDA|
|train/train.py and dataset-specific scripts|random_split after match_seq_len; same student across sets; test used for checkpoint selection; stale split caches|Split students first 70/10/20; separate validation selection; data/training correction|No|Overlap, heldout-label independence|
|models/utils.py match_seq_len|Stride L+1 loses transitions between chunks|Stride L with one overlap; data correction|No|Full n-1 target coverage|
|models/utils.py import|Global CUDA default tensor type alters unrelated code|Explicit devices; engineering|No|CPU run on CUDA-capable host|
|data_loaders/*.py preprocess|Unchecked pickles and full-data vocabulary; binary/order validation incomplete|Fresh raw preprocessing, train-only vocabulary with UNK; data correction|No|Determinism, heldout record invariance|
|models/dkt*.py train_model|Batch metrics rather than globally pooled observations, test checkpoint selection; DKT+ differences masked on one endpoint only|Common trainer and two-endpoint masks; evaluation/training correction|No|Metric aggregation and short sequences|
|models/dkvmn.py forward/train_model|Unrestricted squeeze removes batch dimension; current-question objective differs from other models|Use squeeze(-1), read target query after current write; alignment correction|No|B=1, target-response perturbation|
|models/gkt.py __init__/update|init_h and bias allocated uninitialized; GRU iterates across skill IDs with zero hidden state rather than per-skill temporal state|Initialize parameters, per-node recurrent update with prior state; mathematical/training correction|Baseline correction|Repeatability and batch independence|
|models/gkt.py docstring|Historical GKT omits erase-add gate and batch normalization explicitly; not canonical published GKT|Label historical simplified PAM/MHA faithfully; canonical comparison remains a limitation|No silent additions|Variant labels and parameter manifests|
|models/sakt.py forward|CPU triangular mask, no padding mask, unsliced P and squeeze|Device mask, padding, sliced P and squeeze(-1); masking correction|No|Leakage/padding tests|
|train/experiment.py and plots|Historical grid/test scores lack immutable run identities; plotting maximum historical scores cannot establish untouched-test results|Immutable configs, SHA references, programmatic result outputs; evaluation/engineering|No|Traceability audit|
|train/abalationstudy.py|Regularization wording may be inherited from TransDKT+; cannot distinguish from dropout for LightSAKT|Block undefined distinct variant; research-design ambiguity|Would require approval to invent|Config semantics|
|checkpoints/results notebooks|No trusted raw-data/split/code binding, no optimizer/RNG resume evidence|Preserve as unverified, fresh training; reproducibility|No|Checkpoint restoration and resume equivalence|

Scope: source inspection includes the six model definitions, all dataset loaders, configuration files, training entrypoints, grid and ablation scripts, visualization/plot scripts, and original test files. Historical binaries are inventoried and hashed, not accepted as evidence of new performance. No architecture redesign is authorized or implemented.
