# Verified selected LightSAKT final-seed results

Generated UTC: 2026-10-14T17:02:15.877581+00:00

Completion and integrity were rechecked from saved artifacts. There are five distinct initialization seeds per dataset; seed 42 reuses the selected tuning checkpoint without retraining. Split seed is 42 for every run. All reported test labels and inference sequence identities match the locked student-disjoint test records.

## Selected hyperparameters and five-seed summary

|dataset|learning_rate|embed_dim|num_heads|dropout|split_seed|training_seeds|seq_len|batch_size|
|---|---|---|---|---|---|---|---|---|
|ASSIST2009|0.001000|64|2|0.100000|42|[42, 123, 2026, 3407, 7777]|100|64|
|ASSIST2015|0.000500|128|4|0.100000|42|[42, 123, 2026, 3407, 7777]|100|64|
|Algebra2005|0.001000|128|4|0.200000|42|[42, 123, 2026, 3407, 7777]|200|64|

|dataset|AUC|ACCURACY|F1|MSE|BCE|
|---|---|---|---|---|---|
|ASSIST2009|0.751056 ± 0.000766|0.732229 ± 0.000801|0.816533 ± 0.001517|0.180233 ± 0.000319|0.538615 ± 0.000886|
|ASSIST2015|0.726088 ± 0.000315|0.749526 ± 0.000419|0.846282 ± 0.000392|0.171233 ± 0.000106|0.516098 ± 0.000340|
|Algebra2005|0.810590 ± 0.000422|0.795574 ± 0.000639|0.870356 ± 0.000350|0.144069 ± 0.000364|0.445272 ± 0.001018|

Mean ± sample SD (ddof=1) on one fixed student split. All five individual results contribute; no best-seed substitution. Hyperparameters were independently checked against the highest validation AUC among all 36 candidates per dataset.

## ASSIST2009 — all individual seeds

|seed|split_seed|best_epoch|auc|accuracy|f1|mse|bce|run_id|
|---|---|---|---|---|---|---|---|---|
|42|42|24|0.750508|0.730861|0.814766|0.180492|0.539145|ASSIST2009-LightSAKT-final-42|
|123|42|29|0.751549|0.732309|0.817761|0.179955|0.537941|ASSIST2009-LightSAKT-final-123|
|2026|42|26|0.752094|0.732586|0.817088|0.179827|0.537406|ASSIST2009-LightSAKT-final-2026|
|3407|42|21|0.750216|0.732948|0.817985|0.180393|0.539183|ASSIST2009-LightSAKT-final-3407|
|7777|42|20|0.750912|0.732437|0.815065|0.180501|0.539401|ASSIST2009-LightSAKT-final-7777|

best_epoch is the selected validation epoch, not the last epoch trained. Checkpoint and source references:

- Seed 42: `checkpoints/ASSIST2009-LightSAKT-grid-42-f08e9494d270/best.pt`; checkpoint SHA256 `d0af3a20f40cf3734883634c23053dbfbf187dee9e7a556c85fcb27b4841197d`; Git commit `2420e8e85bf8bd101fd0866a8f83a10b2818a65d`; metrics `results/raw/ASSIST2009-LightSAKT-final-42/metrics.json`; predictions `results/predictions/ASSIST2009-LightSAKT-final-42.npz`; split SHA256 `134f4bc5eb56083c425f0cef042c9edd09cac43ee7198b37163660500a5f3fe0`. Verification: verified; locked test identities and labels checked.
- Seed 123: `checkpoints/ASSIST2009-LightSAKT-final-123/best.pt`; checkpoint SHA256 `aef075f40bf25d3f59fb8b65930c6f361c80adb72920809d09b5c60a0e1b97a9`; Git commit `1d97d8eb780579a6cf038f8c0329e81dc0e16e03`; metrics `results/raw/ASSIST2009-LightSAKT-final-123/metrics.json`; predictions `results/predictions/ASSIST2009-LightSAKT-final-123.npz`; split SHA256 `134f4bc5eb56083c425f0cef042c9edd09cac43ee7198b37163660500a5f3fe0`. Verification: verified; locked test identities and labels checked.
- Seed 2026: `checkpoints/ASSIST2009-LightSAKT-final-2026/best.pt`; checkpoint SHA256 `2a8f9d128b5f019f7da897d6faefef988dcd5790087cd3f14ba1fb194b58cae8`; Git commit `8bd870e5774c2b83442ae95856eac05dabb110e8`; metrics `results/raw/ASSIST2009-LightSAKT-final-2026/metrics.json`; predictions `results/predictions/ASSIST2009-LightSAKT-final-2026.npz`; split SHA256 `134f4bc5eb56083c425f0cef042c9edd09cac43ee7198b37163660500a5f3fe0`. Verification: verified; locked test identities and labels checked.
- Seed 3407: `checkpoints/ASSIST2009-LightSAKT-final-3407/best.pt`; checkpoint SHA256 `d306739251bd3471081b2d5703e218456696ffd3cbc3cf10668a227070a783ef`; Git commit `8bd870e5774c2b83442ae95856eac05dabb110e8`; metrics `results/raw/ASSIST2009-LightSAKT-final-3407/metrics.json`; predictions `results/predictions/ASSIST2009-LightSAKT-final-3407.npz`; split SHA256 `134f4bc5eb56083c425f0cef042c9edd09cac43ee7198b37163660500a5f3fe0`. Verification: verified; locked test identities and labels checked.
- Seed 7777: `checkpoints/ASSIST2009-LightSAKT-final-7777/best.pt`; checkpoint SHA256 `4bdf2c3fde63c4e885c1c400fde2af7f88fd6e4ed2151d3c1bdedce2166162d8`; Git commit `2ae4fae1508b68bf9052255543ca97b64136c655`; metrics `results/raw/ASSIST2009-LightSAKT-final-7777/metrics.json`; predictions `results/predictions/ASSIST2009-LightSAKT-final-7777.npz`; split SHA256 `134f4bc5eb56083c425f0cef042c9edd09cac43ee7198b37163660500a5f3fe0`. Verification: verified; locked test identities and labels checked.

## ASSIST2015 — all individual seeds

|seed|split_seed|best_epoch|auc|accuracy|f1|mse|bce|run_id|
|---|---|---|---|---|---|---|---|---|
|42|42|18|0.725977|0.749669|0.845897|0.171219|0.515927|ASSIST2015-LightSAKT-final-42|
|123|42|11|0.725897|0.750045|0.846839|0.171121|0.515888|ASSIST2015-LightSAKT-final-123|
|2026|42|12|0.726010|0.749166|0.846253|0.171333|0.516600|ASSIST2015-LightSAKT-final-2026|
|3407|42|13|0.726645|0.749722|0.846481|0.171141|0.515782|ASSIST2015-LightSAKT-final-3407|
|7777|42|13|0.725912|0.749030|0.845942|0.171351|0.516294|ASSIST2015-LightSAKT-final-7777|

best_epoch is the selected validation epoch, not the last epoch trained. Checkpoint and source references:

- Seed 42: `checkpoints/ASSIST2015-LightSAKT-grid-42-64640d03b0d3/best.pt`; checkpoint SHA256 `c08f06c27d9bd31c155dcaff06bbe17b8464578df7e058930233ffbea68d7136`; Git commit `6130a79ff0d2986c116ceddb04398d19127f2afe`; metrics `results/raw/ASSIST2015-LightSAKT-final-42/metrics.json`; predictions `results/predictions/ASSIST2015-LightSAKT-final-42.npz`; split SHA256 `1c1a89044daf1071a8d34d889b02bb7b4fc9ebad4061ea9de75bdd1fb57c25ce`. Verification: verified; locked test identities and labels checked.
- Seed 123: `checkpoints/ASSIST2015-LightSAKT-final-123/best.pt`; checkpoint SHA256 `3e3a98cc956129faf1407b041c0ed65dfa1d5abf082c85dd498872fd63377331`; Git commit `8dbeb81b48a0bcba7fcc9ed493856ad43d387a59`; metrics `results/raw/ASSIST2015-LightSAKT-final-123/metrics.json`; predictions `results/predictions/ASSIST2015-LightSAKT-final-123.npz`; split SHA256 `1c1a89044daf1071a8d34d889b02bb7b4fc9ebad4061ea9de75bdd1fb57c25ce`. Verification: verified; locked test identities and labels checked.
- Seed 2026: `checkpoints/ASSIST2015-LightSAKT-final-2026/best.pt`; checkpoint SHA256 `8377a4cf83ff90054638cafa1cedfeb514ac6a34311cce313b711b95b54f308e`; Git commit `51fc627601e56d833c438b01c8f7b08d46dd60f8`; metrics `results/raw/ASSIST2015-LightSAKT-final-2026/metrics.json`; predictions `results/predictions/ASSIST2015-LightSAKT-final-2026.npz`; split SHA256 `1c1a89044daf1071a8d34d889b02bb7b4fc9ebad4061ea9de75bdd1fb57c25ce`. Verification: verified; locked test identities and labels checked.
- Seed 3407: `checkpoints/ASSIST2015-LightSAKT-final-3407/best.pt`; checkpoint SHA256 `e468362d0ab14f7e16342a472c2c796a1456bf3519c521bfe98632ed18f29ba3`; Git commit `f82780d9497e6359f88454c6fc84e4a28fba78a4`; metrics `results/raw/ASSIST2015-LightSAKT-final-3407/metrics.json`; predictions `results/predictions/ASSIST2015-LightSAKT-final-3407.npz`; split SHA256 `1c1a89044daf1071a8d34d889b02bb7b4fc9ebad4061ea9de75bdd1fb57c25ce`. Verification: verified; locked test identities and labels checked.
- Seed 7777: `checkpoints/ASSIST2015-LightSAKT-final-7777/best.pt`; checkpoint SHA256 `9b8260ac274f98826818a72ca0d6060ac7a057585cf7748834128cf3e0323beb`; Git commit `f35230055cb3eacc28f4b11c80bc28f19ef4e3f6`; metrics `results/raw/ASSIST2015-LightSAKT-final-7777/metrics.json`; predictions `results/predictions/ASSIST2015-LightSAKT-final-7777.npz`; split SHA256 `1c1a89044daf1071a8d34d889b02bb7b4fc9ebad4061ea9de75bdd1fb57c25ce`. Verification: verified; locked test identities and labels checked.

## Algebra2005 — all individual seeds

|seed|split_seed|best_epoch|auc|accuracy|f1|mse|bce|run_id|
|---|---|---|---|---|---|---|---|---|
|42|42|26|0.811010|0.796243|0.870200|0.143540|0.443875|Algebra2005-LightSAKT-final-42|
|123|42|26|0.810613|0.795275|0.870012|0.144115|0.445138|Algebra2005-LightSAKT-final-123|
|2026|42|33|0.810166|0.794669|0.870109|0.144564|0.446732|Algebra2005-LightSAKT-final-2026|
|3407|42|26|0.811000|0.796102|0.870791|0.144096|0.445453|Algebra2005-LightSAKT-final-3407|
|7777|42|23|0.810158|0.795583|0.870669|0.144030|0.445161|Algebra2005-LightSAKT-final-7777|

best_epoch is the selected validation epoch, not the last epoch trained. Checkpoint and source references:

- Seed 42: `checkpoints/Algebra2005-LightSAKT-grid-42-06884c8943b2/best.pt`; checkpoint SHA256 `14b64e7450e0610c300913e933a7dd2ab0d6d3881951e3475557a8e673ee1390`; Git commit `29a355135480044e6e8634311678eaaeab39165c`; metrics `results/raw/Algebra2005-LightSAKT-final-42/metrics.json`; predictions `results/predictions/Algebra2005-LightSAKT-final-42.npz`; split SHA256 `e501ba970774126323e7ae892005d4c38d177bb73b611fe9c90790b3cc9c5643`. Verification: verified; locked test identities and labels checked.
- Seed 123: `checkpoints/Algebra2005-LightSAKT-final-123/best.pt`; checkpoint SHA256 `e096183fb960c4cd0ba6c89ef8f5715f72e6c759b1c22b513f334fda45c87845`; Git commit `7875a6bfc7b9d6e2ee008df31296a1c355baee64`; metrics `results/raw/Algebra2005-LightSAKT-final-123/metrics.json`; predictions `results/predictions/Algebra2005-LightSAKT-final-123.npz`; split SHA256 `e501ba970774126323e7ae892005d4c38d177bb73b611fe9c90790b3cc9c5643`. Verification: verified; locked test identities and labels checked.
- Seed 2026: `checkpoints/Algebra2005-LightSAKT-final-2026/best.pt`; checkpoint SHA256 `257e8e410c0a4e7153b9d3b358f3c52b1f5f47147bc265eba2fa4948551ed15f`; Git commit `7b8e3466643053e29745d59df55d1e9b2b6ce5d1`; metrics `results/raw/Algebra2005-LightSAKT-final-2026/metrics.json`; predictions `results/predictions/Algebra2005-LightSAKT-final-2026.npz`; split SHA256 `e501ba970774126323e7ae892005d4c38d177bb73b611fe9c90790b3cc9c5643`. Verification: verified; locked test identities and labels checked.
- Seed 3407: `checkpoints/Algebra2005-LightSAKT-final-3407/best.pt`; checkpoint SHA256 `65ec77b47c35ae06039c357b7da2edf6e030abfbdbcfe7fbf764d546328bfede`; Git commit `b90b88a0cb18206c82e6399014b5fbee216efd33`; metrics `results/raw/Algebra2005-LightSAKT-final-3407/metrics.json`; predictions `results/predictions/Algebra2005-LightSAKT-final-3407.npz`; split SHA256 `e501ba970774126323e7ae892005d4c38d177bb73b611fe9c90790b3cc9c5643`. Verification: verified; locked test identities and labels checked.
- Seed 7777: `checkpoints/Algebra2005-LightSAKT-final-7777/best.pt`; checkpoint SHA256 `a3da82abe09e86df4fa76c5ddd2c7edc0069d0f5166cc8677ec94d09a059cd77`; Git commit `e597749acb3ee051fdac7744fef88702587ba692`; metrics `results/raw/Algebra2005-LightSAKT-final-7777/metrics.json`; predictions `results/predictions/Algebra2005-LightSAKT-final-7777.npz`; split SHA256 `e501ba970774126323e7ae892005d4c38d177bb73b611fe9c90790b3cc9c5643`. Verification: verified; locked test identities and labels checked.

## Verification limits and reproducibility

Checks include complete run status, checkpoint/prediction hashes, strict finite checkpoint loading, selected validation epoch and metrics, training-source fingerprint, validation-only configuration selection, disjoint students, locked test-sequence order and target labels, and metric recomputation from saved probabilities.

Independent CPU validation/test checkpoint inference verified for 15/15 selected LightSAKT runs. Every test target and probability was checked against the saved evidence with FP32 cross-device tolerances (probabilities: atol=1e-6, rtol=1e-6; metrics: atol=1e-7, rtol=1e-6). Per-run proof: reports/selected_final_inference_verification.json. Independent verification of all remaining training checkpoints remains scheduled.

Source tables: results/tables/LightSAKT_final_summary.csv and LightSAKT_final_individual_seeds.csv. Full verification: reports/light_final_summary_verification.json. Grid verification: reports/light_grid_verification.json. Selection records: experiments/selections/<dataset>-LightSAKT.json.

Reproduce on CPU without signalling training: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.light_final_report --refresh-eta --eta-dir reports/summary_eta/latest`.
