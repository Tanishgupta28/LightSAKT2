# Audited training source equivalence

Only Algebra2005 prepare() branch changed from Step Start Time to First Transaction Time and added ordering documentation. ASSIST preprocessing, models, sequence loaders, training, evaluation and SHA256 utility are byte-identical. Stored processed/split/checkpoint hashes are independently verified. Algebra2005 is excluded.

Source tree 41b7420 exactly reconstructs the prior LightSAKT run fingerprint 42a9f1b495f793154d4788e778edecb2637443bb109c1473f28ab970c2a49783. Current tree 27901b7 and current source reconstruct 0a04dc026130226cf58cd1fad8e9f9a696d91088097b2190cdf9e1bcd1fa48f6. The only source diff is preserved in training_code_transition.diff. This explicit allowlist does not authorize reuse for Algebra, unknown code changes, changed data, changed splits or changed hyperparameters. The original run metadata and checkpoint remain untouched.
