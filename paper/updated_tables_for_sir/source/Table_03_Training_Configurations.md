# Table 3: Training network configurations for different datasets

Validation-selected LightSAKT settings

|Setting|ASSIST2009|ASSIST2015|Algebra 2005–2006|
|---|---|---|---|
|train_ratio|0.7|0.7|0.7|
|validation_ratio|0.1|0.1|0.1|
|test_ratio|0.2|0.2|0.2|
|learning_rate|0.0010|0.0005|0.0010|
|seq_len|100|100|200|
|optimizer|Adam|Adam|Adam|
|embedding_dimension|64|128|128|
|num_heads|2|4|4|
|dropout|0.1|0.1|0.2|
|batch_size|64|64|64|
|max_epochs|50|50|50|
|early_stopping_patience|5|5|5|

LightSAKT configurations selected independently by validation AUC; target student-disjoint ratios are 70/10/20.
Adam: betas (0.9, 0.999), epsilon 1e−8, weight decay 0. Early stopping monitors validation AUC.
Split/grid seed 42; final training seeds: 42, 123, 2026, 3407, 7777. Test data do not select configurations.
