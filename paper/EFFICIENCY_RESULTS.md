# Fresh controlled efficiency results

Executed 120 cases (60 matched comparisons) on the H100 MIG partition. Sequence lengths 50/100/200/500/1000; batches 1/8; equal attention, equal complete dimensions, and documented primary complete models; forward and forward/backward; FP32, three warm-ups and ten timed repetitions with synchronization. Models are newly initialized for compute measurement; this is not predictive-performance evidence. Raw timings and hardware/source manifests are under experiments/efficiency/.

LightSAKT had lower mean latency in 0/60 measured matched comparisons, and lower peak allocated memory in 7/60 comparisons. The data do not support a general claim that the current LightSAKT implementation is faster than SAKT. No architecture was changed to improve these observations.

|Scope|Length|Batch|Operation|LightSAKT ms|SAKT ms|Latency ratio|Peak allocated memory ratio|
|---|---|---|---|---|---|---|---|
|equal_attention|1000|8|inference_forward|2.063|0.799|2.581|1.447|
|equal_attention|1000|8|forward_backward|9.037|1.753|5.156|0.909|
|equal_complete|1000|8|inference_forward|2.400|1.067|2.249|1.668|
|equal_complete|1000|8|forward_backward|10.873|6.498|1.673|1.596|
|documented_complete|1000|8|inference_forward|4.687|1.641|2.857|3.483|
|documented_complete|1000|8|forward_backward|25.300|7.293|3.469|3.278|

Actual primary-model parameter counts:

|Dataset|LightSAKT|SAKT|Ratio|
|---|---|---|---|
|ASSIST2009|148673|112201|1.325|
|ASSIST2015|134849|101401|1.330|
|Algebra2005|108449|197501|0.549|

The primary configurations use different widths; their parameter differences cannot be attributed solely to kernel attention or the reduced feed-forward layer. Equal-dimension comparisons isolate that distinction more closely. Learned positional parameter count increases with benchmark length; each model was allocated the requested positional capacity without changing its attention design.

Theoretical O(L) attention work for fixed head dimensions does not imply lower latency or memory throughout this tested range. The deterministic vectorized kernel prefix stores key-value products with a dimension-dependent constant. Actual complete SAKT uses the existing PyTorch MHA implementation with need_weights=False, while the isolated dense reference uses need_weights=True; this distinction is recorded for each row. The reference and complete models should not be conflated.

Peak allocated and reserved memory describe this process, include model/input/workspace and warm-up state, and are separate from the recorded baseline allocations. Forward and backward peaks are reported separately. Ten repeats on one device are engineering variability, not independent student or training-seed observations. Timings can be affected by shared device infrastructure; these are measurements for the recorded environment, not universal hardware claims.

Suggested manuscript wording: “LightSAKT preserves causal ELU+1 kernel attention and linear sequence-length work for fixed dimensions. In the tested deterministic FP32 implementation and H100 MIG environment, kernel complexity did not yield a consistent latency advantage over PyTorch SAKT. Memory and parameter advantages depended on sequence length, comparison scope and selected dimensions; the measured tables disclose these conditions.”

Evidence: results/tables/efficiency.csv, efficiency_paired_comparisons.csv, parameters.csv; experiments/efficiency/raw_timings.json and environment.json; reports/profiling/efficiency_execution.json; results/figures/efficiency_*.{csv,png,svg,pdf}.
