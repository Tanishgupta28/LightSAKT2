"""Generate manuscript-ready measured findings without claiming theoretical speed."""
from pathlib import Path
import pandas as pd


def main():
    frame=pd.read_csv('results/tables/efficiency.csv')
    keys=['scope','length','batch','operation']
    light=frame[frame.model=='LightSAKT'].set_index(keys)
    sakt=frame[frame.model=='SAKT'].set_index(keys)
    pairs=light[['latency_ms_mean','peak_allocated_bytes','parameters']].join(
        sakt[['latency_ms_mean','peak_allocated_bytes','parameters']],lsuffix='_light',rsuffix='_sakt')
    pairs['latency_ratio_light_over_sakt']=pairs.latency_ms_mean_light/pairs.latency_ms_mean_sakt
    pairs['peak_allocated_ratio_light_over_sakt']=pairs.peak_allocated_bytes_light/pairs.peak_allocated_bytes_sakt
    pairs['parameter_ratio_light_over_sakt']=pairs.parameters_light/pairs.parameters_sakt
    pairs.to_csv('results/tables/efficiency_paired_comparisons.csv')
    text='# Fresh controlled efficiency results\n\n'
    text+=f"Executed {len(frame)} cases ({len(pairs)} matched comparisons) on the H100 MIG partition. Sequence lengths 50/100/200/500/1000; batches 1/8; equal attention, equal complete dimensions, and documented primary complete models; forward and forward/backward; FP32, three warm-ups and ten timed repetitions with synchronization. Models are newly initialized for compute measurement; this is not predictive-performance evidence. Raw timings and hardware/source manifests are under experiments/efficiency/.\n\n"
    text+=f"LightSAKT had lower mean latency in {int((pairs.latency_ratio_light_over_sakt<1).sum())}/{len(pairs)} measured matched comparisons, and lower peak allocated memory in {int((pairs.peak_allocated_ratio_light_over_sakt<1).sum())}/{len(pairs)} comparisons. The data do not support a general claim that the current LightSAKT implementation is faster than SAKT. No architecture was changed to improve these observations.\n\n"
    text+='|Scope|Length|Batch|Operation|LightSAKT ms|SAKT ms|Latency ratio|Peak allocated memory ratio|\n|---|---|---|---|---|---|---|---|\n'
    selected=pairs.reset_index();selected=selected[(selected.batch==8)&(selected.length==1000)]
    for r in selected.itertuples():
        text+=f'|{r.scope}|{r.length}|{r.batch}|{r.operation}|{r.latency_ms_mean_light:.3f}|{r.latency_ms_mean_sakt:.3f}|{r.latency_ratio_light_over_sakt:.3f}|{r.peak_allocated_ratio_light_over_sakt:.3f}|\n'
    params=pd.read_csv('results/tables/parameters.csv')
    text+='\nActual primary-model parameter counts:\n\n|Dataset|LightSAKT|SAKT|Ratio|\n|---|---|---|---|\n'
    for dataset,data in params.groupby('dataset'):
        light_n=int(data[data.model=='LightSAKT'].parameters.iloc[0]);sakt_n=int(data[data.model=='SAKT'].parameters.iloc[0])
        text+=f'|{dataset}|{light_n}|{sakt_n}|{light_n/sakt_n:.3f}|\n'
    text+='\nThe primary configurations use different widths; their parameter differences cannot be attributed solely to kernel attention or the reduced feed-forward layer. Equal-dimension comparisons isolate that distinction more closely. Learned positional parameter count increases with benchmark length; each model was allocated the requested positional capacity without changing its attention design.\n\n'
    text+='Theoretical O(L) attention work for fixed head dimensions does not imply lower latency or memory throughout this tested range. The deterministic vectorized kernel prefix stores key-value products with a dimension-dependent constant. Actual complete SAKT uses the existing PyTorch MHA implementation with need_weights=False, while the isolated dense reference uses need_weights=True; this distinction is recorded for each row. The reference and complete models should not be conflated.\n\n'
    text+='Peak allocated and reserved memory describe this process, include model/input/workspace and warm-up state, and are separate from the recorded baseline allocations. Forward and backward peaks are reported separately. Ten repeats on one device are engineering variability, not independent student or training-seed observations. Timings can be affected by shared device infrastructure; these are measurements for the recorded environment, not universal hardware claims.\n\n'
    text+='Suggested manuscript wording: “LightSAKT preserves causal ELU+1 kernel attention and linear sequence-length work for fixed dimensions. In the tested deterministic FP32 implementation and H100 MIG environment, kernel complexity did not yield a consistent latency advantage over PyTorch SAKT. Memory and parameter advantages depended on sequence length, comparison scope and selected dimensions; the measured tables disclose these conditions.”\n\n'
    text+='Evidence: results/tables/efficiency.csv, efficiency_paired_comparisons.csv, parameters.csv; experiments/efficiency/raw_timings.json and environment.json; reports/profiling/efficiency_execution.json; results/figures/efficiency_*.{csv,png,svg,pdf}.\n'
    Path('paper/EFFICIENCY_RESULTS.md').write_text(text)
    corrections=Path('PAPER_CORRECTIONS.md');existing=corrections.read_text()
    marker='## Fresh efficiency evidence (October 2026)'
    if marker not in existing:
        corrections.write_text(existing+'\n'+marker+'\n\nPaper pages 18–19, Figure 8/efficiency, and pages 21–23, discussion: the historical architecture attribution and uncontrolled runtime statements do not establish LightSAKT speed or parameter efficiency. Fresh evidence now exists: the 120-case exclusive FP32 benchmark and dataset-specific parameter counts. The measured matched latency comparisons contradict an unconditional speed advantage; memory and parameter differences depend on scope and dimensions. The original architecture was preserved. Corrected numerical tables and manuscript-ready wording are generated in paper/EFFICIENCY_RESULTS.md from results/tables/efficiency.csv, efficiency_paired_comparisons.csv and parameters.csv; hardware, raw timings and execution evidence are in experiments/efficiency/ and reports/profiling/efficiency_execution.json. Original PDF and historical claims remain preserved.\n')
    print('Genuine efficiency pairs',len(pairs),'lower LightSAKT latency',int((pairs.latency_ratio_light_over_sakt<1).sum()))


if __name__=='__main__':main()
