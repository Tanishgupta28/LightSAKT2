"""Generate manuscript-ready text and artifact references from saved results."""
import json
from pathlib import Path
import pandas as pd
from src.datasets.preprocess import sha256
from src.training.engine import write_json


def readable_table(frame):
    columns=list(frame.columns)
    lines=['|'+'|'.join(columns)+'|','|'+'|'.join(['---']*len(columns))+'|']
    for values in frame.itertuples(index=False,name=None):
        formatted=['pending' if pd.isna(v) else f'{v:.6f}' if isinstance(v,float) else str(v) for v in values]
        lines.append('|'+'|'.join(formatted)+'|')
    return '\n'.join(lines)


def main():
    if Path('reports/light_grid_verification.json').exists():
        # This gate exists only after all 108 settings completed. Re-derive the
        # selected-settings table from immutable evidence, never test scores.
        from scripts.light_grid_results import generate
        generate()
    lines=['# Corrected results text — generated from experiment artifacts','',
           'This document accompanies the preserved manuscript; it does not overwrite historical claims. Incomplete experiments remain explicitly pending.','',
           '## Corrected protocol','',
           'We split students into training (70%), validation (10%), and testing (20%) before sequence chunking. Split seed 42 was prespecified for the rebuilt protocol. Chronological response-event order was preserved, and vocabularies were fitted on training students. Histories were chunked at lengths 100, 100, and 200 for ASSIST2009, ASSIST2015, and Algebra2005, respectively, with one shared interaction at boundaries and no carried latent state. All valid next-response targets were retained; padding contributed to neither loss nor metrics.','',
           'LightSAKT retained interaction and target-question embeddings, learned positions, multi-head learned projections, the ELU+1 kernel, residual connections, layer normalization, dropout, reduced-width feed-forward layers, and sigmoid prediction. Causal key and key–value prefixes share the same boundary and exclude invalid keys. Historical global aggregation was corrected because it exposed future responses.','',
           'Adam used beta values (0.9, 0.999), epsilon 1e-8, and zero weight decay. Training lasted at most 50 epochs with patience five on validation AUC. Test data were excluded from hyperparameter and checkpoint selection. Each model variant received 36 validation-tuning configurations per dataset. The LightSAKT grid comprised learning rates {0.0005, 0.001, 0.005}, widths {32, 64, 128}, heads {2, 4}, and dropout {0.1, 0.2}. Final training seeds were {42, 123, 2026, 3407, 7777}; selected tuning checkpoints were reused for seed 42.','',
           'The GKT PAM and MHA rows refer to the corrected historical simplified implementation and do not establish canonical published GKT reproducibility. Local dataset hashes and processing are verified, while independent publisher byte identity and redistribution licensing remain unresolved; Algebra covers the available labelled training subset.','',
           '## Dataset statistics','']
    stats=pd.read_csv('results/tables/Table_2.csv')
    lines += [readable_table(stats[['dataset','raw_rows','clean_interactions','students','unique_skill_identifiers','num_q']]),'',
              '## LightSAKT validation selection','']
    selected=Path('results/tables/LightSAKT_selected_configurations.csv')
    if selected.exists():
        candidates=pd.read_csv(selected)
        lines += [readable_table(candidates[['dataset','learning_rate','embed_dim','num_heads','dropout','validation_auc']]),'',
                  'These configurations maximize validation AUC among all 36 prespecified candidates per dataset, with a lexical run-ID tie-break. The selection records are frozen before final evaluation. Validation scores are separate from held-out test performance; no test metric determined these choices. Run IDs and frozen split hashes are in LightSAKT_selected_configurations.csv and experiments/selections/.','']
    else:lines += ['The complete prespecified grid remains pending; no configuration is selected using test results.','']
    lines += ['## Matched final comparisons','']
    summary_path=Path('results/tables/Table_7.csv')
    try:summary=pd.read_csv(summary_path)
    except pd.errors.EmptyDataError:summary=pd.DataFrame()
    if summary.empty:lines += ['Final selected-configuration repeated-seed experiments are pending; primary single-seed results are recorded separately in FINAL_RESEARCH_REPORT.md.','']
    else:
        lines += [readable_table(summary[['dataset','model','completed_seeds','auc_mean','auc_std','accuracy_mean','f1_mean','mse_mean','bce_mean']]),'',
                  'Means summarize every completed prespecified seed. Sample standard deviations and Student-t intervals describe conditional initialization variability on one fixed student split; they do not measure new-student sampling uncertainty. Partial seed counts are not complete five-seed results.','']
        for dataset,data in summary.groupby('dataset'):
            light=data[data.model=='LightSAKT']
            if light.empty:continue
            light=light.iloc[0]
            for row in data[data.model!='LightSAKT'].itertuples():
                if light.completed_seeds==5 and row.completed_seeds==5:
                    difference=light.auc_mean-row.auc_mean
                    lines.append(f'{dataset}: LightSAKT mean test AUC minus {row.model} was {difference:+.6f} across the five prespecified seeds (Table 7).')
        lines.append('')
    lines += ['## Ablations and paired inference','',
              'Default LightSAKT reused the final-model evidence. Component variants removed dropout, positional contributions, multiple heads, or attention. Without attention, the attention contribution was zero while the original embedding residual and feed-forward pathway remained. By explicit author instruction, “without regularization” was equivalent to “without dropout,” because the original model supplied no other explicit regularizer and weight decay was zero. These two rows reuse the same evidence and are not independent observations.','',
              'Relative changes and sample standard deviations are generated in Table 8 and paired_ablations.csv. No outcome was required to favor the default architecture. Paired signed-rank comparisons used matching model seeds and the same frozen student split, with exact sign enumeration, zero omission, midranks for ties, and Holm adjustment across the 18 prespecified comparisons. Five nonzero pairs permit a minimum two-sided exact p-value of 0.0625, so conventional significance cannot be established with that design even before correction.','',
              '## Efficiency and interpretation','']
    if Path('results/tables/efficiency.csv').exists():
        from scripts.efficiency_evidence import manuscript_findings
        lines += [manuscript_findings(pd.read_csv('results/tables/efficiency.csv'),
                                     pd.read_csv('results/tables/parameters.csv')), '']
    else:
        lines += ['The ELU+1 prefix mechanism has linear sequence-length work at fixed head dimensions. Controlled runtime, memory and parameter measurements remain pending; no measured efficiency advantage is established.', '']
    lines += [
              'Kernel heatmaps reconstruct normalized causal ELU+1 weights from selected trained checkpoints and genuine held-out sequences; requesting weights leaves predictions unchanged. Skill trajectories are predicted next-response correctness conditional on observed history, rather than directly observed mastery. Historical TransDKT/TransDKT+ figures are not relabelled as LightSAKT.','',
              '## Artifact sources','',
              'Tables: results/tables/. Numerical figure data: results/figures/*.csv, experiments/efficiency/raw_timings.json, and experiments/case_studies/*.npz. Each run records the checkpoint, split hash, configuration, seed, code version, predictions, and metrics. PAPER_CORRECTIONS.md documents page-specific discrepancies with suggested wording.']
    Path('paper/CORRECTED_RESULTS.md').write_text('\n'.join(lines)+'\n')
    trace=pd.read_csv('reports/PAPER_TRACEABILITY.csv')
    runs=pd.read_csv('results/tables/all_verified_runs.csv')
    for index,row in trace.iterrows():
        item=row['item'];state=row['execution_status']
        if item=='Table 2':state='executed; raw and processed statistics verified'
        elif item=='Table 3':state='executed; configured primary protocol, distinct from selected final settings'
        elif item=='Figures 1-4':state='executed; corrected conceptual diagrams, not empirical results'
        elif item in ['Table 4','Table 5','Table 6']:
            dataset={'Table 4':'Algebra2005','Table 5':'ASSIST2009','Table 6':'ASSIST2015'}[item]
            n=len(runs[(runs.dataset==dataset)&(runs.model=='LightSAKT')&(runs.group=='grid')]);state=f'{n}/36 configurations completed'
        elif item=='Table 7':state=f'{len(runs[runs.group=="final"])}/105 final seed slots completed (21 checkpoint reuse slots)'
        elif item=='Table 8':state=f'{len(runs[runs.group=="ablation"])}/75 ablation slots completed (15 equivalent aliases); default reuses final LightSAKT'
        elif item in ['Figure 5','Figure 6','Figure 7']:
            dataset={'Figure 5':'Algebra2005','Figure 6':'ASSIST2009','Figure 7':'ASSIST2015'}[item]
            n=len(runs[(runs.dataset==dataset)&(runs.group=='primary')]);state=f'{n}/7 primary model variants completed; plots generated by scripts/analyze.py'
            trace.loc[index,'source_artifacts']=f'results/raw/{dataset}-*/history.json; results/figures/{item.replace(" ","_")}.csv; scripts/analyze.py'
        elif item=='Figure 8':state='executed' if Path('results/tables/parameters.csv').exists() else 'pending efficiency stage'
        elif item=='Figure 9':state=f'{len(list(Path("experiments/case_studies").glob("*_final_42_attention.npz")))}/3 final trained causal maps executed'
        elif item=='Figures 10-11':state='executed' if Path('experiments/case_studies/ASSIST2009_final_42_skill_probabilities.npz').exists() else 'pending selected final ASSIST2009 checkpoint'
        elif item=='Significance':state='executed; paired seed limitations disclosed' if Path('results/tables/significance.csv').exists() and Path('results/tables/significance.csv').stat().st_size>5 else 'pending completed matched final pairs'
        elif item=='Efficiency':
            evidence=Path('reports/efficiency_verification.json')
            if evidence.exists():
                verified=json.load(open(evidence))
                state=f'executed; {verified["cases"]} cases, {verified["matched_comparisons"]} matched comparisons, {verified["timed_repetitions_verified"]} raw repetitions verified'
                trace.loc[index,'source_artifacts']='experiments/efficiency/raw_timings.json; results/tables/efficiency.csv; reports/efficiency_verification.json; paper/CORRECTED_RESULTS.md; paper/EFFICIENCY_RESULTS.md'
            else:state='executed' if Path('results/tables/efficiency.csv').exists() else 'pending isolated benchmark stage'
        trace.loc[index,'execution_status']=state
    trace.to_csv('reports/PAPER_TRACEABILITY.csv',index=False)
    artifacts=[]
    for folder in ['results/tables','results/figures','experiments/case_studies','experiments/efficiency']:
        for path in sorted(Path(folder).glob('*')):
            if path.is_file():artifacts.append({'path':str(path),'sha256':sha256(path)})
    artifacts.append({'path':'reports/PAPER_TRACEABILITY.csv','sha256':sha256('reports/PAPER_TRACEABILITY.csv')})
    write_json('paper/RESULT_ARTIFACTS.json',artifacts)


if __name__=='__main__':main()
