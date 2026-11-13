"""Generate evidence-linked asset instructions, not a manuscript or replacement paper."""
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import scipy
from common import ASSETS,ROOT,DATASETS,MODELS,DISPLAY,SHORT,csv,write


def md(frame):
    return frame.fillna('PENDING').to_markdown(index=False)


def main():
    singles=pd.read_csv(ASSETS/'source_data/verified_metric_references/individual_seed_metrics.csv')
    master=pd.read_csv(ASSETS/'source_data/verified_metric_references/master_metrics.csv')
    matrix=pd.read_csv(ASSETS/'source_data/verified_metric_references/completion_matrix.csv')
    missing=matrix[matrix.state!='VERIFIED_COMPLETED']
    csv(ASSETS/'tables/full_model_comparison/current_completion_matrix.csv',matrix)
    csv(ASSETS/'tables/full_model_comparison/remaining_seed_slots.csv',missing)
    gains=pd.read_csv(ASSETS/'tables/full_model_comparison/difficulty_absolute_AUC_gains.csv')
    paper=json.loads((ASSETS/'verification/original_paper_review.json').read_text())
    graphmap=[]
    for num,page,title in [(1,2,'An Example of Learning Process'),(2,3,'continuous assessment cycle with KT'),(3,7,'Broader View of the Proposed Framework')]:
        graphmap.append({'original_figure':num,'PDF_page':page,'original_title':title,'updated_dataset':'N/A',
          'updated_models':'N/A','new_file':'NONE','Python_program':'N/A','source_experiments':'N/A',
          'validation_status':'Conceptual/background illustration; no empirical replacement produced','replacement_suitability':'Retain/manual architecture review'})
    for num,page,title,d in [(4,16,'ASSIST 2009 results','ASSIST2009'),(5,16,'ASSIST 2015 results',DATASETS[-1]),(6,17,'Algebra 2005 results','Algebra2005')]:
        folder='EdNet_KT1_5k' if d==DATASETS[-1] else d
        available=singles[(singles.dataset==d)&(singles.seed==42)].model.tolist()
        graphmap.append({'original_figure':num,'PDF_page':page,'original_title':title,'updated_dataset':DISPLAY.get(d,d),
          'updated_models':'; '.join(available),'new_file':f'graphs/{folder}/training_performance_seed42.pdf',
          'Python_program':'python_scripts/generate_training_curves.py',
          'source_experiments':'source_data/epoch_records/completed_final_epochs.csv; original per-run history paths/SHA256 in every row',
          'validation_status':'PASS genuine epoch histories; seed42 only; missing models omitted',
          'replacement_suitability':'Conditional replacement: update caption, dataset, model coverage and objective-loss semantics'})
    graphmap.append({'original_figure':7,'PDF_page':18,'original_title':'Comparison of Model Parameters.','updated_dataset':'All three exact frozen datasets',
      'updated_models':'Actual restored configurations; pending models marked','new_file':'graphs/parameters/model_parameters.pdf',
      'Python_program':'python_scripts/generate_parameter_graph.py','source_experiments':'source_data/verified_metric_references/individual_seed_metrics.csv',
      'validation_status':'PASS counts from strict restored model instances; GKT count from one selected checkpoint',
      'replacement_suitability':'Conditional: replace old num_q100 illustration with per-dataset actual counts; no timing superiority claim'})
    for num,title in [(8,"The variation in students’ knowledge across different concepts"),(9,"The progression of a student’s knowledge state over time")]:
        graphmap.append({'original_figure':num,'PDF_page':19,'original_title':title,'updated_dataset':'All three descriptive alternatives',
          'updated_models':'SAKT, LightSAKT, LightSAKT+QD; seed42',
          'new_file':'NONE for mastery/latent projection; alternatives graphs/student_knowledge/*_prediction_sequence.pdf',
          'Python_program':'python_scripts/generate_model_comparison.py',
          'source_experiments':'source_data/verified_metric_references/anonymized_prediction_examples.csv',
          'validation_status':'Latent mastery/projection unavailable; genuine next-response probabilities available',
          'replacement_suitability':'NOT a direct mastery-figure replacement; requires new title/caption and removal of mastery assertions'})
    csv(ASSETS/'documentation/PAPER_GRAPH_MAPPING.csv',graphmap)
    (ASSETS/'documentation/PAPER_GRAPH_MAPPING.md').write_text('# Original-paper graph mapping\n\nPDF pages are one-indexed and match the printed page numbers.\n\n'+md(pd.DataFrame(graphmap))+'\n\nSupplementary GraphsD–H are graphs/model_comparisons/test_{auc,accuracy,f1,mse,bce}.{png,pdf,svg}; '
      'K is question_difficulty_comparison; L is individual_seed_test_auc. GraphJ is graphs/ablations/corrected_ablation_test_auc. '
      'Each graph has a provenance JSON with exact input hashes and caption. PNG320DPI, vector PDF/SVG. '
      'Figures4–6 show one real seed, so differing early stops need no interpolation; five-seed summaries are separate.\n')
    tablemap=[(1,4,'Some Recent Knowledge Tracing Techniques','NONE','Literature/background; not regenerated'),
      (2,8,'Overview of Datasets','dataset_overview/Table_2_dataset_overview','PASS actual processed scope; raw scope distinguished'),
      (3,13,'Training Network configurations for different datasets.','training_configurations/Table_3_training_configurations','PASS model-specific settings, budget/actual epochs'),
      (4,14,'Hyper parameter tuning results for Algebra 2005-2006','hyperparameter_tuning/Table_4_Algebra2005_LightSAKT_grid','PASS36 validation-only candidates; replaces unsupported original numbers'),
      (5,'14–15','Hyper parameter tuning results for ASSIST2009','hyperparameter_tuning/Table_5_ASSIST2009_LightSAKT_grid','PASS36 validation-only candidates'),
      (6,15,'Hyper parameter tuning results for ASSIST2015','hyperparameter_tuning/Table_6_EdNet_actual_selection_protocol','No EdNet grid exists; protocol table only, not numerical HPO replacement'),
      (7,17,'Performance comparison of different knowledge tracing models across datasets.','full_model_comparison/Table_7_full_model_comparison','15 complete groups; two partialGKT groups;12 groups pending five seeds'),
      (8,18,'Ablation Study','ablation_study/Table_8_corrected_ablation','PASS existing corrected removals; missing EdNet/norm removals unavailable')]
    tables=[]
    for n,p,t,f,status in tablemap:
        tables.append({'original_table':n,'PDF_page':p,'original_title':t,'new_file':f'tables/{f}.pdf' if f!='NONE' else 'NONE',
          'updated_datasets':'ASSIST2009,Algebra2005,exactEdNet5k (availability marked)' if n>1 else 'N/A',
          'Python_program':'python_scripts/generate_paper_tables.py' if n>1 else 'N/A',
          'source_experiments':'source_data/verified_metric_references and source_data/dataset_provenance; per-row source/hash references',
          'validation_status':status,'replacement_suitability':'Manual caption/context update required' if n>1 else 'Retain literature review'})
    csv(ASSETS/'documentation/PAPER_TABLE_MAPPING.csv',tables)
    (ASSETS/'documentation/PAPER_TABLE_MAPPING.md').write_text('# Original-paper table mapping\n\n'+md(pd.DataFrame(tables))+
      '\n\nEvery generated table has CSV, escaped LaTeX and vector PDF exports. Table7 also has an original-style metric-row/model-column layout. '
      'The complete precision/recall, unrounded means/sampleSD and individual seeds are in full_model_comparison/. '
      'CSV/LaTeX are authoritative for manuscript typesetting; standalone PDF sheets are landscape review exports.\n')
    limitations='''# Remaining scientific and execution limitations

New GPU training remains blocked: the historical shared MIG has technical headroom but no current ownership/allocation confirmation or expanded total invocation/window budget. The previous15-run EdNet authorization is consumed. Current telemetry is in verification/resource_checks/. No GPU context, new training, peer signals or HiTSKT access occurred. Retain4GiB allocator/900second windows, admission10GiB free/runtime6GiB, workerRSS4GiB, hostheadroom2GiB, pacing0.1s and stepstop30s if a new campaign is granted. This is not a fresh allocation. At least58 selected-configuration training outcomes still need to be produced, plus validation-only AlgebraGKT configuration selection. 58×900=52200seconds is merely a one-window-per-run planning envelope (14.5hours), not an approved budget or a guarantee runs fit. EdNetGKT1452skills must first be profiled under an authorized window. Exact-resume windows and configuration trials require separate accounting.

There are15complete five-seed groups of27. GKT/PAM and GKT-MHA onASSIST2009 each have one newly CPU-tested, validation-selected seed42 checkpoint; four other seeds each remain. AlgebraGKT configurations still need selection. EdNetDKT/DKT+/DKVMN/GKT/GKT-MHA and SAKT+QD onallthree datasets need training. No two-seed/one-seed performance is passed off as a five-seed aggregate. All72 ASSIST2009 GKT grid trials are configuration candidates, not72independent seeds. Old primary single-seed runs are not equivalent to selected-final protocols. Old ASSIST2015 incomplete status markers are historical, not active jobs or current primary data.

Item difficulty adds independently identified item information and fitted aggregate-estimation capacity absent from skill-only baselines. SAKT+QD, the critical feature-matched control, is untrained onallthree datasets. GKT and GKT-MHA are simplified repository implementations that omit erase-add gating/MLP batch normalization; they are not canonical published reproductions. Hyperparameter search exposure is unequal. GKTbatch8 makes more updates than batch64 models, even with the same50epoch ceiling. Collected timings/device/pacing policies are not matched efficiency measurements.

No held-out-label or own-student self-label leakage was identified in repeated actual-data perturbation/sidecar reconstruction tests. This supports only the documented OFFLINE fixed student split. Both skill and item rates for train targets exclude the whole own student. Held-out features use training labels only, item smoothing10 and skill Beta(1,1), with unseen-item correctionzero. Full training-corpus estimates can contain future-dated interactions of OTHER training students. EdNet temporal audit flags99.6012%validation/99.5887%test targets for later training observations in either aggregate. No chronological/online availability claim is supported.

ASSIST2009 uses authentic problem_id; Algebra uses a (problem,step) composite and a local split of the original training subset; EdNet uses authentic question_id and composite-tag skill inputs. EdNet is exactly5000uniformly selected students, seed20261104, first presented question per solving session chosen before answer filtering, NOT fullKT1. Subset selection, record counts, masking, vocabularies and fixed split hashes are unchanged. Algebra's26.29%test unseen-item fraction and EdNet's0.165% differ substantially; item-level gains need contextual interpretation. Long histories dominate pooled target metrics; EdNet's largest validation student contributes19.11%of targets. Unseen identities map according to the frozen training vocabulary.

Five initialization seeds share ONE student split per dataset. SD measures optimization variability, not independent cohort uncertainty. Positive matched-seed differences do not establish population significance. Allnine exact two-sided n5 Wilcoxon tests fail Holm-adjusted0.05; no minimum attainable n5 two-sided p below0.0625 exists without more independent observations. The old manuscript W78/p0.00048 is not reproduced; grid points/epochs must not be treated as independent trials. Released test results are reporting-only, never configuration selection. Independent-cohort/split and chronological evaluation remain future work.

The old manuscript's latent-space/mastery figures cannot be certified from these next-response artifacts. The new student examples are saved real probabilities, not verified mastery or latent projections; questions/skills may change. No artificial monotonic progression or fake embeddings were produced. No AI image generation, outcome simulation, curve interpolation, missing-value imputation or fabricated confidence intervals was used.

Thirty historical baseline prediction files contain original GPU outputs, whereas the November10 table used prior independently verified CPU replay scores. New arithmetic on the preserved actual files differs by at most1.85e-8 in the checked AUC/accuracy/F1/BCE metrics. Every difference is tabulated in verification/metrics_verification/prior_CPU_vs_saved_array_differences.csv. Original records/reports are unchanged. This package consistently uses fresh float64 calculations on each referenced saved test array. Native selected validation AUC is reported separately. BCE clamps probabilities to[1e-7,1−1e-7]; MSE is mean squared probability error (Brier score); classification threshold≥0.5.
'''
    (ASSETS/'documentation/REMAINING_LIMITATIONS.md').write_text(limitations)
    corrections='''# Suggested corrections for manual manuscript revision

This document does not edit or regenerate the manuscript. The exact25-page PDF remains unchanged.

- Abstract/page1 and Results/pages16–17: replace claims that original LightSAKT uniformly beats DKT/DKT+/DKVMN/SAKT. Verified original LightSAKT is below DKT onASSIST2009 and DKT+ onAlgebra2005, and below SAKT onAlgebra2005. The item-augmented model is a separate model, not a retroactive architecture result.
- Pages14–17, Tables4–7 and Figures4–6: the manuscript's original LightSAKT testAUC0.8658/0.9175/0.8923 and associated curves/metrics are not reproduced by the corrected causal protocol. Use the measured tables here; do not present the difference as a new drop caused by difficulty or combine old/new protocols. Provenance and the historical causal-correction issue are retained in prior reports.
- Page8 Table2 and pages8–9 preprocessing: replace unsupported raw/processed counts (including Algebra3967records) with actual counts and distinct raw scopes. Current preprocessing uses one-overlap windows retaining every per-student n−1target, with context reset per window; it does not simply truncate each student to100events. Clarify student-disjoint70/10/20 split, training-only skill vocabulary/UNK and skill-token versus independently identified item semantics.
- Page13 Table3: uniformLR0.005/dimensions/head/dropout/length assumptions conflict with actual model-specific selected settings. Algebra length200, variable model widths/LRs and fixed seeds/budgets are explicit in newTable3. EdNet core settings were prespecified, not selected from a fabricated36point grid.
- Pages9–11 architecture: current q indexes skill/composite-skill vocabulary, not independently identified question IDs. There is one causalELU+1 linear-attention encoder. Positional embeddings apply to past interaction embeddings only; question queries are not position-encoded despite the wording onpage11. Residual is attention_output + past_interaction_embedding + query_embedding; FFN widths are d→d/2→d, not d→d→d. Padding/causality are explicit. QD leaves that structure unchanged and adds one scalar to the readout logit. External fitted rates add information/capacity beyond that one trainable scalar.
- Page18 Figure7 and pages17/20 efficiency: use actual dataset-specific counts. Parameter totals do not prove matched runtime or memory reductions, since configurations/hardware/pacing/search differ and fitted statistics have storage/cost. Do not carry forward the fixednum_q100 chart as measured final-model complexity. Dense attention weights are optional visualization only.
- Page18 Wilcoxon: old W78/p0.00048 cannot support current claims. Replace with genuine seed-matched n5 observations/exact pvalues/Holm9 and the conditional fixed-split limitation. Grid candidates and epochs are not independent repetitions.
- Page18 Table8 and page20 error analysis: removing position does not uniformly hurt; single-head can improve Algebra. No completed corrected layer-normalization-removal or EdNet component-removal experiment exists. no_regularization is an alias for no_dropout, not a second independent result. Additional item information is a separate comparison.
- Pages1/8/12/15/16/21 ASSIST2015 conclusions: primary third dataset is now the exactEdNet5k first/session subset. ASSIST2015 remains historical and cannot provide a valid question-difficulty comparison without independent itemIDs. Do not relabel oldASSIST2015 curves asEdNet.
- Page19 Figures8/9 and pages19–21 deployment/mastery: probability predictions do not establish mastery or valid monotonic knowledge acquisition. Use only the explicitly relabelled genuine examples, or omit unverified figures. No real-time/online claim follows from offline full-training-corpus item statistics. Independent cohort and chronological studies plus feature-matched SAKT+QD remain necessary.
- Cross-paper leaderboard comparisons toAKT/DTransformer cannot be causal claims across different datasets/splits/features/settings. No such models were retrained here. Mark missing nine-model groupsPENDING and avoid publication-readiness assertions.
'''
    (ASSETS/'documentation/SUGGESTED_MANUSCRIPT_CORRECTIONS.md').write_text(corrections)
    display=pd.read_csv(ASSETS/'tables/full_model_comparison/Table_7_full_model_comparison.csv')
    gains_focus=gains[gains.baseline.isin(['SAKT','LightSAKT'])]
    report=f'''# Verified paper-update assets — evidence audit performed13November2026

Asset directory retains the requested20261112 identifier. This is an asset/research report, NOT a rewritten manuscript.

The exact original PDF is25pages, with the expected title, verified independently byPyMuPDF andpypdf and reviewed visually onallpages. SHA256 `{paper['sha256']}`. Figures4/5/6 are onpages16/16/17; Figure7 is onpage18; mastery illustrations8/9 are onpage19. Original paper and historical artifacts unchanged.

**New training runs:0.** No verified current allocation ownership or expanded totalGPUbudget was found. Two already-selected GKT checkpoints were evaluated onCPU, without retraining or new selection. We reused all75existing seed results, audited40existing component ablations and72existing LightSAKT grid candidates. Current verified seed results:{len(singles)}; complete five-seed groups:15/27; remaining verified seed slots:{len(missing)}. Allnine models are NOT complete onallthree datasets.

## Complete available numerical comparison

Mean±sampleSD, exactly seeds42,123,2026,3407,7777. PENDING means no five-seed aggregate; single-seed GKT evidence is separate. ValidationAUC uses the selected native epoch, test metrics fresh arithmetic on real referenced saved predictions. No original value was overwritten.

{md(display)}

## Genuine test-AUC differences (absolute AUC units)

{md(gains_focus)}

LightSAKT+QD improves testAUC over both SAKT andoriginalLightSAKT inallfive matched seeds ofeachdataset. That is an offline fixed-split finding, not architectural superiority or population significance. SAKT+QD remains untrained. OriginalLightSAKT does not beat every availablebaseline. ExactpairedWilcoxon n5 tests withHolm9 produce no rejection at0.05; old manuscript significance is not reused.

## Current missing work

{md(missing.groupby(['dataset','model','state'],sort=False).size().reset_index(name='seed slots'))}

The two selectedASSIST2009GKT trainings are reused. Their per-seed CPUvalidation/test evidence is in verification/checkpoint_verification/new_GKT_CPU_replays.json and the individual-seedCSV. Neither partial group receives a mean/SD. MissingAlgebraGKTselection andEdNetGKTfeasibility remain distinct blockers from missing currentresourceauthorization. No old scheduler was resumed.

## Verification and metric definitions

All referenced best checkpoints were hash-checked and strictly restored onCPU; selected epoch equals the firstmaximum nativevalidationAUC in complete loggedhistory. Data/splitSHA256, studentdisjointness, exactper-studentn−1labels/order, finiteweights, and originalvocab/masks were checked. Storedtestprobabilities were independently recalculated using rank-sumAUC and manualconfusion counts, crosschecked againstsklearn. Float64MSE/BCE/precision/recall added; threshold0.5, BCEclip1e−7, sampleSDddof1. Corrected ablationweights and72realgridrecords were verified. Allsourceinputs are listed in verification/source_fingerprints.json.

Repeated real-data labelperturbation and frozenfeature reconstruction found no heldoutlabel or ownstudent targetleakage under the documentedofflineprotocol. Futureobservations ofOTHERtrainstudents still enter fittedrates; chronologicalavailability is NOT validated. See current_leakage/ and copiedtemporalprovenance. Attentioncode unchanged:onecausalELU+1encoder; QDadds alpha×[logit(smootheditemsuccess)−logit(smoothedskillsuccess)] toreadoutlogit. Trainfeaturesexcludewholeownstudent; heldoutestimatesfittraininglabelsonly; unseenitemsreturnzero.

## Graphs/tables and reproduction

All numerical graphs were executed withPython,pandas,NumPy andMatplotlib. Actualseed42epochlogs are used withoutsmoothing,interpolation,extendingearlystops orfabricatingmissingcurves. DKT+trainingobjectiveincludes penalties and islabelledaccordingly. Crossmodelbarsrequirefivecomplete seeds. Studentexamplesareactualanonymizedtestpredictions,notmastery. PNG320DPI plusvectorPDF/SVG; eachhasinputhash/captionprovenance. Re-executionchecks byteidenticalplotfiles. Tables2–8haveCSV/LaTeX/PDFexports; Table6is anhonestEdNetprotocoltable because nogridexists. See documentation/PAPER_GRAPH_MAPPING.md andPAPER_TABLE_MAPPING.md for exactlocations/programs/sources/replacementconditions.

Portable programs read compactgenuineinputextracts and do notneedweights/datasets toredrawassets. Fullcheckpoint/predictionreaudit requires the separatelypreserved localdata andcheckpoints. Thosebinaries,rawdatasets,fullstudentIDs andoriginalpaper are not inGit/ZIP. Snapshotpaths/hashes/sourcecommitsremaintraceable. See REPRODUCE.md. Datasetstatistics/preprocessing/splitprovenance are under source_data/dataset_provenance/. Raw/procsubsetscopes andmodelsettings differ; these are notfullEdNetleaderboardmeasurements.

## Scope and manuscript safety

The manuscriptwasnotmodifiedorregenerated. HiTSKTwasnotaccessed,resumedormodified; noGPUtrainingorpeerprocesssignals. HistoricalASSIST2015remainsseparate. Alllimitations,featurematched/independentcohort/chronologicalgaps,unequalHPOandGKTimplementationdifferencesare in documentation/REMAINING_LIMITATIONS.md. Suggestedcorrectionsare separate fromthepaper. Publicationstatementsmustdescribe15completegroups,notacomplete9×3campaign.
'''
    (ASSETS/'REPORT.md').write_text(report)
    short=[]
    for d in DATASETS:
        g=master[master.dataset==d].set_index('model')
        short.append(f"{DISPLAY.get(d,d)}: originalLightSAKT testAUC {g.loc['LightSAKT','test_auc_mean']:.6f}; +QD {g.loc['LightSAKT + Question Difficulty','test_auc_mean']:.6f} (gain {g.loc['LightSAKT + Question Difficulty','test_auc_mean']-g.loc['LightSAKT','test_auc_mean']:+.6f}).")
    (ASSETS/'PROFESSOR_SUMMARY.md').write_text('# Professor-ready progress summary\n\n'+
      'We verified the original25-page paper and generated replacement tables/plots exclusively from genuine saved experiments. '+
      ' '.join(short)+' Gains over originalLightSAKT andSAKT occur inallfive matched seeds, with lower MSE/BCE under the offline protocol. '+
      'They do not yet establish attention-architecture superiority: the SAKT+difficulty feature-matched control remains untrained, and independent-cohort/chronological evaluation is still needed. '+
      f'We reused75existing seeds and CPU-evaluated two selectedGKTcheckpoints;15of27five-seedgroups are complete and{len(missing)}seedresults remain. '+
      'No newtraining was launched because currentGPUallocation ownership and an expandedcampaignbudget are unconfirmed. '+
      'The package includes honestPENDINGcells, uncertainty, fullprovenance, trueepochcurves and manuscriptcorrectionnotes; thepaper itself isunchanged. '+
      'Exactn5Wilcoxon/Holmtests do not establish significance at0.05. These are verified interim assets, not a publication-ready complete nine-model comparison.\n')
    env={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__,'scipy':scipy.__version__}
    write(ASSETS/'verification/python_environment.json',env)
    (ASSETS/'python_scripts/requirements-plotting.txt').write_text('\n'.join(f'{k}=={v}' for k,v in env.items() if k!='python')+'\nPillow\ntabulate\n')
    (ASSETS/'REPRODUCE.md').write_text('''# Reproduce without training

From this asset directory, with the recordedPython/NumPy/pandas/Matplotlib/SciPy stack andPillow/tabulate:

```bash
python -B python_scripts/generate_all_paper_graphs.py
python -B python_scripts/generate_paper_tables.py
python -B python_scripts/verify_assets.py --reproduce
```

These programs use only bundled genuine CSV/JSON extracts. No raw dataset, GPU, weights, random outcome generator or model trainer is used. Fonts DejaVuSans, SVGhashsalt andPDF/SVGtimestamps arefixed to make bytecomparison meaningful in the recorded environment. Graphprovenance records map every output toinputSHA256. Changing software/fonts may changebytes while retainingnumerics. Do not silently bless changeddata.

Full originalsource audit (localresearchcheckout only, not aportable data redistribution):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B python_scripts/collect_verified_sources.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B python_scripts/verify_leakage.py
python -B python_scripts/verify_assets.py --sources --reproduce
```

Those CPUprograms need the localprocessed/rawfiles/checkpoints referenced bySHA256; they nevertrain orwritehistoricalinputs. The separate replay_reusable_GKT.py evaluated exactlytwo alreadyfrozen selectedcheckpoints, caching actualprobabilities underignored.tools; repeatedexecution verifies/reuses itscache and doesnotretrain. Missingmodeltraining is NOTimplementedorimplicitlyauthorized here. PyMuPDF/pypdf wereused onlytoinspectoriginalPDF; they are notneededfordrawing thesegraphs. OriginalPDF isreferenced, notredistributed inthispackage.

Run package_assets.py --destination /your/export/directory tobuildandCRC/SHA256verify aportableZIP. It excludesweights,rawdatasets andfullpredictionarrays. Thecopiedanonymizedexamples are a smallgenuineextract with nostudentIDs, notsimulateddata. Scriptsdo notmodifythepaper. Toextendthe inventoryafternewauthorizedtraining, versiontheexperimentmanifest andindependentlyverify thenewseedartifacts beforeaggregating. Neverfillmissingcells oruse testresultsforselection.
''')
    (ASSETS/'documentation/MANUAL_REPLACEMENT_GUIDE.md').write_text('''# Manual manuscript replacement guide

1. Preserve the SHA256-identified25-pageoriginalPDF. No manuscript replacement is performed bytheseprograms.
2. Use PAPER_GRAPH_MAPPING.md andPAPER_TABLE_MAPPING.md. Figure4→ASSIST2009seed42curves; Figure5→exactEdNet5kfirst/sessioncurves; Figure6→Algebra2005seed42curves; Figure7→actualper-dataset horizontalparameters. Use vectorPDF/SVGwherepossible; PNG320DPIprovided.
3. Change captions tosayseed42, actualearlystopping, objectives(DKT+penalties), missingmodels andfrozen subset. Five-seed variability belongsinthefinalbars/tables. Neverclaimcurvesareaverages.
4. Use Table7_original_structure fordataset/metricrows andmodelcolumns, orfull_model_comparison forreadability. KeepPENDING,1/5statusesandtheGKTimplemented-variantqualifier. Do notcopyanindividualseed intotheaggregatecells. Table6isactualselectionprotocol,notanEdNetHPOgrid. Table8separatescomponentremovals frominputaugmentation andhasunavailableEdNet/normremovals.
5. Review SUGGESTED_MANUSCRIPT_CORRECTIONS.md; oldAUC,significance,universalbaselinevictory,mastery,efficiencyandASSIST2015claimsarenotsupportedbythisprotocol. NoFigs8/9masteryreplacementhasbeencertified. Optionalstudentexamplesrequirea newnext-response caption andcannotretainmasteryinterpretation.
6. Cite fixedstudentdisjointsplit,one-overlapwindowmask,train-onlyvocabulary,itemsemantics,whole-own-studentexclusion,offlinefutureOTHERtraininformation,HPOdifferencesandmissingfeaturecontrol. Reportmean±sampleSDacrossfiveinitializations,notconfidenceintervals/independentcohorts.
7. Verify regeneration andarchivechecks beforemanual insertion. Publication oftheseassets isnot completionofthependingtrainingcampaign. ConsultREPORT.md/currentcompletionmatrix;resourceconfirmationremainsrequired.
''')
    print('Generated mappings, genuine results report, professor summary and separate manuscript corrections.')


if __name__=='__main__':main()
