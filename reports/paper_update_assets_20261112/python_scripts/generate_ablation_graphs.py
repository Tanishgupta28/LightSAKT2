"""Existing corrected component removals plus separately labelled item-information addition."""
import os
os.environ['PAPER_GENERATOR']='generate_ablation_graphs.py'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from common import ASSETS, DATASETS, csv, save_figure


def main():
    ab='source_data/verified_metric_references/corrected_ablation_seeds.csv'
    base='source_data/verified_metric_references/individual_seed_metrics.csv'
    a=pd.read_csv(ASSETS/ab);b=pd.read_csv(ASSETS/base)
    variants=['default','no_attention','no_dropout','no_position','single_head','plus_question_difficulty']
    records=[]
    for d in DATASETS:
        for v in variants:
            if v in ['default','plus_question_difficulty']:
                model='LightSAKT' if v=='default' else 'LightSAKT + Question Difficulty'
                g=b[(b.dataset==d)&(b.model==model)]
            else:g=a[(a.dataset==d)&(a.variant==v)]
            row={'dataset':d,'variant':v,'seed_count':len(g),'status':'VERIFIED_5_SEEDS' if len(g)==5 else 'UNAVAILABLE'}
            for m in ['test_auc','test_accuracy','test_f1','test_mse','test_bce']:
                row[m+'_mean']=g[m].mean() if len(g)==5 else None
                row[m+'_std']=g[m].std(ddof=1) if len(g)==5 else None
            records.append(row)
    csv(ASSETS/'source_data/verified_metric_references/ablation_summary.csv',records)
    fig,axes=plt.subplots(1,3,figsize=(13,3.6))
    names=['Default','No attention','No dropout','No positions','Single head','+ Question difficulty']
    frame=pd.DataFrame(records)
    for ax,d in zip(axes,DATASETS):
        group=frame[frame.dataset==d].set_index('variant')
        for i,v in enumerate(variants):
            r=group.loc[v]
            if r.seed_count==5:ax.bar(i,r.test_auc_mean,yerr=r.test_auc_std,capsize=3,color='#17becf' if v=='plus_question_difficulty' else '#7fa5bf')
            else:ax.text(i,.04,'UNAVAILABLE',rotation=90,ha='center',transform=ax.get_xaxis_transform(),fontsize=6)
        ax.set_xticks(range(6),names,rotation=50,ha='right');ax.set(title=d.replace('EdNetKT1_5k_first','EdNet-KT1 (5k first/session)'),ylabel='Test AUC',ylim=(.5,.9));ax.grid(axis='y')
    fig.tight_layout()
    save_figure(fig,'graphs/ablations/corrected_ablation_test_auc',[ab,base],
      'Actual five-seed component-removal experiments on ASSIST2009/Algebra2005; none on EdNet. QD adds item information and is separate from component removals. '
      'No-layer-normalization removal was never completed. Historical no_regularization aliases no_dropout and is not an independent experiment. '
      'No-attention keeps trainable inactive attention parameters; parameter count is not evidence of attention usage.')


if __name__=='__main__':main()
