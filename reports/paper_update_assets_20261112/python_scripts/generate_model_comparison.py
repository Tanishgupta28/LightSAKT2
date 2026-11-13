"""Genuine five-seed bars, matched-seed gains, and real student predictions."""
import os
os.environ['PAPER_GENERATOR']='generate_model_comparison.py'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import ASSETS, DATASETS, MODELS, COLORS, SHORT, DISPLAY, FOLDERS, csv, save_figure


def metric_plot(data,metric,label):
    fig,axes=plt.subplots(1,3,figsize=(15,4.0),sharey=False)
    for ax,dataset in zip(axes,DATASETS):
        group=data[data.dataset==dataset].set_index('model');present=[]
        for i,model in enumerate(MODELS):
            r=group.loc[model]
            if r.seed_count==5:
                v,s=r[metric+'_mean'],r[metric+'_std'];assert np.isfinite(v) and np.isfinite(s)
                ax.bar(i,v,yerr=s,width=.7,color=COLORS[model],capsize=2,error_kw={'elinewidth':.8});present.append(v)
            else:
                ax.text(i,.04,f'PENDING ({int(r.seed_count)}/5)',rotation=90,ha='center',va='bottom',fontsize=6,
                        transform=ax.get_xaxis_transform(),color='#555555')
        ax.set_xticks(range(len(MODELS)),[SHORT.get(m,m) for m in MODELS],rotation=65,ha='right')
        ax.set_xlim(-.6,8.6)
        ax.set(title=DISPLAY.get(dataset,dataset),ylabel=label);ax.grid(axis='y')
        if metric in ['test_auc','test_accuracy','test_f1']:ax.set_ylim(max(0,min(present)-.06),min(1,max(present)+.04))
        else:ax.set_ylim(0,max(present)*1.12)
    fig.suptitle(f'{label}: mean ± sample SD, five matched seeds; PENDING = no five-seed aggregate',fontsize=10)
    fig.tight_layout()
    save_figure(fig,f'graphs/model_comparisons/{metric}',
        ['source_data/verified_metric_references/master_metrics.csv'],
        f'{label}, fixed student split, five initializations; error bars are sample SD, not confidence intervals. '
        'No bar or numeric value for incomplete groups. AUC/accuracy/F1 panels use explicitly shown truncated axes. '
        'DKT+/DKVMN/GKT implementations/configuration-search differences are documented; difficulty adds information.')


def main():
    source='source_data/verified_metric_references/master_metrics.csv'
    data=pd.read_csv(ASSETS/source)
    for metric,label in [('test_auc','Held-out test AUC'),('test_accuracy','Test accuracy'),('test_f1','Test F1'),
                         ('test_mse','Test MSE / Brier score'),('test_bce','Test BCE (natural log)')]:
        metric_plot(data,metric,label)
    source_seed='source_data/verified_metric_references/individual_seed_metrics.csv'
    seeds=pd.read_csv(ASSETS/source_seed)
    fig,axes=plt.subplots(1,3,figsize=(15,4.0))
    markers=['o','s','^','D','v']
    for ax,d in zip(axes,DATASETS):
        for i,model in enumerate(MODELS):
            group=seeds[(seeds.dataset==d)&(seeds.model==model)].sort_values('seed')
            if len(group)==5:
                for j,(_,r) in enumerate(group.iterrows()):
                    ax.scatter(i+(j-2)*.10,r.test_auc,marker=markers[j],s=24,color=COLORS[model],edgecolor='white',linewidth=.25)
            else:ax.text(i,.03,'PENDING',rotation=90,transform=ax.get_xaxis_transform(),ha='center',fontsize=6)
        ax.set_xticks(range(9),[SHORT.get(m,m) for m in MODELS],rotation=65,ha='right')
        ax.set_xlim(-.6,8.6)
        ax.set(title=DISPLAY.get(d,d),ylabel='Held-out test AUC');ax.grid(axis='y')
    from matplotlib.lines import Line2D
    fig.legend([Line2D([],[],marker=m,color='black',linestyle='') for m in markers],
               ['seed42','seed123','seed2026','seed3407','seed7777'],loc='upper center',ncol=5)
    fig.tight_layout(rect=[0,0,1,.92])
    save_figure(fig,'graphs/model_comparisons/individual_seed_test_auc',[source_seed],
       'Five actual checkpoint-selected test AUCs per complete group. Horizontal offsets identify seeds, not independent cohorts.')
    # Four explicitly requested input-matched/control slots; unavailable controls remain visible.
    four=['SAKT','SAKT + Question Difficulty','LightSAKT','LightSAKT + Question Difficulty']
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    gains=[]
    for ax,d in zip(axes,DATASETS):
        group=data[data.dataset==d].set_index('model')
        for i,m in enumerate(four):
            row=group.loc[m]
            if row.seed_count==5:ax.bar(i,row.test_auc_mean,yerr=row.test_auc_std,color=COLORS[m],capsize=3)
            else:ax.text(i,.08,'PENDING',rotation=90,ha='center',transform=ax.get_xaxis_transform())
        for baseline in ['SAKT','LightSAKT']:
            a=seeds[(seeds.dataset==d)&(seeds.model=='LightSAKT + Question Difficulty')].set_index('seed')
            b=seeds[(seeds.dataset==d)&(seeds.model==baseline)].set_index('seed')
            for s in a.index:
                gains.append({'dataset':d,'augmented':'LightSAKT + Question Difficulty','baseline':baseline,'seed':int(s),
                    'test_auc_difference':float(a.loc[s,'test_auc']-b.loc[s,'test_auc']),
                    'validation_auc_difference':float(a.loc[s,'validation_auc']-b.loc[s,'validation_auc'])})
        ax.set_xticks(range(4),[SHORT.get(m,m) for m in four],rotation=35,ha='right')
        ax.set(title=DISPLAY.get(d,d),ylabel='Held-out test AUC',ylim=(.5,.9));ax.grid(axis='y')
    fig.tight_layout()
    csv(ASSETS/'tables/full_model_comparison/matched_difficulty_gains.csv',gains)
    save_figure(fig,'graphs/model_comparisons/question_difficulty_comparison',[source,source_seed],
       'Same frozen offline difficulty protocol. SAKT+QD is untrained on all datasets. Positive LightSAKT+QD gains cannot isolate attention superiority.')
    # An actual held-out student, selected by identity/order and length, never by model accuracy.
    student_source='source_data/verified_metric_references/anonymized_prediction_examples.csv'
    examples=pd.read_csv(ASSETS/student_source)
    for d in DATASETS:
        fig,ax=plt.subplots(figsize=(8.0,2.8));group=examples[examples.dataset==d]
        for model,g in group.groupby('model',sort=False):
            ax.plot(g.target_interaction_index,g.probability,label=SHORT.get(model,model),color=COLORS[model],marker='.',markersize=3)
        observed=group.drop_duplicates('target_interaction_index')
        ax.scatter(observed.target_interaction_index,observed.observed_correct,marker='|',s=55,color='black',label='Observed correctness')
        ax.set(title=f'{DISPLAY.get(d,d)} — anonymized real test student, seed42',xlabel='Target interaction index',
               ylabel='Predicted P(correct)',ylim=(-.04,1.04));ax.grid(True);ax.legend(loc='best',fontsize=7)
        fig.tight_layout()
        save_figure(fig,f'graphs/student_knowledge/{FOLDERS[d]}_prediction_sequence',[student_source],
          'Descriptive next-response probabilities from saved test predictions, NOT validated knowledge/mastery states or a fixed-skill trajectory. '
          'First lexicographically ordered test student with ≥20 targets; first ≤50 targets. Items/skills can change; prior-history windows can reset. '
          'No selection based on outcomes or model performance. Raw student IDs withheld. Cannot replace the original latent-state Figure8 or mastery claim.')


if __name__=='__main__':main()
