import json
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.summarize import summarize


def save_figure(name):
    Path('results/figures').mkdir(parents=True,exist_ok=True)
    for ext in ['png','svg','pdf']:plt.savefig(f'results/figures/{name}.{ext}',dpi=300,bbox_inches='tight')
    plt.close()


def holm(pvalues):
    order=np.argsort(pvalues);adjusted=np.empty(len(pvalues));previous=0.
    for i,index in enumerate(order):
        previous=max(previous,min(1.,pvalues[index]*(len(pvalues)-i)));adjusted[index]=previous
    return adjusted


def exact_signed_rank(differences):
    """Enumerate sign assignments; handles ties and zero differences explicitly."""
    nonzero=np.asarray(differences)[np.asarray(differences)!=0]
    if not len(nonzero):return 0.,1.,0.
    ranks=rankdata(np.abs(nonzero));observed=float((ranks*np.sign(nonzero)).sum())
    permutations=[abs(float(np.dot(signs,ranks))) for signs in itertools.product([-1,1],repeat=len(nonzero))]
    p=sum(value>=abs(observed)-1e-12 for value in permutations)/len(permutations)
    positive=ranks[nonzero>0].sum();negative=ranks[nonzero<0].sum()
    return float(min(positive,negative)),float(p),float(observed/ranks.sum())


def significance(frame):
    rows=[];final=frame[frame.group=='final']
    for dataset,data in final.groupby('dataset'):
        light=data[data.model=='LightSAKT'].set_index('seed')
        for model,group in data[data.model!='LightSAKT'].groupby('model'):
            baseline=group.set_index('seed');seeds=sorted(set(light.index)&set(baseline.index))
            differences=np.array([light.loc[s,'test_auc']-baseline.loc[s,'test_auc'] for s in seeds],dtype=float)
            differences=differences[np.isfinite(differences)];n=len(differences)
            if n<2:continue
            statistic,p_value,effect=exact_signed_rank(differences)
            half=t.ppf(.975,n-1)*differences.std(ddof=1)/np.sqrt(n)
            rows.append({'dataset':dataset,'baseline':model,'n_pairs':n,'seeds':json.dumps(seeds),
                         'pairing':'same model seed; one fixed student split; conditional initialization variability',
                         'test':'exact two-sided signed-rank sign enumeration; zeros omitted, midranks for ties','statistic':statistic,
                         'p_value':p_value,'rank_biserial_effect':effect,
                         'mean_auc_difference':float(differences.mean()),'difference_ci95_lower':float(differences.mean()-half),
                         'difference_ci95_upper':float(differences.mean()+half),
                         'limitation':'Five seeds cannot support exact two-sided p<.05; seed repeats do not sample new students.'})
    if rows:
        # Prespecified family: 3 datasets x 6 baseline variants; incomplete comparisons have p=1.
        adjusted=holm([r['p_value'] for r in rows]+[1.]*(18-len(rows)))[:len(rows)]
        for row,p in zip(rows,adjusted):row['holm_p']=float(p);row['interpretation']='reject conditional null' if p<.05 else 'insufficient evidence to reject'
    pd.DataFrame(rows).to_csv('results/tables/significance.csv',index=False)


def main():
    summarize();path=Path('results/tables/all_verified_runs.csv')
    frame=pd.read_csv(path)
    if frame.empty:return
    significance(frame)
    for dataset,fig in [('Algebra2005',5),('ASSIST2009',6),('ASSIST2015',7)]:
        data=frame[(frame.dataset==dataset)&(frame.group=='primary')];curve_rows=[]
        if data.empty:continue
        plt.figure(figsize=(7,4))
        for row in data.itertuples():
            h=json.load(open(Path('results/raw')/row.run_id/'history.json'))
            label=row.model+(' (historical simplified)' if row.model.startswith('GKT') else '')
            plt.plot([x['epoch'] for x in h],[x['validation']['auc'] for x in h],label=label)
            for x in h:curve_rows.append({'run_id':row.run_id,'model':row.model,'epoch':x['epoch'],'validation_auc':x['validation']['auc'],'train_objective':x['train_objective'],'validation_accuracy':x['validation']['accuracy']})
        plt.xlabel('Epoch');plt.ylabel('Validation AUC');plt.title(dataset+' — completed primary runs, seed 42');plt.legend(fontsize=7);plt.grid(alpha=.2)
        save_figure(f'Figure_{fig}_partial')
        pd.DataFrame(curve_rows).to_csv(f'results/figures/Figure_{fig}_partial.csv',index=False)
        for metric in ['train_objective','validation_accuracy']:
            plt.figure(figsize=(7,4));curves=pd.DataFrame(curve_rows)
            for model,data in curves.groupby('model'):plt.plot(data.epoch,data[metric],label=model)
            plt.xlabel('Epoch');plt.ylabel(metric.replace('_',' '));plt.title(dataset+' — primary seed 42');plt.legend(fontsize=7);save_figure(dataset+'_'+metric)
    final=frame[frame.group=='final']
    if not final.empty:
        for metric in ['auc','accuracy','f1','mse','bce']:
            summary=final.groupby(['dataset','model'])[f'test_{metric}'].agg(['mean','std','count']).reset_index()
            summary.to_csv(f'results/figures/{metric}_seed_summary.csv',index=False)
            for dataset,data in summary.groupby('dataset'):
                plt.figure(figsize=(8,4));plt.bar(data.model,data['mean'],yerr=data['std'].fillna(0));plt.ylabel('Test '+metric.upper());plt.xticks(rotation=30);plt.title(dataset+' — completed seeds; counts in CSV');save_figure(dataset+'_'+metric+'_repeated')
    for dataset,data in frame[frame.group=='grid'].groupby('dataset'):
        pivot=data.pivot_table(index='embed_dim',columns='learning_rate',values='validation_auc',aggfunc='max')
        pivot.to_csv(f'results/figures/{dataset}_grid_heatmap.csv')
        plt.figure(figsize=(5,4));plt.imshow(pivot.values,aspect='auto',cmap='viridis');plt.xticks(range(len(pivot.columns)),pivot.columns);plt.yticks(range(len(pivot.index)),pivot.index.astype(int));plt.xlabel('Learning rate');plt.ylabel('Embedding width');plt.title(dataset+' — best heads/dropout, completed grids');plt.colorbar(label='Validation AUC');save_figure(dataset+'_grid_heatmap')
    # Ablations remain paired to the same seed and selected default, never compared to historical scores.
    ablation_rows=[]
    for row in frame[frame.group=='ablation'].itertuples():
        default=final[(final.dataset==row.dataset)&(final.model=='LightSAKT')&(final.seed==row.seed)]
        if default.empty:continue
        c=json.load(open(Path('results/raw')/row.run_id/'config.json'))
        for metric in ['auc','accuracy','f1','mse','bce']:
            value=getattr(row,f'test_{metric}');base=float(default.iloc[0][f'test_{metric}'])
            ablation_rows.append({'dataset':row.dataset,'variant':c.get('scientific_variant',c['model_config']['variant']),'equivalent_to':c.get('equivalent_to'),'seed':row.seed,'metric':metric,'value':value,'default':base,'relative_change':(value-base)/base if base else None,'run_id':row.run_id})
    pd.DataFrame(ablation_rows).to_csv('results/tables/paired_ablations.csv',index=False)

if __name__=='__main__':main()
