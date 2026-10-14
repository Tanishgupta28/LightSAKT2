import json
import itertools
import io
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from scripts.summarize import summarize


def save_figure(name):
    Path('results/figures').mkdir(parents=True,exist_ok=True)
    # Stable vector IDs and omitted creation dates make unchanged observations
    # produce identical artifacts instead of needless NFS writes/Git churn.
    with matplotlib.rc_context({'svg.hashsalt':'LightSAKT2/'+name}):
        for ext in ['png','svg','pdf']:
            metadata={'Date':None} if ext=='svg' else {'CreationDate':None,'ModDate':None} if ext=='pdf' else None
            output=io.BytesIO();plt.savefig(output,format=ext,dpi=300,bbox_inches='tight',metadata=metadata)
            path=Path('results/figures')/(name+'.'+ext);payload=output.getvalue()
            if not path.exists() or path.read_bytes()!=payload:
                temporary=path.with_suffix(f'.{os.getpid()}.tmp');temporary.write_bytes(payload);temporary.replace(path)
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
        figure_name=f'Figure_{fig}' if data.model.nunique()==7 else f'Figure_{fig}_partial'
        save_figure(figure_name)
        pd.DataFrame(curve_rows).to_csv(f'results/figures/{figure_name}.csv',index=False)
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
    for dataset,data in frame[(frame.group=='grid') & (frame.model=='LightSAKT')].groupby('dataset'):
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
    if ablation_rows:
        ablations=pd.DataFrame(ablation_rows)
        for (dataset,metric),data in ablations.groupby(['dataset','metric']):
            summary=data.groupby('variant').agg(mean=('value','mean'),std=('value','std'),
                                               relative_change_mean=('relative_change','mean'),count=('value','count')).reset_index()
            name=f'{dataset}_ablation_{metric}'
            summary.to_csv(f'results/figures/{name}.csv',index=False)
            plt.figure(figsize=(8,4));plt.bar(summary.variant,summary['mean'],yerr=summary['std'].fillna(0))
            base=float(data.default.mean());plt.axhline(base,color='black',linestyle='--',label='Matched default mean')
            plt.ylabel('Test '+metric.upper());plt.xticks(rotation=25,ha='right');plt.legend(fontsize=8)
            plt.title(dataset+' — no regularization equals no dropout; counts in CSV');save_figure(name)
    for dataset,data in final.groupby('dataset'):
        # All five observed seeds are retained; fixed horizontal offsets are display-only.
        values=data[['model','seed','test_auc']].sort_values(['model','seed'])
        name=dataset+'_seed_variability';values.to_csv(f'results/figures/{name}.csv',index=False)
        models=sorted(values.model.unique());plt.figure(figsize=(8,4))
        for i,model in enumerate(models):
            group=values[values.model==model];positions=i+np.linspace(-.12,.12,len(group))
            plt.scatter(positions,group.test_auc,s=25)
        plt.xticks(range(len(models)),models,rotation=25);plt.ylabel('Individual-seed test AUC');plt.title(dataset+' — all completed matched seeds');save_figure(name)
        # AUC/parameter comparisons retain individual runs rather than hiding seed variability.
        name=dataset+'_parameter_efficiency'
        data[['run_id','model','seed','parameters','test_auc']].to_csv(f'results/figures/{name}.csv',index=False)
        plt.figure(figsize=(7,4))
        for model,group in data.groupby('model'):plt.scatter(group.parameters,group.test_auc,label=model,s=25)
        plt.xscale('log');plt.xlabel('Trainable parameters');plt.ylabel('Individual-seed test AUC');plt.legend(fontsize=7);save_figure(name)
        # Display one prespecified seed for ROC curves; model selection is already frozen.
        plt.figure(figsize=(6,5));roc_rows=[]
        for row in data[data.seed==42].itertuples():
            result=json.load(open(Path('results/raw')/row.run_id/'metrics.json'))
            with np.load(result['prediction_path']) as arrays:
                fpr,tpr,thresholds=roc_curve(arrays['target'],arrays['probability'])
            plt.plot(fpr,tpr,label=row.model)
            roc_rows.extend({'run_id':row.run_id,'model':row.model,'false_positive_rate':float(f),'true_positive_rate':float(t),'threshold':float(s)} for f,t,s in zip(fpr,tpr,thresholds))
        pd.DataFrame(roc_rows).to_csv(f'results/figures/{dataset}_ROC_seed42.csv',index=False)
        plt.plot([0,1],[0,1],'k--',alpha=.3);plt.xlabel('False positive rate');plt.ylabel('True positive rate');plt.title(dataset+' — selected models, prespecified seed 42');plt.legend(fontsize=7);save_figure(dataset+'_ROC_seed42')

if __name__=='__main__':main()
