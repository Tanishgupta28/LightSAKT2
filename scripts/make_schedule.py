import hashlib
import itertools
import json
from pathlib import Path
from src.training.engine import write_json

DATASETS=['ASSIST2009','ASSIST2015','Algebra2005']
MODELS=['DKT','DKT+','DKVMN','GKT','SAKT','LightSAKT','GKT_MHA']
SEEDS=[42,123,2026,3407,7777]


def model_defaults(model,length,dataset):
    if model in ['DKT','DKT+']:
        c={'emb_size':100,'hidden_size':100}
        if model=='DKT+':c.update(lambda_r=.01,lambda_w1=.003,lambda_w2=3.)
        return c
    if model=='DKVMN':return {'dim_s':50,'size_m':20}
    if model.startswith('GKT'):return {'hidden_size':30,'num_attn_heads':2,'method':'MHA' if model=='GKT_MHA' else 'PAM'}
    if model=='SAKT':return {'n':length,'d':100,'num_attn_heads':5,'dropout':.2}
    return {'max_seq_len':length,'embed_dim':64 if dataset=='Algebra2005' else 128,'num_heads':2,'dropout':.1}


def config(dataset,model,group,model_config,lr=.001,seed=42,evaluate_test=False):
    length=200 if dataset=='Algebra2005' else 100
    c={'dataset':dataset,'model':model,'group':group,'seed':seed,'seq_len':length,'batch_size':64,
       'max_epochs':50,'patience':5,'learning_rate':lr,'betas':[.9,.999],'weight_decay':0.,'epsilon':1e-8,
       'model_config':model_config,'evaluate_test':evaluate_test}
    suffix=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:12]
    c['run_id']=f'{dataset}-{model.replace("+","plus")}-{group}-{seed}-{suffix}'
    return c


def write_config(c):
    path=Path('configs/runs')/(c['run_id']+'.json');write_json(path,c);return str(path)


def main():
    jobs=[]
    for dataset,model in itertools.product(DATASETS,MODELS):
        length=200 if dataset=='Algebra2005' else 100
        c=config(dataset,model,'primary',model_defaults(model,length,dataset),lr=.005 if model=='LightSAKT' else .001,evaluate_test=True)
        jobs.append({'config':write_config(c),'run_id':c['run_id'],'phase':5,'status':'pending'})
    for dataset,model in itertools.product(DATASETS,MODELS):
        length=200 if dataset=='Algebra2005' else 100
        candidates=[]
        if model in ['LightSAKT','SAKT']:
            for lr,d,h,drop in itertools.product([.0005,.001,.005],[32,64,128],[2,4],[.1,.2]):
                mc=({'max_seq_len':length,'embed_dim':d,'num_heads':h,'dropout':drop} if model=='LightSAKT' else
                    {'n':length,'d':d,'num_attn_heads':h,'dropout':drop})
                candidates.append((lr,mc))
        elif model in ['DKT','DKT+']:
            for lr,emb,hidden in itertools.product([.0005,.001,.005],[32,64,128],[32,64,100,128]):
                mc=model_defaults(model,length,dataset);mc.update(emb_size=emb,hidden_size=hidden);candidates.append((lr,mc))
        elif model=='DKVMN':
            for lr,d,memory in itertools.product([.0005,.001,.005],[32,50,64],[10,20,50,100]):
                candidates.append((lr,{'dim_s':d,'size_m':memory}))
        else:
            for lr,hidden,heads in itertools.product([.0005,.001,.005],[16,30,64],[1,2,4,8]):
                # PAM has no active heads; tune four hidden sizes per embedding/learning-rate instead.
                if model=='GKT':
                    mc={'hidden_size':hidden,'num_attn_heads':2,'method':'PAM'}
                    # Additional coordinate is neighbor width via overall hidden size, with no duplicates.
                    mc['hidden_size']={16:[16,24,32,40],30:[48,56,64,72],64:[80,88,96,104]}[hidden][[1,2,4,8].index(heads)]
                else:mc={'hidden_size':32 if hidden==30 else hidden,'num_attn_heads':heads,'method':'MHA'}
                candidates.append((lr,mc))
        assert len(candidates)==36 and len({json.dumps(c,sort_keys=True) for c in candidates})==36
        for lr,mc in candidates:
            group='grid' if model=='LightSAKT' else 'baseline_tuning'
            c=config(dataset,model,group,mc,lr)
            jobs.append({'config':write_config(c),'run_id':c['run_id'],'phase':6,'status':'pending'})
    for dataset,model,seed in itertools.product(DATASETS,MODELS,SEEDS):
        jobs.append({'run_id':f'{dataset}-{model}-final-{seed}','phase':7,'dataset':dataset,'model':model,'seed':seed,'status':'pending','dependency':'all 36 tuning runs for this dataset/model'})
    for dataset,variant,seed in itertools.product(DATASETS,['no_dropout','no_position','single_head','no_attention','no_regularization'],SEEDS):
        jobs.append({'run_id':f'{dataset}-LightSAKT-{variant}-{seed}','phase':8,'dataset':dataset,'model':'LightSAKT','seed':seed,
                     'variant':variant,'status':'blocked' if variant=='no_regularization' else 'pending',
                     'dependency':'selected LightSAKT config','failure_reason':'No distinct historical regularizer defined' if variant=='no_regularization' else None})
    write_json('experiments/manifest.json',{'protocol':'configs/protocol.json','jobs':jobs})
    print('Scheduled',len(jobs),'slots; exact LightSAKT grid',sum(j.get('phase')==6 and 'LightSAKT-grid' in j['run_id'] for j in jobs))

if __name__=='__main__':main()
