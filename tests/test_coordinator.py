import json
from pathlib import Path
from scripts.coordinator import refreshed_jobs,rank,eligible


def test_complete_coordinator_skips_completed_and_keeps_scope(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path('experiments/runtime_jobs').mkdir(parents=True);Path('results/raw/done').mkdir(parents=True)
    jobs=[{'run_id':'done','status':'pending','phase':6,'model':'LightSAKT','config':'c.json'},
          {'run_id':'pending','status':'pending','phase':6,'model':'GKT','config':'c.json'}]
    Path('experiments/manifest.json').write_text(json.dumps({'jobs':jobs}))
    Path('results/raw/done/status.json').write_text(json.dumps({'status':'completed'}))
    Path('c.json').write_text(json.dumps({'model':'LightSAKT'}))
    before=Path('experiments/manifest.json').read_bytes()
    observed=refreshed_jobs()
    assert len(observed)==2 and observed[0]['status']=='completed' and observed[1]['status']=='pending'
    assert Path('experiments/manifest.json').read_bytes()==before


def test_priority_preserves_config_and_does_not_use_scores(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    jobs=[]
    for model,phase in [('GKT',6),('LightSAKT',6),('LightSAKT',7),('LightSAKT',8),('DKT',6)]:
        path=Path(f'{model}{phase}.json');path.write_text(json.dumps({'model':model,'learning_rate':.001}))
        jobs.append({'model':model,'phase':phase,'config':str(path)})
    ordered=sorted(jobs,key=rank)
    assert [(j['model'],j['phase']) for j in ordered]==[('LightSAKT',6),('LightSAKT',7),('LightSAKT',8),('DKT',6),('GKT',6)]


def test_ablations_wait_for_all_matched_final_seeds():
    job={'phase':8,'dataset':'A','model':'LightSAKT','config':'unused'}
    deps=[{'phase':7,'dataset':'A','model':'LightSAKT','status':'completed'} for _ in range(5)]
    deps[0]['status']='pending'
    assert not eligible(job,deps)
