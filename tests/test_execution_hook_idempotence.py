import json
from pathlib import Path
from scripts import execution_hooks as hooks
from src.training.engine import write_json


def test_completed_gkt_probe_never_runs_again(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);write_json('experiments/gkt_thread_probe_request.json',{'enabled':True})
    calls=[]
    class Child:
        def wait(self):return 0
    def spawn(command,**kwargs):calls.append(command);return Child()
    monkeypatch.setattr(hooks.subprocess,'Popen',spawn)
    hooks.additional_gkt_probe();hooks.additional_gkt_probe()
    assert len(calls)==1 and calls[0][-1]=='scripts.probe_gkt_threads'
    assert json.load(open('reports/profiling/gkt_thread_probe_status.json'))['status']=='completed'


def test_ablation_verification_never_dispatches_a_gkt_probe(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);write_json('experiments/ablation_gpu_verification_request.json',{'enabled':True})
    write_json('reports/ablation_gpu_verification_status.json',{'status':'completed'})
    calls=[]
    class Result:returncode=0
    def command(argv,**kwargs):calls.append(argv);return Result()
    monkeypatch.setattr(hooks.subprocess,'run',command)
    def forbidden(*args,**kwargs):raise AssertionError('Completed inference or unrelated GKT probe repeated')
    monkeypatch.setattr(hooks.subprocess,'Popen',forbidden)
    hooks.verify_ablations_between_runs()
    assert [c[-1] for c in calls]==['scripts.diagnose_ablation_ranking','scripts.verify_light_grid_inference']
    write_json('reports/ablation_cpu_gpu_ranking_diagnostic.json',{'verified':True})
    write_json('reports/light_grid_inference_verification.json',{'verified':True})
    hooks.verify_ablations_between_runs();assert len(calls)==2
