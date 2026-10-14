"""Bounded inter-run concurrency and correctness gate; no research training."""
import json
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from src.training.engine import write_json

ROOT=Path('reports/profiling')


def precision_probe():
    from scripts.profile_runtime import prepare
    from src.models.factory import predict
    from src.training.engine import training_loss
    from src.training.reproducibility import seed_everything
    rows=[]
    for dataset in ['ASSIST2009','ASSIST2015','Algebra2005']:
        config=json.load(open(ROOT/(dataset+'-LightSAKT-primary-config.json')))
        model,_,batch,_=prepare(config);reference={};probabilities={};losses={}
        for precision in ['fp32','bf16']:
            seed_everything(config['seed']);model.train();model.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
                p,all_p=predict(model,config['model'],batch)
            # Preserve probability-based BCE in FP32; BCE autocast is unsupported.
            loss=training_loss(model,config['model'],batch,p.float(),all_p)
            loss.backward();probabilities[precision]=p.detach().float();losses[precision]=float(loss.detach())
            reference[precision]={k:v.grad.detach().clone() for k,v in model.named_parameters() if v.grad is not None}
        prob_error=float((probabilities['fp32']-probabilities['bf16']).abs().max())
        grad_error=max(float((v-reference['bf16'][k]).abs().max()) for k,v in reference['fp32'].items())
        rows.append({'dataset':dataset,'model':'LightSAKT','comparison':'FP32 versus BF16 autocast on identical train batch and dropout RNG',
            'max_probability_absolute_difference':prob_error,'max_gradient_absolute_difference':grad_error,
            'loss_difference':abs(losses['fp32']-losses['bf16']),
            'strict_probability_tolerance':1e-6,'strict_gradient_tolerance':1e-6,
            'passes_strict_equivalence':prob_error<=1e-6 and grad_error<=1e-6,'enabled_for_research':False})
        del model,batch;torch.cuda.empty_cache()
    write_json(ROOT/'precision_probe.json',rows)
    return rows


def concurrency(case,name):
    reference=ROOT/(name+'-serial.json')
    subprocess.run(['python','-m','scripts.profile_runtime','--worker',str(case),'--output',str(reference)],check=True,timeout=180)
    flag=ROOT/(name+'-start.flag');flag.unlink(missing_ok=True)
    processes=[];outputs=[]
    try:
        for i in range(2):
            output=ROOT/(name+f'-parallel-{i}.json');Path(str(output)+'.ready').unlink(missing_ok=True);outputs.append(output)
            processes.append(subprocess.Popen(['python','-m','scripts.profile_runtime','--worker',str(case),'--output',str(output),'--start-flag',str(flag)]))
        deadline=time.monotonic()+150
        while not all(Path(str(out)+'.ready').exists() for out in outputs):
            if any(p.poll() is not None for p in processes):raise RuntimeError('Diagnostic worker exited before ready')
            if time.monotonic()>deadline:raise TimeoutError('Diagnostic initialization')
            time.sleep(.05)
        flag.write_text('start')
        for process in processes:
            if process.wait(timeout=180):raise RuntimeError('Diagnostic worker failed')
    finally:
        for process in processes:
            if process.poll() is None:process.terminate()
        for process in processes:
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
    serial=torch.load(str(reference)+'.pt',map_location='cpu',weights_only=False)
    peers=[json.load(open(out)) for out in outputs];exact=True
    for out in outputs:
        state=torch.load(str(out)+'.pt',map_location='cpu',weights_only=False)
        for group in ['parameters','gradients']:
            for key,value in serial[group].items():
                other=state[group][key]
                exact &= other is None if value is None else torch.equal(value,other)
    reference_record=json.load(open(reference));config=json.load(open(case))
    return {'case':name,'dataset':config['dataset'],'model':config['model'],'workers':2,
            'exact_parameters_and_gradients':bool(exact),
            'loop_throughput_speedup':2*reference_record['wall_seconds']/max(p['wall_seconds'] for p in peers),
            'serial_step_mean':float(np.mean(reference_record['step_seconds'])),
            'parallel_step_means':[float(np.mean(p['step_seconds'])) for p in peers],
            'peak_reserved_bytes_per_worker':max(p['peak_reserved_bytes'] for p in peers)}


def main():
    status=ROOT/'execution_validation.json'
    write_json(status,{'status':'running','scope':'Train-only bit-exact concurrency probes and full correctness gate'})
    precision=precision_probe()
    records=[]
    for dataset in ['ASSIST2009','ASSIST2015','Algebra2005']:
        name=dataset+'-GKT-primary';records.append(concurrency(ROOT/(name+'-config.json'),name))
    for model in ['GKT','GKT_MHA']:
        name='Algebra2005-'+model+'-wide';records.append(concurrency(ROOT/(name+'-config.json'),name))
    assert all(r['exact_parameters_and_gradients'] for r in records),'Concurrent execution changed updates'
    capacity=torch.cuda.get_device_properties(0).total_memory
    assert 2*(max(r['peak_reserved_bytes_per_worker'] for r in records)+1024**3)<.8*capacity,'Insufficient safe concurrency memory'
    log=Path('reports/test_runs')/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-execution-gate.log')
    with open(log,'w') as stream:rc=subprocess.run(['python','-m','pytest','-q'],stdout=stream,stderr=subprocess.STDOUT,timeout=300).returncode
    write_json('reports/latest_test_run.json',{'log':str(log),'tests_passed':rc==0,'exit_code':rc,'scope':'CPU and CUDA; inter-run exclusive correctness gate'})
    assert rc==0,f'Correctness gate failed: {log}'
    write_json(status,{'status':'completed','updated_utc':datetime.now(timezone.utc).isoformat(),
        'concurrency':records,'maximum_validated_workers':2,'tests_log':str(log),'tests_passed':True,
        'mixed_precision_probe':precision,'mixed_precision_enabled':False,
        'source_and_architecture_unchanged':True,'precision':'FP32; deterministic; AMP/TF32/compile disabled'})
    print(json.dumps(json.load(open(status)),indent=2),flush=True)


if __name__=='__main__':main()
