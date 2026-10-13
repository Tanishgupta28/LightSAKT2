import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.datasets.sequences import Sequences
from src.datasets.preprocess import sha256
from src.models.factory import build_model, predict
from src.evaluation.metrics import masked_bce, metrics
from src.training.reproducibility import seed_everything, seed_worker, rng_state, restore_rng, environment


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def save_state(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp");torch.save(value,tmp);tmp.replace(path)


def training_loss(model, name, batch, p, all_p):
    q,r,qry,target,valid=batch
    loss=masked_bce(p,target,valid)
    if name=="DKT+":
        current=all_p.gather(-1,q.unsqueeze(-1)).squeeze(-1)
        loss=loss+model.lambda_r*masked_bce(current,r,valid)
        adjacent=valid[:,1:] & valid[:,:-1]
        if adjacent.any():
            difference=all_p[:,1:]-all_p[:,:-1]
            loss=loss+model.lambda_w1*difference.abs().sum(-1)[adjacent].mean()/model.num_q
            loss=loss+model.lambda_w2*difference.square().sum(-1)[adjacent].mean()/model.num_q
    return loss


@torch.no_grad()
def evaluate(model,name,loader,device):
    model.eval();targets=[];predictions=[]
    for batch in loader:
        batch=[x.to(device) for x in batch]
        p,_=predict(model,name,batch);valid=batch[-1]
        targets.append(batch[3][valid].cpu().numpy());predictions.append(p[valid].cpu().numpy())
    if not targets:
        raise ValueError("Empty heldout dataset")
    y,p=np.concatenate(targets),np.concatenate(predictions)
    return metrics(y,p),y,p


def update_early_stop(auc,best,bad,patience=5):
    if auc is None:
        raise ValueError("Validation AUC undefined: no checkpoint selection is scientifically defined")
    improved=auc>best
    return (auc if improved else best), (0 if improved else bad+1), improved, (not improved and bad+1>=patience)


def run(config, root=".", resume=True, stop_after_epoch=None):
    root=Path(root);run_id=config["run_id"]
    directory=root/"results/raw"/run_id;directory.mkdir(parents=True,exist_ok=True)
    cfg_path=directory/"config.json";status_path=directory/"status.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text())!=config:
        raise ValueError("Cannot mutate an existing run configuration")
    write_json(cfg_path,config)
    if status_path.exists() and json.loads(status_path.read_text())["status"]=="completed":
        return json.loads((directory/"metrics.json").read_text())
    seed_everything(config["seed"])
    device=torch.device(config.get("device","cuda" if torch.cuda.is_available() else "cpu"))
    dataset=config["dataset"];stats=json.loads((root/"reports"/f"{dataset}_statistics.json").read_text())
    path=root/"data/processed"/dataset/"interactions.csv"
    if sha256(path)!=stats["processed_sha256"] or sha256(root/"data/splits"/f"{dataset}.json")!=stats["split_sha256"]:
        raise ValueError("Data/split checksum changed")
    generator=torch.Generator().manual_seed(config["seed"])
    def loader(split):
        ds=Sequences(path,split,config["seq_len"])
        return DataLoader(ds,batch_size=config["batch_size"],shuffle=split=="train",
                          num_workers=0,worker_init_fn=seed_worker,generator=generator if split=="train" else None)
    train,validation=loader("train"),loader("validation")
    # Test loader is only constructed after training and checkpoint selection below.
    model=build_model(config["model"],stats["num_q"],config["model_config"]).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=config["learning_rate"],
                               betas=tuple(config["betas"]),eps=config["epsilon"],weight_decay=config["weight_decay"])
    ckpt=root/"checkpoints"/run_id;ckpt.mkdir(parents=True,exist_ok=True)
    latest=ckpt/"latest.pt";best_path=ckpt/"best.pt"
    history=[];epoch=0;best=-float("inf");bad=0;best_epoch=None;elapsed=0.
    env=environment();write_json(directory/"environment.json",env)
    if resume and latest.exists():
        state=torch.load(latest,map_location=device,weights_only=False)
        if state["config"]!=config or state["processed_sha256"]!=stats["processed_sha256"]:
            raise ValueError("Resume metadata mismatch")
        model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"])
        epoch,best,bad,best_epoch=state["epoch"],state["best_auc"],state["bad_epochs"],state["best_epoch"]
        history,elapsed=state["history"],state["elapsed_seconds"]
        generator.set_state(state["loader_rng"].cpu());restore_rng(state["rng"])
    status={"run_id":run_id,"status":"running","last_completed_epoch":epoch,"config":str(cfg_path),
            "seed":config["seed"],"resume_command":f"python -m scripts.run --config {cfg_path}",
            "best_checkpoint":str(best_path),"failure_reason":None}
    write_json(status_path,status)
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
    clock=time.perf_counter()
    try:
        for epoch in range(epoch+1,config["max_epochs"]+1):
            if bad>=config["patience"]:break
            started=time.perf_counter();model.train();loss_sum=0.;count=0
            for batch in train:
                batch=[x.to(device) for x in batch];optimizer.zero_grad(set_to_none=True)
                p,all_p=predict(model,config["model"],batch)
                loss=training_loss(model,config["model"],batch,p,all_p)
                if not torch.isfinite(loss):raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                if any(x.grad is not None and not torch.isfinite(x.grad).all() for x in model.parameters()):
                    raise FloatingPointError("Nonfinite training gradient")
                optimizer.step();n=int(batch[-1].sum());loss_sum+=float(loss.detach())*n;count+=n
            validation_metrics,_,_=evaluate(model,config["model"],validation,device)
            best,bad,improved,stop=update_early_stop(validation_metrics["auc"],best,bad,config["patience"])
            if improved:
                best_epoch=epoch
                save_state(best_path,{"model":model.state_dict(),"config":config,"epoch":epoch,
                                      "validation":validation_metrics,"processed_sha256":stats["processed_sha256"],
                                      "split_sha256":stats["split_sha256"],"git_commit":env["git_commit"]})
            history.append({"epoch":epoch,"train_objective":loss_sum/count,"validation":validation_metrics,
                            "epoch_seconds":time.perf_counter()-started})
            state={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch,
                   "rng":rng_state(),"loader_rng":generator.get_state(),"config":config,"best_auc":best,
                   "best_epoch":best_epoch,"bad_epochs":bad,"history":history,"environment":env,
                   "elapsed_seconds":elapsed+time.perf_counter()-clock,"processed_sha256":stats["processed_sha256"],
                   "split_sha256":stats["split_sha256"]}
            save_state(latest,state);write_json(directory/"history.json",history)
            status["last_completed_epoch"]=epoch;write_json(status_path,status)
            print(f"{run_id} epoch={epoch} validation_auc={validation_metrics['auc']:.6f} seconds={history[-1]['epoch_seconds']:.1f}",flush=True)
            if stop_after_epoch is not None and epoch>=stop_after_epoch:
                status["status"]="interrupted";write_json(status_path,status);return None
            if stop:break
        selected=torch.load(best_path,map_location=device,weights_only=False)
        model.load_state_dict(selected["model"])
        val,_,_=evaluate(model,config["model"],validation,device)
        result={"run_id":run_id,"dataset":dataset,"model":config["model"],"seed":config["seed"],
                "group":config["group"],"validation":val,"best_epoch":best_epoch,"epochs_trained":len(history),
                "training_seconds":elapsed+time.perf_counter()-clock,"checkpoint":str(best_path),
                "checkpoint_sha256":sha256(best_path),"processed_sha256":stats["processed_sha256"],
                "split_sha256":stats["split_sha256"],"git_commit":selected["git_commit"],
                "parameters":sum(x.numel() for x in model.parameters() if x.requires_grad),
                "peak_allocated_bytes":torch.cuda.max_memory_allocated() if device.type=="cuda" else None,
                "peak_reserved_bytes":torch.cuda.max_memory_reserved() if device.type=="cuda" else None,
                "test":None}
        if config.get("evaluate_test",False):
            test_loader=loader("test")
            result["test"],y,p=evaluate(model,config["model"],test_loader,device)
            dest=root/"results/predictions"/f"{run_id}.npz";dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest,target=y,probability=p)
            write_json(directory/"prediction_identity.json",test_loader.dataset.identities)
            result["prediction_path"]=str(dest);result["prediction_sha256"]=sha256(dest)
        write_json(directory/"metrics.json",result)
        status["status"]="completed";write_json(status_path,status)
        return result
    except BaseException as error:
        status["status"]="interrupted" if isinstance(error,(KeyboardInterrupt,SystemExit)) else "failed"
        status["failure_reason"]=f"{type(error).__name__}: {error}"
        write_json(status_path,status);raise
