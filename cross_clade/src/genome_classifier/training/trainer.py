from pathlib import Path
import csv, json, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import EmbeddingDataset
from .heads import build_head
from .metrics import compute_metrics

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def train_fold(fold, head_name, cfg, output_dir, resume=None):
    seed_everything(cfg["training"].get("seed",42)+int(fold)); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    e=cfg["embedding"]; t=cfg["training"]; m=cfg.get("model",{}); device=torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    train_ds=EmbeddingDataset(e["selection_file"],e["index_file"],e["cache_dir"],fold,"train"); val_ds=EmbeddingDataset(e["selection_file"],e["index_file"],e["cache_dir"],fold,"val"); test_ds=EmbeddingDataset(e["selection_file"],e["index_file"],e["cache_dir"],fold,"test")
    bs=int(t.get("batch_size",128)); nw=int(t.get("num_workers",0)); train_dl=DataLoader(train_ds,bs,shuffle=True,num_workers=nw); val_dl=DataLoader(val_ds,bs,shuffle=False,num_workers=nw); test_dl=DataLoader(test_ds,bs,shuffle=False,num_workers=nw)
    model=build_head(head_name,int(m.get("num_classes",4)),float(m.get("dropout",.1))).to(device); print(f"Prediction head trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    opt=torch.optim.AdamW(model.parameters(),lr=float(cfg.get("optimizer",{}).get("lr",3e-4)),weight_decay=float(cfg.get("optimizer",{}).get("weight_decay",1e-2))); sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",patience=2,factor=.5); loss_fn=torch.nn.CrossEntropyLoss(); amp=torch.cuda.is_available() and bool(t.get("mixed_precision",True)); scaler=torch.cuda.amp.GradScaler(enabled=amp)
    start=0; best=-float("inf"); best_epoch=0
    if resume and Path(resume).exists():
        ck=torch.load(resume,map_location=device,weights_only=False); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); start=ck.get("epoch",-1)+1; best=ck.get("best_val_macro_f1",best); best_epoch=ck.get("best_epoch",0)
    logs=[]; patience=0
    for epoch in range(start,int(t.get("max_epochs",50))):
        model.train(); total=0
        for x,y in train_dl:
            x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp): l=loss_fn(model(x),y)
            scaler.scale(l).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),float(t.get("gradient_clip_norm",1.0))); scaler.step(opt); scaler.update(); total += l.item()*len(y)
        val= _evaluate(model,val_dl,device,int(m.get("num_classes",4))); vf=val[0]["Macro-F1"]; sched.step(vf); logs.append({"epoch":epoch,"train_loss":total/max(1,len(train_ds)),"val_macro_f1":vf})
        ck={"model":model.state_dict(),"optimizer":opt.state_dict(),"epoch":epoch,"best_val_macro_f1":best,"best_epoch":best_epoch}
        torch.save(ck,out/"last.pt")
        if vf>best: best=vf; best_epoch=epoch; ck.update(best_val_macro_f1=best,best_epoch=best_epoch); torch.save(ck,out/"best.pt"); patience=0
        else: patience+=1
        if patience>=int(t.get("early_stopping_patience",7)): break
    with (out/"training_log.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=["epoch","train_loss","val_macro_f1"]); w.writeheader(); w.writerows(logs)
    best_ck=torch.load(out/"best.pt",map_location=device,weights_only=False); model.load_state_dict(best_ck["model"]); metrics,cm=_evaluate(model,test_dl,device,int(m.get("num_classes",4))); np.savetxt(out/"confusion_matrix.csv",cm,fmt="%d",delimiter=",")
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4,4)); ax.imshow(cm,cmap="Blues"); ax.set_xlabel("Predicted"); ax.set_ylabel("True"); fig.tight_layout(); fig.savefig(out/"confusion_matrix.png",dpi=160); plt.close(fig)
    except Exception: pass
    result={"fold":int(fold),"head":head_name,"best_epoch":int(best_ck.get("best_epoch",best_epoch)),"best_val_macro_f1":float(best_ck.get("best_val_macro_f1",best)),"test":metrics,"confusion_matrix":cm.tolist()}; (out/"metrics.json").write_text(json.dumps(result,indent=2,allow_nan=True)); return result

def _evaluate(model, loader, device, classes):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for x,y in loader: ys.extend(y.numpy()); ps.append(torch.softmax(model(x.to(device)),1).cpu().numpy())
    p=np.concatenate(ps) if ps else np.zeros((0,classes)); return compute_metrics(np.asarray(ys),p,classes)
