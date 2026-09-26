import argparse,csv,json,subprocess,sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import yaml,numpy as np
def main():
 p=argparse.ArgumentParser(); p.add_argument("--head",choices=["baseline","transformer"],required=True); p.add_argument("--config",default="configs/experiment.yaml"); a=p.parse_args(); c=yaml.safe_load(Path(a.config).read_text()); root=Path(c["output_root"])/c["experiment_name"]/a.head; tc=c["train_config"]; rows=[]; cms=[]
 for fold in c.get("folds",[0,1,2,3,4]):
  d=root/f"fold_{fold}"; m=d/"metrics.json"
  if not (c.get("skip_completed",True) and m.exists()): subprocess.run([sys.executable,"scripts/train.py","--fold",str(fold),"--head",a.head,"--config",tc],check=True)
  r=json.loads(m.read_text()); rows.append({"fold":fold,**r["test"]}); cms.append(np.array(r["confusion_matrix"]))
 root.mkdir(parents=True,exist_ok=True); fields=["fold","Accuracy","MCC","Macro-AUPRC","Macro-AUROC","Macro-F1","Macro-Precision","Macro-Recall"]
 with (root/"cv_metrics.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
 summary={k:{"mean":float(np.nanmean([r[k] for r in rows])),"std":float(np.nanstd([r[k] for r in rows],ddof=1)),"variance":float(np.nanvar([r[k] for r in rows],ddof=1))} for k in fields[1:]}; summary["folds"]=rows; summary["confusion_matrix"]=np.sum(cms,axis=0).tolist(); (root/"cv_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=True))
 try:
  import matplotlib.pyplot as plt
  fig,ax=plt.subplots(figsize=(4,4)); ax.imshow(np.sum(cms,axis=0),cmap="Blues"); ax.set_xlabel("Predicted"); ax.set_ylabel("True"); fig.tight_layout(); fig.savefig(root/"cv_confusion_matrix.png",dpi=160); plt.close(fig)
 except Exception: pass
 print("5CV summary"); [print(f"{k}: {v['mean']:.4f} ± {v['std']:.4f} (variance {v['variance']:.6f})") for k,v in summary.items() if isinstance(v,dict) and 'mean' in v]
if __name__=="__main__": main()
