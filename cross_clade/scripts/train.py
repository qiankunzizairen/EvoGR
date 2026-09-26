import argparse, yaml, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from genome_classifier.training.trainer import train_fold
def main():
    p=argparse.ArgumentParser(); p.add_argument("--fold",type=int,required=True); p.add_argument("--head",choices=["baseline","transformer"],required=True); p.add_argument("--config",default="configs/train.yaml"); p.add_argument("--resume"); a=p.parse_args(); cfg=yaml.safe_load(Path(a.config).read_text()); root=Path(cfg.get("output_root","./outputs")); name=cfg.get("experiment_name","evo2_species_classifier"); train_fold(a.fold,a.head,cfg,root/name/a.head/f"fold_{a.fold}",a.resume)
if __name__=="__main__": main()
