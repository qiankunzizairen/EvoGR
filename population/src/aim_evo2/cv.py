from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import numpy as np
from torch.utils.data import DataLoader
from .dataset import FixedFullDataset, VariableSNPDataset, VariableContextDataset, collate_fixed, collate_variable_snp, resolve_buckets
from .metrics import compute_metrics
from .model import AttentionSetClassifier
from .representations import load_delta_embeddings_by_context
from .config import all_context_lengths, fixed_context_length, variable_context_lengths
from .utils import utc_now_iso
from .manifests import embedding_manifest_contract, representation_manifest_contract, assert_manifest_current

def _device(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")
def _model(cfg, dim, n, device):
    m=cfg['model']; return AttentionSetClassifier(dim,n,num_classes=5,projection_dim=int(m['locus_projection_dim']),attention_hidden_dim=int(m['attention_hidden_dim']),classifier_hidden_dim=int(m['classifier_hidden_dim']),dropout=float(m['dropout'])).to(device)
def _predict(model, ds, cfg, collate, device):
    ys=[]; ps=[]; model.eval()
    with torch.no_grad():
      for b in DataLoader(ds,batch_size=int(cfg['training']['batch_size']),shuffle=False,collate_fn=collate):
        z,_=model(b['features'].to(device),b['locus_indices'].to(device),b['attention_mask'].to(device)); ys.extend(b['label'].tolist()); ps.append(torch.softmax(z,1).cpu().numpy())
    return np.asarray(ys),np.concatenate(ps)
def _fit(train_ds,val_fn,dim,n,cfg,collate,device):
    t=cfg['training']; model=_model(cfg,dim,n,device); opt=torch.optim.AdamW(model.parameters(),lr=float(t['learning_rate']),weight_decay=float(t['weight_decay'])); loss_fn=torch.nn.CrossEntropyLoss(); best=(-np.inf,np.inf,1); stale=0
    for epoch in range(1,int(t['max_epochs'])+1):
      if hasattr(train_ds,'set_epoch'): train_ds.set_epoch(epoch)
      model.train()
      for b in DataLoader(train_ds,batch_size=int(t['batch_size']),shuffle=True,collate_fn=collate):
        opt.zero_grad(); z,_=model(b['features'].to(device),b['locus_indices'].to(device),b['attention_mask'].to(device)); loss=loss_fn(z,b['label'].to(device)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),float(t['gradient_clip_norm'])); opt.step()
      score,vl=val_fn(model)
      if score>best[0] or (score==best[0] and vl<best[1]): best=(score,vl,epoch); stale=0
      else: stale+=1
      if stale>=int(t['early_stopping_patience']): break
    return model,best[2]
def run_pipeline(config,data,folds,pipeline='all'):
    root=Path(config['_root']); path=root/config['artifacts']['embedding_dir']/'allele_embeddings.pt'; device=_device(); names=['fixed_full','variable_snp','variable_context'] if pipeline=='all' else [pipeline]; lengths=all_context_lengths(config); assert_manifest_current(path.parent/'embedding_manifest.json',embedding_manifest_contract(config,data),'embedding'); ds=load_delta_embeddings_by_context(str(path),int(config['evo2']['active_pool_radius']),data.rsid_order,lengths); repdir=root/config['artifacts'].get('representation_dir','artifacts/representations'); embedding_manifest=json.loads((path.parent/'embedding_manifest.json').read_text(encoding='utf-8')); hidden_dim=int(embedding_manifest['hidden_dim']); assert_manifest_current(repdir/'representation_manifest.json',representation_manifest_contract(config,data,path,hidden_dim),'representation'); pre={l:torch.from_numpy(np.load(repdir/f'representations_L{l}.npy',mmap_mode='r')) for l in set(lengths)}; outputs={}
    for name in names:
      fm=[]
      for fi,fold in enumerate(folds):
        tr,va,te=fold['train'],fold['val'],fold['test']
        if name=='fixed_full':
          length=fixed_context_length(config); d=ds[length]; train,val,test=[FixedFullDataset(data.genotypes,data.rsid_order,d,x,precomputed=pre[length]) for x in (tr,va,te)]; vf=lambda m:(compute_metrics(*_predict(m,val,config,collate_fixed,device),data.label_mapping)[0]['Macro-F1'],0.0); model,e=_fit(train,vf,d.shape[1]*2,data.n_aim,config,collate_fixed,device); y,p=_predict(model,test,config,collate_fixed,device); fm.append({'epoch':e,'metrics':compute_metrics(y,p,data.label_mapping)[0]})
        elif name=='variable_snp':
          length=fixed_context_length(config); d=ds[length]; v=config['variable_snp']; ks=sorted(set(max(int(v['min_snps']),min(data.n_aim,round(float(f)*data.n_aim))) for f in v['eval_fractions'])); train=VariableSNPDataset(data.genotypes,data.rsid_order,d,tr,buckets=resolve_buckets(data.n_aim,int(v['min_snps']),v['bucket_fractions']),seed=int(config['project']['seed']),fold_id=fi,precomputed=pre[length]); val=FixedFullDataset(data.genotypes,data.rsid_order,d,va,precomputed=pre[length]); vf=lambda m:(compute_metrics(*_predict(m,val,config,collate_fixed,device),data.label_mapping)[0]['Macro-F1'],0.0); model,e=_fit(train,vf,d.shape[1]*2,data.n_aim,config,collate_variable_snp,device); fm.append({'epoch':e,'eval_k_values':ks})
        else:
          train_lengths=variable_context_lengths(config,'train'); eval_lengths=variable_context_lengths(config,'eval'); train=VariableContextDataset(data.genotypes,data.rsid_order,ds,train_lengths,sample_ids=tr,seed=int(config['project']['seed']),fold_id=fi,precomputed_by_length=pre); vals={str(l):FixedFullDataset(data.genotypes,data.rsid_order,ds[int(l)],va,precomputed=pre[int(l)]) for l in eval_lengths}; vf=lambda m:(float(np.mean([compute_metrics(*_predict(m,x,config,collate_fixed,device),data.label_mapping)[0]['Macro-F1'] for x in vals.values()])),0.0); model,e=_fit(train,vf,next(iter(ds.values())).shape[1]*2,data.n_aim,config,collate_fixed,device); fm.append({'epoch':e,'context_lengths':list(vals)})
      out={'schema_version':'2.0','pipeline_name':name,'completed':True,'x':int(config['data']['added_aims_per_autosome']),'n_aim':data.n_aim,'cv':{'n_splits':5,'seed':int(config['project']['seed'])},'fold_metrics':fm,'provenance':{'created_at_utc':utc_now_iso()}}
      if name=='variable_snp': out.update(context_length=fixed_context_length(config),eval_k_values=ks)
      if name=='variable_context': out.update(train_lengths=variable_context_lengths(config,'train'),eval_lengths=variable_context_lengths(config,'eval'))
      outputs[name]=out; outpath=root/config['results'][name+'_json']; outpath.parent.mkdir(parents=True,exist_ok=True); outpath.write_text(json.dumps(out,indent=2),encoding='utf-8')
    return outputs
