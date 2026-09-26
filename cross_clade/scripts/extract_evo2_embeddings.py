#!/usr/bin/env python3
from pathlib import Path
import argparse, csv, yaml, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from genome_classifier.embeddings.extractor import load_evo2, sequence_from_fasta, extract_embeddings
from genome_classifier.embeddings.cache import ShardedCache

def main():
    p=argparse.ArgumentParser(description="Extract frozen Evo2 embeddings into sharded cache"); p.add_argument("--config",required=True); args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    model=load_evo2(cfg["model"]["name"], cfg["model"]["local_path"]); print("Evo2 trainable params: 0")
    cache=ShardedCache(cfg["cache_dir"], cfg["inference"].get("shard_size",256)); done=cache.index
    rows=[]; seen=set(); records={}
    with Path(cfg["selection_file"]).open(newline="") as f:
        for row in csv.DictReader(f):
            wid=row["window_id"]
            if wid not in done and wid not in seen: rows.append(row); seen.add(wid)
    tokenizer=getattr(model, "tokenizer", None)
    if rows and tokenizer is None:
        raise RuntimeError("Evo2 model does not expose a tokenizer; provide tokenization in the server adapter")
    ids=[]; tokens=[]
    for row in rows:
        seq=sequence_from_fasta(row, records)
        encoded=tokenizer.encode(seq) if hasattr(tokenizer,"encode") else tokenizer(seq)
        import torch
        tokens.append(torch.as_tensor(encoded)); ids.append(row["window_id"])
    if ids: extract_embeddings(model, tokens, ids, cache, cfg["model"]["layer"], cfg["pooling"]["segments"], cfg["inference"]["batch_size"])
    cache.write_meta({"model":cfg["model"]["name"],"checkpoint":cfg["model"]["local_path"],"layer":cfg["model"]["layer"],"pooling":f"{cfg['pooling']['segments']} contiguous mean segments","embedding_shape":[32,4096],"dtype":"float16","window_size":8192})
if __name__ == "__main__": main()
