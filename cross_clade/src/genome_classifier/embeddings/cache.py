from __future__ import annotations
import csv, json, os
from pathlib import Path

def load_index(path):
    if not path.exists(): return {}
    with path.open(newline="") as f: return {r["window_id"]: (r["shard"], int(r["row"])) for r in csv.DictReader(f)}

class ShardedCache:
    def __init__(self, cache_dir, shard_size=256):
        self.root=Path(cache_dir); self.shards=self.root/"shards"; self.shards.mkdir(parents=True, exist_ok=True)
        self.index_path=self.root/"embedding_index.csv"; self.index=load_index(self.index_path); self.shard_size=shard_size
    def add(self, ids, tensor):
        import torch
        tensor=tensor.detach().to(dtype=torch.float16, device="cpu")
        if tensor.shape[0] != len(ids): raise ValueError("ids and tensor batch differ")
        rows=list(tensor)
        with self.index_path.open("a", newline="") as f:
            writer=csv.DictWriter(f, fieldnames=["window_id","shard","row"])
            if f.tell()==0: writer.writeheader()
            existing_numbers = [int(p.stem.split("_")[1]) for p in self.shards.glob("shard_*.pt")]
            shard_no = max(existing_numbers) if existing_numbers else 0
            while rows:
                path=self.shards/f"shard_{shard_no:05d}.pt"
                existing=torch.load(path) if path.exists() else []
                if len(existing) >= self.shard_size:
                    shard_no += 1; continue
                room=self.shard_size-len(existing); take=min(room,len(rows))
                existing.extend(rows[:take]); torch.save(torch.stack(existing), path.with_suffix(".tmp")); os.replace(path.with_suffix(".tmp"), path)
                for i,wid in enumerate(ids[:take]): self.index[wid]=(path.name, len(existing)-take+i); writer.writerow({"window_id":wid,"shard":path.name,"row":len(existing)-take+i})
                ids,rows=ids[take:],rows[take:]
                if rows: shard_no += 1
    def write_meta(self, meta):
        tmp=self.root/"cache_meta.json.tmp"; tmp.write_text(json.dumps(meta, indent=2)); os.replace(tmp, self.root/"cache_meta.json")
