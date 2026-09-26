from pathlib import Path
import csv
from collections import OrderedDict
import torch
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):
    def __init__(self, selection_file, index_file, cache_dir, fold, split, max_cached_shards=2):
        self.cache_dir = Path(cache_dir)
        self.index = {}
        with Path(index_file).open(newline="") as f:
            for r in csv.DictReader(f): self.index[r["window_id"]] = (r["shard"], int(r["row"]))
        self.rows = []
        with Path(selection_file).open(newline="") as f:
            for r in csv.DictReader(f):
                if int(r["experiment_fold"]) == int(fold) and r["split"] == split:
                    if r["window_id"] not in self.index: raise KeyError(f"Missing embedding index for {r['window_id']}")
                    self.rows.append((r["window_id"], int(r["class_id"])))
        self.max_cached_shards = max_cached_shards; self._cache = OrderedDict()
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        wid, label = self.rows[i]; shard, row = self.index[wid]
        if shard not in self._cache:
            try:
                value = torch.load(self.cache_dir / "shards" / shard, map_location="cpu", weights_only=True)
            except TypeError:
                value = torch.load(self.cache_dir / "shards" / shard, map_location="cpu")
            self._cache[shard] = value
            while len(self._cache) > self.max_cached_shards: self._cache.popitem(last=False)
        value = self._cache.pop(shard); self._cache[shard] = value
        return value[row].to(torch.float32), torch.tensor(label, dtype=torch.long)
