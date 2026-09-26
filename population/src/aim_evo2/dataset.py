from __future__ import annotations

import hashlib
import math

import pandas as pd
import torch
from torch.utils.data import Dataset

from .representations import build_sample_representation


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def resolve_buckets(n_aim: int, min_snps: int, fractions: list[list[float]]) -> list[tuple[int, int]]:
    buckets = []
    for low, high in fractions:
        left, right = max(min_snps, math.ceil(low * n_aim)), min(n_aim, max(min_snps, math.floor(high * n_aim)))
        if left <= right and (left, right) not in buckets: buckets.append((left, right))
    if not buckets: raise ValueError(f"no valid SNP buckets for N_AIM={n_aim}")
    return buckets


class _DatasetBase(Dataset):
    def __init__(self, genotypes: pd.DataFrame, rsid_order: list[str], deltas: torch.Tensor, sample_ids: list[str] | None = None, precomputed: torch.Tensor | None = None):
        source_frame = genotypes.reset_index(drop=True)
        positions = {sid: i for i, sid in enumerate(source_frame["SampleID"])}
        self.frame = (genotypes if sample_ids is None else genotypes.set_index("SampleID").loc[sample_ids].reset_index()).reset_index(drop=True)
        self._sample_positions = [positions[sid] for sid in self.frame["SampleID"]]
        self.rsid_order, self.deltas, self.precomputed = rsid_order, deltas, precomputed
    def __len__(self): return len(self.frame)
    def make_item(self, index: int, selected: list[int], deltas: torch.Tensor | None = None):
        row = self.frame.iloc[index]
        source = self.deltas if deltas is None else deltas
        if self.precomputed is not None:
            features = self.precomputed[int(self._sample_positions[index])][selected].float()
        else:
            features = build_sample_representation([row[self.rsid_order[i]] for i in selected], source[selected])
        return {"sample_id": row.SampleID, "features": features, "locus_indices": torch.tensor(selected, dtype=torch.long), "label": torch.tensor(int(row.Label), dtype=torch.long)}


class FixedFullDataset(_DatasetBase):
    def __getitem__(self, index): return self.make_item(index, list(range(len(self.rsid_order))))


class VariableSNPDataset(_DatasetBase):
    def __init__(self, *args, buckets: list[tuple[int, int]], seed: int, fold_id: int, static_subsets: dict[str, list[int]] | None = None, **kwargs):
        super().__init__(*args, **kwargs); self.buckets, self.seed, self.fold_id, self.static_subsets, self.epoch = buckets, seed, fold_id, static_subsets, 0
    def set_epoch(self, epoch: int): self.epoch = epoch
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        if self.static_subsets is not None:
            selected = self.static_subsets[str(row.SampleID)]
        else:
            worker = torch.utils.data.get_worker_info()
            worker_id = 0 if worker is None else worker.id
            rng = torch.Generator().manual_seed(stable_seed(self.seed, "variable_snp", self.fold_id, self.epoch, worker_id, row.SampleID))
            low, high = self.buckets[int(torch.randint(len(self.buckets), (1,), generator=rng))]
            selected = torch.randperm(len(self.rsid_order), generator=rng)[:int(torch.randint(low, high + 1, (1,), generator=rng))].tolist()
        return self.make_item(index, selected)


class StaticVariableSNPDataset(_DatasetBase):
    """Deterministic SNP subsets for validation or final test evaluation."""
    def __init__(self, *args, subsets: dict[tuple[str, int, int], list[int]], k_values: list[int], replicates: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.subsets = subsets
        self.k_values = list(k_values)
        self.replicates = int(replicates)
        self.instances = [(sid, k, rep) for sid in self.frame.SampleID for k in self.k_values for rep in range(self.replicates)]

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        sample_id, k, rep = self.instances[index]
        row_index = int(self.frame.index[self.frame.SampleID == sample_id][0])
        selected = self.subsets[(str(sample_id), int(k), int(rep))]
        return self.make_item(row_index, selected)


class VariableContextDataset(_DatasetBase):
    def __init__(self, genotypes, rsid_order, deltas_by_length, lengths, seed=42, fold_id=0, sample_ids=None, static_length=None, precomputed_by_length=None):
        super().__init__(genotypes, rsid_order, next(iter(deltas_by_length.values())), sample_ids); self.by_length, self.precomputed_by_length, self.lengths, self.seed, self.fold_id, self.static_length, self.epoch = deltas_by_length, precomputed_by_length, list(lengths), seed, fold_id, static_length, 0
    def set_epoch(self, epoch): self.epoch = epoch
    def __getitem__(self, index):
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else worker.id
        length = self.static_length or self.lengths[stable_seed(self.seed, "variable_context", self.fold_id, self.epoch, worker_id, self.frame.iloc[index].SampleID) % len(self.lengths)]
        old = self.precomputed
        if self.precomputed_by_length is not None: self.precomputed = self.precomputed_by_length[int(length)]
        item = self.make_item(index, list(range(len(self.rsid_order))), self.by_length[length]); self.precomputed = old; item["context_length"] = length; return item


def collate_variable_snp(items):
    max_k, dim = max(x["features"].shape[0] for x in items), items[0]["features"].shape[1]
    features = torch.zeros((len(items), max_k, dim)); indices = torch.zeros((len(items), max_k), dtype=torch.long); mask = torch.zeros((len(items), max_k), dtype=torch.bool)
    for i, item in enumerate(items):
        k = item["features"].shape[0]; features[i, :k], indices[i, :k], mask[i, :k] = item["features"], item["locus_indices"], True
    return {"sample_id": [x["sample_id"] for x in items], "features": features, "locus_indices": indices, "attention_mask": mask, "label": torch.stack([x["label"] for x in items])}


collate_fixed = collate_variable_snp

# Backward-compatible name used by the original training modules.
AimRepresentationDataset = FixedFullDataset


def collate_variable_context(items):
    batch = collate_variable_snp(items)
    batch["context_length"] = torch.tensor([int(x["context_length"]) for x in items], dtype=torch.long)
    return batch


def make_static_subsets(sample_ids: list[str], n_aim: int, k_values: list[int], replicates: int, seed: int, fold_id: int, split_name: str) -> dict[str, list[int]]:
    """Create deterministic, label-independent SNP subsets for validation/test."""
    result: dict[str, list[int]] = {}
    for sample_id in sample_ids:
        for k in k_values:
            for rep in range(replicates):
                rng = torch.Generator().manual_seed(stable_seed(seed, fold_id, split_name, sample_id, k, rep))
                selected = torch.randperm(n_aim, generator=rng)[: int(k)].tolist()
                result[f"{sample_id}|{k}|{rep}"] = selected
    return result


def make_static_subset_map(sample_ids: list[str], n_aim: int, k_values: list[int], replicates: int, seed: int, fold_id: int, split_name: str) -> dict[tuple[str, int, int], list[int]]:
    """Return deterministic, label-independent subsets keyed by sample/K/replicate."""
    result = {}
    for sample_id in sample_ids:
        for k in k_values:
            if not 1 <= int(k) <= n_aim:
                raise ValueError(f"invalid evaluation K={k} for N_AIM={n_aim}")
            for rep in range(replicates):
                rng = torch.Generator().manual_seed(stable_seed(seed, fold_id, split_name, sample_id, int(k), rep))
                result[(str(sample_id), int(k), int(rep))] = torch.randperm(n_aim, generator=rng)[: int(k)].tolist()
    return result
