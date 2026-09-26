from __future__ import annotations

import torch

from .data import VALID_GT


def contextual_delta(ref_embedding: torch.Tensor, alt_embedding: torch.Tensor) -> torch.Tensor:
    return alt_embedding - ref_embedding


def parse_genotype(gt: str) -> tuple[int, int]:
    if gt not in VALID_GT:
        raise ValueError(f"Invalid genotype {gt!r}")
    return int(gt[0]), int(gt[2])


def diploid_locus_representation(gt: str, delta: torch.Tensor) -> torch.Tensor:
    a1, a2 = parse_genotype(gt)
    zero = torch.zeros_like(delta)
    d1 = delta if a1 == 1 else zero
    d2 = delta if a2 == 1 else zero
    mean_effect = (d1 + d2) / 2
    het_effect = torch.abs(d1 - d2)
    return torch.cat([mean_effect, het_effect], dim=-1)


def build_sample_representation(genotypes: list[str], deltas: torch.Tensor) -> torch.Tensor:
    if len(genotypes) != deltas.shape[0]:
        raise ValueError(f"Expected {deltas.shape[0]} genotypes, found {len(genotypes)}")
    return torch.stack(
        [diploid_locus_representation(gt, deltas[index]) for index, gt in enumerate(genotypes)],
        dim=0,
    )


def load_delta_embeddings(embedding_path: str, pool_radius: int, rsid_order: list[str], context_length: int | None = None) -> torch.Tensor:
    data = torch.load(embedding_path, map_location="cpu")
    # New caches are indexed by context length then pool radius; accept the
    # legacy radius-first layout for backwards compatibility.
    if context_length is not None and context_length in data:
        data = data[context_length]
    elif context_length is not None and str(context_length) in data:
        data = data[str(context_length)]
    if context_length is not None and not isinstance(data, dict):
        raise ValueError(f"context length {context_length} cache entry is not a mapping")
    pool_key = str(pool_radius)
    if pool_key not in data and pool_radius not in data:
        raise KeyError(f"pool radius {pool_radius} is absent from embedding cache")
    embeddings = data[pool_key] if pool_key in data else data[pool_radius]
    deltas = []
    for rsid in rsid_order:
        if rsid not in embeddings:
            raise ValueError(f"embedding cache is missing rsID {rsid} for context {context_length}")
        if 0 not in embeddings[rsid] or 1 not in embeddings[rsid]:
            raise ValueError(f"embedding cache is missing REF/ALT allele for {rsid}")
        ref = embeddings[rsid][0]
        alt = embeddings[rsid][1]
        deltas.append(contextual_delta(ref.float(), alt.float()))
    result = torch.stack(deltas, dim=0).float().cpu()
    if result.shape[0] != len(rsid_order):
        raise ValueError("delta embedding rsID dimension does not match requested order")
    return result


def load_delta_embeddings_by_context(embedding_path: str, pool_radius: int, rsid_order: list[str], context_lengths: list[int]) -> dict[int, torch.Tensor]:
    """Load context-specific ALT-REF effects and reject incomplete caches."""
    result = {}
    for length in sorted({int(x) for x in context_lengths}):
        result[length] = load_delta_embeddings(embedding_path, pool_radius, rsid_order, length)
    return result
