from __future__ import annotations
from pathlib import Path
import csv
from .pooling import normalize_hidden, mean_pool_segments
from .cache import ShardedCache

def load_evo2(name, local_path):
    try:
        from evo2 import Evo2
    except ImportError as exc:
        raise RuntimeError("Evo2 is required for extraction and must be installed on Linux") from exc
    model = Evo2(name, local_path=local_path); model.eval()
    for parameter in model.parameters(): parameter.requires_grad = False
    assert all(not parameter.requires_grad for parameter in model.parameters())
    return model

def extract_layer(model, tokens, layer):
    """Call Evo2 while accepting common API return forms."""
    try:
        result = model(tokens, return_embeddings=True, layer_names=[layer])
    except TypeError:
        result = model(tokens, return_embeddings=True, layer=layer)
    if isinstance(result, dict):
        if layer in result: return result[layer]
        for key in ("embeddings", "hidden_states", "logits"):
            if key in result and hasattr(result[key], "shape"): return result[key]
    if isinstance(result, (tuple, list)): result = result[0]
    return result

def sequence_from_fasta(row, records):
    source = row["source_fasta"]
    if source not in records:
        from genome_classifier.data.fasta import read_fasta
        records[source] = {r.sequence_id:r.sequence for r in read_fasta(Path(source))}
    seq=records[source].get(row["sequence_id"])
    if seq is None: raise KeyError(f"Sequence {row['sequence_id']} not found in {row['source_fasta']}")
    return seq[int(row["start"]):int(row["end"])]

def extract_embeddings(model, token_batches, window_ids, cache, layer="blocks.28.mlp.l3", segments=32, batch_size=1):
    """Extract pre-tokenized batches, reducing batch size on CUDA OOM."""
    import torch
    pending = list(zip(window_ids, token_batches)); pos = 0; current = max(1, int(batch_size))
    while pos < len(pending):
        chunk = pending[pos:pos + current]; ids = [x[0] for x in chunk]; tokens = [x[1] for x in chunk]
        try:
            batch = torch.stack(tokens) if hasattr(tokens[0], "shape") else tokens
            with torch.inference_mode():
                hidden = normalize_hidden(extract_layer(model, batch, layer))
                pooled = mean_pool_segments(hidden, segments)
            cache.add(ids, pooled); pos += len(chunk)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or current == 1: raise RuntimeError("Evo2 extraction failed with CUDA OOM at batch size 1") from exc
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            current = max(1, current // 2)
