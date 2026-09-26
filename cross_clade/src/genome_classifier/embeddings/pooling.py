from __future__ import annotations
def mean_pool_segments(hidden, segments=32):
    """Pool [B,T,H] hidden states into contiguous segment means [B,S,H]."""
    shape = getattr(hidden, "shape", None)
    if shape is None or len(shape) != 3: raise ValueError("hidden state must have shape [B, T, H]")
    b, tokens, width = map(int, shape)
    if tokens % segments: raise ValueError(f"token count {tokens} is not divisible by segments {segments}")
    segment_len = tokens // segments
    view = hidden.reshape(b, segments, segment_len, width)
    if hasattr(view, "dtype") and not (getattr(view.dtype, "is_floating_point", False) or getattr(view.dtype, "is_complex", False)):
        view = view.float()
    return view.mean(dim=2)

def normalize_hidden(hidden, batch_size=None, tokens=8192, width=4096):
    shape = tuple(int(x) for x in getattr(hidden, "shape", ()))
    if len(shape) == 3 and shape[1:] == (width, tokens):
        hidden = hidden.transpose(1, 2)
        shape = tuple(int(x) for x in hidden.shape)
    expected = (batch_size, tokens, width) if batch_size is not None else None
    if len(shape) != 3 or (batch_size is not None and shape != expected) or shape[1:] != (tokens, width):
        raise ValueError(f"expected hidden state [B, {tokens}, {width}], got {shape}")
    return hidden.contiguous()
