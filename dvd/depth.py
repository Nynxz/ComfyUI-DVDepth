"""The video side of DVD's inference (test_script/test_single_video.py in the DVD repo), in torch.

A clip is processed in windows of 4n+1 frames; each window is resized to cover the training box, encoded with
the Wan VAE, regressed in one pass and decoded. Consecutive windows share `overlap` frames: the new window is
fitted to the previous one with a least-squares scale and shift on the shared frames (the paper's global affine
coherence), clipped to [0.7, 1.5], and the two are cross-faded over the overlap.
"""

import math

import torch
import torch.nn.functional as F

SCALE_RANGE = (0.7, 1.5)


def fit_size(height, width, target_height, target_width):
    """The size the reference resizes to: cover the target box keeping the aspect ratio, rounded up to 16."""
    ratio = max(target_height / height, target_width / width)
    h = math.ceil(math.ceil(height * ratio) / 16) * 16
    w = math.ceil(math.ceil(width * ratio) / 16) * 16
    return h, w


def resize_frames(frames, height, width):
    """(T, H, W, C) -> (T, height, width, C), bilinear like the reference (torch.nn.functional.interpolate)."""
    if frames.shape[1] == height and frames.shape[2] == width:
        return frames
    x = frames.movedim(-1, 1)
    x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
    return x.movedim(1, -1)


def pad_frames(frames):
    """Repeat the last frame until the count is 4n+1 (the Wan VAE's temporal stride). Returns (frames, original_count)."""
    t = frames.shape[0]
    remainder = t % 4
    if remainder != 1:
        pad = (4 - remainder + 1) % 4
        frames = torch.cat([frames, frames[-1:].expand(pad, *frames.shape[1:])], dim=0)
    return frames, t


def windows(total, size, overlap):
    """Frame ranges [(start, end), ...]; the last window is shifted back so every window has `size` frames."""
    size = max(1, min(size, total))
    overlap = max(0, min(overlap, size - 1))
    if total <= size:
        return [(0, total)]
    out = [(0, size)]
    start = size - overlap
    while start < total:
        end = start + size
        if end < total:
            out.append((start, end))
            start += size - overlap
        else:
            start = max(0, total - size)
            out.append((start, total))
            break
    return out


def scale_and_shift(current, reference):
    """Least-squares a, b with a * current + b ~= reference."""
    cur = current.double()
    ref = reference.double()
    a00 = (cur * cur).sum()
    a01 = cur.sum()
    a11 = torch.tensor(float(cur.numel()), dtype=torch.float64, device=cur.device)
    b0 = (cur * ref).sum()
    b1 = ref.sum()
    det = a00 * a11 - a01 * a01
    if det == 0:
        return 1.0, 0.0
    return float((a11 * b0 - a01 * b1) / det), float((-a01 * b0 + a00 * b1) / det)


def merge_windows(depths, ranges, scale_only=False):
    """Stitch per-window depth (T_i, H, W) tensors into one (T, H, W) clip, aligning and cross-fading the overlaps."""
    merged = depths[0]
    prev_end = ranges[0][1]
    for depth, (start, end) in zip(depths[1:], ranges[1:]):
        overlap = prev_end - start
        if overlap > 0:
            ref = merged[-overlap:]
            cur = depth[:overlap]
            if scale_only:
                scale = float((cur.double() * ref.double()).sum() / ((cur.double() * cur.double()).sum() + 1e-6))
                shift = 0.0
            else:
                scale, shift = scale_and_shift(cur, ref)
            scale = min(max(scale, SCALE_RANGE[0]), SCALE_RANGE[1])
            aligned = (depth * scale + shift).clamp_min_(0)
            alpha = torch.linspace(0, 1, overlap, dtype=depth.dtype, device=depth.device)[:, None, None]
            blended = (1 - alpha) * ref + alpha * aligned[:overlap]
            merged = torch.cat([merged[:-overlap], blended, aligned[overlap:]], dim=0)
        else:
            merged = torch.cat([merged, depth], dim=0)
        prev_end = end
    return merged


def normalize(depth, mode):
    """'video': min/max over the clip (DVD's visualisation), 'frame': per frame, 'none': as decoded."""
    if mode == "video":
        lo, hi = depth.min(), depth.max()
        return (depth - lo) / (hi - lo + 1e-8)
    if mode == "frame":
        flat = depth.flatten(1)
        lo = flat.min(dim=1).values[:, None, None]
        hi = flat.max(dim=1).values[:, None, None]
        return (depth - lo) / (hi - lo + 1e-8)
    return depth
