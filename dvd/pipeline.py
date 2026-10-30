"""Frames in, depth out: what the DVD Video Depth node runs (and what the tests measure)."""

import comfy.model_management
import comfy.utils
import torch

from . import depth as D
from .sampling import empty_context, predict, regression_model


def estimate(model, vae, images, height=480, width=640, window_size=81, overlap=21, denoise_step=0.5, scale_only=False,
             resize_back=True, progress=True, keep_windows=False):
    """images: (T, H, W, C) in 0..1 (a ComfyUI IMAGE). Returns depth (T, H', W') float32 on the intermediate device,
    the raw decoded values (relative, affine-invariant disparity-like: near is bright), not normalised.

    `model` is the Wan 2.1 DVD model from Load Diffusion Model, `vae` the Wan 2.1 VAE. Each window of frames is
    resized to cover height x width (aspect kept, multiples of 16), padded to 4n+1 frames, encoded, regressed in one
    pass and decoded; windows are aligned and cross-faded over `overlap` frames. With keep_windows the per-window
    results (before alignment) are returned too.
    """
    model = regression_model(model)
    context = empty_context()
    total, in_h, in_w = images.shape[:3]
    proc_h, proc_w = D.fit_size(in_h, in_w, height, width)
    ranges = D.windows(total, window_size, overlap)
    device = comfy.model_management.get_torch_device()
    bar = comfy.utils.ProgressBar(len(ranges)) if progress else None

    outputs = []
    for start, end in ranges:
        comfy.model_management.throw_exception_if_processing_interrupted()
        frames, count = D.pad_frames(images[start:end])
        frames = torch.cat([D.resize_frames(chunk.to(device, torch.float32), proc_h, proc_w) for chunk in frames.split(16)])
        latent = vae.encode(frames[..., :3])
        depth_latent = predict(model, latent, context, denoise_step)
        decoded = vae.decode(depth_latent)  # (1, T, H, W, 3) in 0..1
        decoded = decoded.reshape(-1, *decoded.shape[-3:])[:count]
        outputs.append(decoded.float().mean(dim=-1))
        if bar is not None:
            bar.update(1)

    depth = D.merge_windows(outputs, ranges, scale_only)
    if resize_back and (in_h, in_w) != (proc_h, proc_w):
        depth = torch.cat([D.resize_frames(chunk[..., None], in_h, in_w)[..., 0] for chunk in depth.split(64)])
    return (depth, outputs) if keep_windows else depth
