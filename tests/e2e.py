"""Headless end-to-end run through the stock ComfyUI nodes plus DVD Video Depth (the graph of the example workflow).

    <comfy>/.venv/bin/python tests/e2e.py --comfy <ComfyUI> --video clip.mp4 --out <dir>
        [--unet dvd_1.1_bf16.safetensors] [--vae wan_2.1_vae.safetensors] [--height 480 --width 640]
        [--window 81 --overlap 21] [--frames N] [--colormap grayscale|turbo] [--cpu]

Model names are relative to the configured model folders (extra_model_paths.yaml is honoured). Writes
<name>_depth.mp4 (the depth video at the input's frame rate), <name>_sheet.png (frames over their depth) and prints
timings.
"""

import argparse
import importlib
import os
import sys
import time
from fractions import Fraction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--unet", default="dvd_1.1_bf16.safetensors")
    ap.add_argument("--vae", default="wan_2.1_vae.safetensors")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--window", type=int, default=81)
    ap.add_argument("--overlap", type=int, default=21)
    ap.add_argument("--frames", type=int, default=0, help="use only the first N frames (0 = all)")
    ap.add_argument("--colormap", default="grayscale", choices=["grayscale", "turbo"])
    ap.add_argument("--normalize", default="video", choices=["video", "frame", "none"])
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    pack_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, args.comfy)
    sys.argv = sys.argv[:1] + (["--cpu"] if args.cpu else [])
    import comfy.options
    comfy.options.enable_args_parsing()
    from utils.extra_config import load_extra_path_config
    extra = os.path.join(args.comfy, "extra_model_paths.yaml")
    if os.path.isfile(extra):
        load_extra_path_config(extra)

    import torch
    sys.path.insert(0, os.path.dirname(pack_root))
    pack = importlib.import_module(os.path.basename(pack_root))
    import nodes
    from comfy_api.latest import InputImpl, Types

    with torch.inference_mode():  # as ComfyUI's executor runs nodes
        run(args, nodes, pack, InputImpl, Types)


def run(args, nodes, pack, InputImpl, Types):
    DVDVideoDepth = pack.dvd.nodes.DVDVideoDepth
    import numpy as np
    import torch
    from PIL import Image

    os.makedirs(args.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.video))[0]
    t0 = time.time()
    components = InputImpl.VideoFromFile(args.video).get_components()  # what Load Video + Get Video Components do
    images = components.images
    if args.frames:
        images = images[: args.frames]
    fps = components.frame_rate
    print(f"video: {tuple(images.shape)} at {float(fps):.3f} fps, decoded in {time.time() - t0:.1f}s")

    t0 = time.time()
    (model,) = nodes.UNETLoader().load_unet(args.unet, "default")
    (vae,) = nodes.VAELoader().load_vae(args.vae)
    print(f"loaded {args.unet} ({model.model.get_dtype()}) and {args.vae} in {time.time() - t0:.1f}s")

    t0 = time.time()
    (depth,) = DVDVideoDepth.execute(model, vae, images, args.height, args.width, args.window, args.overlap, args.normalize, args.colormap,
                                     False, "input", 0.5, False).result
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t0
    n_windows = len(pack.dvd.depth.windows(images.shape[0], args.window, args.overlap))
    print(f"DVD Video Depth: {tuple(depth.shape)} in {dt:.1f}s ({n_windows} windows of {args.window}, {dt / n_windows:.1f}s each, "
          f"{images.shape[0] / dt:.1f} frames/s)")
    if torch.cuda.is_available():
        print(f"peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    out_video = os.path.join(args.out, f"{name}_depth.mp4")
    video = InputImpl.VideoFromComponents(Types.VideoComponents(images=depth, audio=None, frame_rate=Fraction(fps)))
    video.save_to(out_video, format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264)
    picks = np.linspace(0, images.shape[0] - 1, min(4, images.shape[0])).round().astype(int)
    top = np.concatenate([(images[i].numpy() * 255).astype(np.uint8) for i in picks], axis=1)
    bottom = np.concatenate([(depth[i].numpy() * 255).astype(np.uint8) for i in picks], axis=1)
    sheet = os.path.join(args.out, f"{name}_sheet.png")
    Image.fromarray(np.concatenate([top, bottom], axis=0)).save(sheet)
    print(f"wrote {out_video} and {sheet} (frames {picks.tolist()})")


if __name__ == "__main__":
    main()
