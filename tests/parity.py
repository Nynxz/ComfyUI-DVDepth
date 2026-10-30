"""Numerical parity of the ComfyUI path against the reference DVD implementation (tests/reference.py outputs).

    <comfy>/.venv/bin/python tests/parity.py --comfy <ComfyUI> --model <converted dvd_1.1_bf16.safetensors> \
        --unit unit.npz [--e2e e2e.npz --vae <wan_2.1_vae.safetensors> --out <dir>]

unit.npz (tests/reference.py unit): the DiT alone on a fixed random latent at timestep 500 with the zero context,
from the reference in fp32 (peft-merged weights, and the converter's merged weights) and in bf16. The ComfyUI side
runs the same latent through the node's sampling path (regression_model + predict, i.e. CFGGuider) from the
converted file in fp32 (implementation parity), in fp16 (what Load Diffusion Model picks on most GPUs) and bf16.

e2e.npz (tests/reference.py video): the reference's per-window depth and stitched clip for N frames of a video;
the ComfyUI side runs dvd.pipeline.estimate on the same frames and compares windows, the stitched clip at model
and input size, and writes side-by-side PNGs.
"""

import argparse
import importlib
import os
import sys
import time


def setup(comfy):
    pack_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, comfy)
    sys.path.insert(0, os.path.dirname(pack_root))
    sys.argv = sys.argv[:1]
    import comfy.options
    comfy.options.enable_args_parsing()
    return importlib.import_module(os.path.basename(pack_root))


def report(name, a, b, tol):
    a, b = a.float().flatten().cpu(), b.float().flatten().cpu()
    rel = float((a - b).norm() / b.norm().clamp_min(1e-12))
    cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    ok = rel <= tol
    print(f"  {'ok ' if ok else 'BAD'} {name:<44} rel err {rel:.2e}  max abs {float((a - b).abs().max()):.2e}  cos {cos:.6f}")
    return ok


def unit(args, pack):
    import comfy.model_management
    import comfy.sd
    import numpy as np
    ref = np.load(args.unit)
    x = torch.from_numpy(ref["x"])
    ok = True
    print("DiT: depth latents from the video latent (timestep 500, zero context)")
    for label, options, targets in (
        ("fp32", {"dtype": torch.float32}, [("reference fp32, converter weights", "y_fp32_convertedweights", 2e-4), ("reference fp32, peft merge", "y_fp32", 5e-2)]),
        ("default (fp16 on this GPU)", {}, [("reference bf16", "y_bf16", 1e-1), ("reference fp32", "y_fp32", 1e-1)]),
        ("bf16", {"dtype": torch.bfloat16}, [("reference bf16", "y_bf16", 1e-1), ("reference fp32", "y_fp32", 1e-1)]),
    ):
        model = comfy.sd.load_diffusion_model(args.model, model_options=options)
        assert type(model.model.model_config).__name__ == "WAN21_T2V", type(model.model.model_config).__name__
        fmt = model.model.latent_format
        raw = fmt.process_out(x)  # the npz holds the normalised latent the DiT sees; predict() normalises on the way in
        m = pack.dvd.sampling.regression_model(model)
        with torch.inference_mode():
            pack.dvd.sampling.predict(m, raw[:, :, :1, :8, :8])  # warm up (loads the model)
            torch.cuda.synchronize()
            t0 = time.time()
            y = pack.dvd.sampling.predict(m, raw)
            torch.cuda.synchronize()
        dt = time.time() - t0
        y = fmt.process_in(y.float())
        print(f"  ComfyUI {label}: model dtype {model.model.get_dtype()}, forward {dt:.2f}s for latent {tuple(x.shape)}")
        for name, key, tol in targets:
            if key in ref:
                ok &= report(f"{label} vs {name}", y, torch.from_numpy(ref[key]), tol)
        del m, model
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
    return ok


def e2e(args, pack):
    import comfy.model_management
    import comfy.sd
    import comfy.utils
    import numpy as np
    from PIL import Image
    ref = np.load(args.e2e)
    frames = torch.from_numpy(ref["frames"]).float() / 255.0  # T H W 3
    windows = [tuple(int(v) for v in w) for w in ref["windows"]]
    size = windows[0][1] - windows[0][0]
    overlap = windows[0][1] - windows[1][0] if len(windows) > 1 else 0
    print(f"video: {tuple(frames.shape)} frames, windows {windows} (size {size}, overlap {overlap})")
    model = comfy.sd.load_diffusion_model(args.model)
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(args.vae))
    with torch.inference_mode():
        torch.cuda.synchronize()
        t0 = time.time()
        depth_model_res, raw = pack.dvd.pipeline.estimate(model, vae, frames, args.height, args.width, size, overlap, resize_back=False,
                                                           progress=False, keep_windows=True)
        torch.cuda.synchronize()
        dt = time.time() - t0
        depth = torch.cat([pack.dvd.depth.resize_frames(c[..., None], frames.shape[1], frames.shape[2])[..., 0] for c in depth_model_res.split(64)])
    print(f"  ComfyUI: {len(raw)} windows in {dt:.1f}s ({dt / len(raw):.2f}s per window of {size} frames at {tuple(depth_model_res.shape[1:])})")
    ok = True
    for i, w in enumerate(raw):
        ok &= report(f"window {i} raw depth (model res)", w, torch.from_numpy(ref[f"raw_{i}"]), 5e-2)
    ok &= report("stitched clip, model res", depth_model_res, torch.from_numpy(ref["depth_model_res"]).mean(-1), 5e-2)
    ok &= report("stitched clip, input res", depth, torch.from_numpy(ref["depth"]).mean(-1), 5e-2)
    r = torch.from_numpy(ref["depth"]).mean(-1)
    rn = (r - r.min()) / (r.max() - r.min() + 1e-8)
    dn = pack.dvd.depth.normalize(depth, "video")
    print(f"  normalised clips: mean abs diff {float((rn - dn).abs().mean()):.4f} (0..1 scale), 99th pct {float((rn - dn).abs().flatten().kthvalue(int(0.99 * rn.numel())).values):.4f}")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        picks = [0, frames.shape[0] // 2, frames.shape[0] - 1]
        rows = []
        for t in picks:
            rgb = (frames[t].numpy() * 255).astype(np.uint8)
            a = (rn[t].numpy() * 255).astype(np.uint8)
            b = (dn[t].numpy() * 255).astype(np.uint8)
            diff = (np.abs(rn[t].numpy() - dn[t].numpy()) * 255 * 10).clip(0, 255).astype(np.uint8)
            rows.append(np.concatenate([rgb, np.stack([a] * 3, -1), np.stack([b] * 3, -1), np.stack([diff] * 3, -1)], axis=1))
        Image.fromarray(np.concatenate(rows, axis=0)).save(os.path.join(args.out, "parity_e2e.png"))
        print(f"  wrote {os.path.join(args.out, 'parity_e2e.png')} (frame | reference | ComfyUI | 10x |diff|, frames {picks})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--model", required=True, help="converted dvd_*_bf16.safetensors")
    ap.add_argument("--unit", help="tests/reference.py unit output")
    ap.add_argument("--e2e", help="tests/reference.py video output")
    ap.add_argument("--vae", help="wan_2.1_vae.safetensors (for --e2e)")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--out", help="dir for the e2e comparison image")
    args = ap.parse_args()
    global torch
    pack = setup(args.comfy)
    import torch
    ok = True
    if args.unit:
        ok &= unit(args, pack)
    if args.e2e:
        ok &= e2e(args, pack)
    print("PARITY OK" if ok else "PARITY FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
