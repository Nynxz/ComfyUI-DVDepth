"""Convert a DVD checkpoint (FayeHongfeiZhang/DVD) into a ComfyUI Wan 2.1 diffusion model file.

DVD's checkpoint is the state of diffsynth's WanVideoPipeline: the Wan 2.1 T2V 1.3B DiT with an unmerged
rank-512 LoRA on every attention and FFN linear (pipe.dit.*, peft layout: <linear>.base_layer.weight,
<linear>.lora_A.default.weight, <linear>.lora_B.default.weight) plus a copy of the Wan 2.1 VAE (pipe.vae.*).
ComfyUI implements Wan 2.1 natively, so the conversion is: merge the LoRA into the base weights
(W + alpha/r * B @ A, in fp32, exactly what peft's merge() does), drop the diffsynth prefixes and save in
bf16. The result loads with Load Diffusion Model as a Wan 2.1 T2V model. The VAE is the stock Wan 2.1 VAE
(bit-identical to Comfy-Org's wan_2.1_vae.safetensors), so it is not written unless --vae-out is given.

    python scripts/convert.py --src FayeHongfeiZhang/DVD --file dvd_1.1.safetensors --out /path/to/ComfyUI/models
    python scripts/convert.py --src /path/to/dvd_1.1.safetensors --out /path/to/ComfyUI/models --dtype fp16

Writes <out>/diffusion_models/dvd_<version>_<dtype>.safetensors (version from the file name: model.safetensors
is DVD v1.0, dvd_1.1.safetensors is v1.1).
"""

import argparse
import json
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
DIT = "pipe.dit."
VAE = "pipe.vae.model."
# Wan 2.1 T2V 1.3B as ComfyUI detects it from the keys (comfy/model_detection.py); checked after the merge.
EXPECTED = {"dim": 1536, "ffn_dim": 8960, "num_layers": 30, "in_dim": 16, "out_dim": 16, "text_dim": 4096}


def version_from_name(path):
    name = os.path.basename(path)
    if name == "model.safetensors":
        return "1.0"
    m = re.search(r"(\d+\.\d+)", name)
    return m.group(1) if m else "x"


def merge_dit(f, scaling, dtype):
    """Merge the peft LoRA layers of the DiT and strip the diffsynth prefix."""
    keys = [k for k in f.keys() if k.startswith(DIT)]  # noqa: SIM118 (safe_open handle)
    out, merged, ranks = {}, 0, set()
    for k in keys:
        name = k[len(DIT):]
        if ".lora_A." in name or ".lora_B." in name:
            continue
        if name.endswith(".base_layer.weight"):
            base = name[: -len(".base_layer.weight")]
            a = f.get_tensor(f"{DIT}{base}.lora_A.default.weight").float()
            b = f.get_tensor(f"{DIT}{base}.lora_B.default.weight").float()
            w = f.get_tensor(k).float() + scaling(a.shape[0]) * (b @ a)
            out[f"{base}.weight"] = w.to(dtype)
            merged += 1
            ranks.add(a.shape[0])
        elif name.endswith(".base_layer.bias"):
            out[name.replace(".base_layer.bias", ".bias")] = f.get_tensor(k).to(dtype)
        else:
            out[name] = f.get_tensor(k).to(dtype)
    return out, merged, ranks


def check_wan21(sd):
    dim = sd["head.modulation"].shape[-1]
    got = {
        "dim": dim,
        "ffn_dim": sd["blocks.0.ffn.0.weight"].shape[0],
        "num_layers": len({k.split(".")[1] for k in sd if k.startswith("blocks.")}),
        "in_dim": sd["patch_embedding.weight"].shape[1],
        "out_dim": sd["head.head.weight"].shape[0] // 4,
        "text_dim": sd["text_embedding.0.weight"].shape[1],
    }
    if got != EXPECTED:
        raise SystemExit(f"unexpected architecture {got}, expected Wan 2.1 T2V 1.3B {EXPECTED}")
    for bad in ("img_emb.proj.0.bias", "vace_patch_embedding.weight", "control_adapter.conv.weight"):
        if bad in sd:
            raise SystemExit(f"{bad} present: ComfyUI would not detect this as a Wan 2.1 T2V model")
    leftovers = [k for k in sd if "lora_" in k or "base_layer" in k]
    if leftovers:
        raise SystemExit(f"unmerged LoRA keys left: {leftovers[:3]}")
    return got


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="DVD checkpoint .safetensors, or the HF repo (FayeHongfeiZhang/DVD)")
    ap.add_argument("--file", default="dvd_1.1.safetensors", help="file to fetch when --src is a repo")
    ap.add_argument("--out", required=True, help="ComfyUI models dir (diffusion_models/ and vae/ are created)")
    ap.add_argument("--dtype", default="bf16", choices=DTYPES, help="weights dtype (the originals are bf16)")
    ap.add_argument("--lora-alpha", type=float, default=None, help="peft lora_alpha; DVD trains with alpha = rank (scale 1)")
    ap.add_argument("--vae-out", action="store_true", help="also write the bundled Wan 2.1 VAE to <out>/vae")
    args = ap.parse_args()

    src = args.src
    if not os.path.exists(src):
        from huggingface_hub import hf_hub_download
        src = hf_hub_download(args.src, args.file)
    dtype = DTYPES[args.dtype]
    version = version_from_name(src)
    scaling = (lambda r: (args.lora_alpha / r)) if args.lora_alpha else (lambda r: 1.0)

    with safe_open(src, "pt", device="cpu") as f:
        dit, merged, ranks = merge_dit(f, scaling, dtype)
        arch = check_wan21(dit)
        print(f"DiT: {len(dit)} tensors, {merged} LoRA layers merged (rank {sorted(ranks)}), {arch}")
        os.makedirs(os.path.join(args.out, "diffusion_models"), exist_ok=True)
        path = os.path.join(args.out, "diffusion_models", f"dvd_{version}_{args.dtype}.safetensors")
        meta = {"format": "pt", "dvd_source": f"FayeHongfeiZhang/DVD {os.path.basename(src)}", "dvd_version": version,
                "dvd_lora_rank": ",".join(str(r) for r in sorted(ranks)), "dvd_converter": "ComfyUI-DVDepth scripts/convert.py",
                "modelspec.architecture": "wan2.1-t2v-1.3b", "modelspec.title": f"DVD {version} (Wan 2.1 T2V 1.3B, LoRA merged)"}
        save_file(dit, path, metadata=meta)
        print(f"wrote {path} ({os.path.getsize(path) / 1e9:.2f} GB)")

        if args.vae_out:
            vae = {k[len(VAE):]: f.get_tensor(k) for k in f.keys() if k.startswith(VAE)}  # noqa: SIM118
            os.makedirs(os.path.join(args.out, "vae"), exist_ok=True)
            vpath = os.path.join(args.out, "vae", "wan_2.1_vae.safetensors")
            save_file(vae, vpath, metadata={"format": "pt", "dvd_source": os.path.basename(src)})
            print(f"wrote {vpath} ({len(vae)} tensors): this is the stock Wan 2.1 VAE")
    print(json.dumps({"diffusion_model": path, "version": version, "dtype": args.dtype}))


if __name__ == "__main__":
    main()
