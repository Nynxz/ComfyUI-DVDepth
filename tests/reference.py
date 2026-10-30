"""Reference DVD runner (ground truth for tests/parity.py): diffsynth's WanModel + peft LoRA (merged like test_single_video.py) + diffsynth's Wan VAE,
running the regression pass exactly as WanVideoPipeline.__call__ does in mode='regression' (context = zeros,
timestep = 1000 * denoise_step, output = depth latents). Helpers are copied verbatim from test_script/test_single_video.py.

  DVD_REPO=<clone of EnVision-Research/DVD> DVD_PYDEPS=<dir with peft+accelerate> python -I tests/reference.py unit --ckpt dvd_1.1.safetensors --out unit.npz            # DiT forward on a fixed random latent (fp32 + bf16)
  ... python -I tests/reference.py video --ckpt dvd_1.1.safetensors --video x.mp4 --frames 181 --out e2e.npz
"""
import argparse, importlib, os, sys, time, types
import numpy as np, torch, torch.nn.functional as F
from safetensors.torch import load_file

DVD = os.environ.get("DVD_REPO", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "references", "DVD"))
for extra in os.environ.get("DVD_PYDEPS", "").split(os.pathsep):
    if extra:
        sys.path.insert(0, extra)
for blocked in ("sageattention", "flash_attn", "flash_attn_interface"):  # force SDPA (exact) in the reference
    sys.modules[blocked] = None
pkg = types.ModuleType("dvdref"); pkg.__path__ = [f"{DVD}/diffsynth/models"]; sys.modules["dvdref"] = pkg
try:
    import diffusers.models.lora  # noqa
except Exception:
    m = types.ModuleType("diffusers.models.lora"); m.LoRALinearLayer = object; sys.modules["diffusers.models.lora"] = m
wan_dit = importlib.import_module("dvdref.wan_video_dit")
wan_vae = importlib.import_module("dvdref.wan_video_vae")

DIT_CONFIG = dict(has_image_input=False, patch_size=[1, 2, 2], in_dim=16, dim=1536, ffn_dim=8960, freq_dim=256,
                  text_dim=4096, out_dim=16, num_heads=12, num_layers=30, eps=1e-6)  # Wan2.1-T2V-1.3B (from_civitai)


def build_dit(ckpt, device, dtype, merged_sd=None):
    """merged_sd: a ComfyUI-layout merged state dict to load instead (to test the implementation with identical weights)."""
    from peft import LoraConfig, inject_adapter_in_model
    from peft.tuners.lora.layer import Linear as LoraLinear
    dit = wan_dit.WanModel(**DIT_CONFIG).to(torch.bfloat16)
    if merged_sd is None:
        dit = inject_adapter_in_model(LoraConfig(r=512, lora_alpha=512, target_modules="q,k,v,o,ffn.0,ffn.2".split(",")), dit, adapter_name="default")
        sd = load_file(ckpt, device="cpu")
        dit.load_state_dict({k[len("pipe.dit."):]: v for k, v in sd.items() if k.startswith("pipe.dit.")}, strict=True)
        n = 0
        for _, mod in dit.named_modules():
            if isinstance(mod, LoraLinear):
                mod.merge(); n += 1
        print(f"reference: merged {n} LoRA layers (peft, on cpu like test_single_video.py)")
    else:
        dit.load_state_dict({k: v.to(torch.bfloat16) for k, v in merged_sd.items()}, strict=True)
    return dit.to(device=device, dtype=dtype).eval().requires_grad_(False)


def build_vae(ckpt, device, dtype=torch.bfloat16):
    vae = wan_vae.WanVideoVAE()
    sd = load_file(ckpt, device="cpu")
    vae.load_state_dict({k[len("pipe.vae."):]: v for k, v in sd.items() if k.startswith("pipe.vae.")}, strict=True)
    return vae.to(device=device, dtype=dtype).eval().requires_grad_(False)


# ---- verbatim from test_script/test_single_video.py -------------------------------------------------------------
def compute_scale_and_shift(curr_frames, ref_frames, mask=None):
    if mask is None:
        mask = np.ones_like(ref_frames)
    a_00 = np.sum(mask * curr_frames * curr_frames); a_01 = np.sum(mask * curr_frames); a_11 = np.sum(mask)
    b_0 = np.sum(mask * curr_frames * ref_frames); b_1 = np.sum(mask * ref_frames)
    det = a_00 * a_11 - a_01 * a_01
    if det != 0:
        scale = (a_11 * b_0 - a_01 * b_1) / det; shift = (-a_01 * b_0 + a_00 * b_1) / det
    else:
        scale, shift = 1.0, 0.0
    return scale, shift

def read_video(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path); fps = cap.get(cv2.CAP_PROP_FPS); frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    video_np = np.stack(frames)
    return torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0, fps, video_np

def resize_for_training_scale(video_tensor, target_h=480, target_w=640):
    B, T, C, H, W = video_tensor.shape
    ratio = max(target_h / H, target_w / W)
    new_H = int(np.ceil(H * ratio)); new_W = int(np.ceil(W * ratio))
    new_H = (new_H + 15) // 16 * 16; new_W = (new_W + 15) // 16 * 16
    if new_H == H and new_W == W:
        return video_tensor, (H, W)
    resized = F.interpolate(video_tensor.view(B * T, C, H, W), size=(new_H, new_W), mode="bilinear", align_corners=False)
    return resized.view(B, T, C, new_H, new_W), (H, W)

def resize_depth_back(depth_np, orig_size):
    orig_H, orig_W = orig_size
    depth_tensor = torch.from_numpy(depth_np).permute(0, 3, 1, 2).float()
    depth_tensor = F.interpolate(depth_tensor, size=(orig_H, orig_W), mode='bilinear', align_corners=False)
    return depth_tensor.permute(0, 2, 3, 1).cpu().numpy()

def pad_time_mod4(video_tensor):
    B, T, C, H, W = video_tensor.shape
    remainder = T % 4
    if remainder != 1:
        pad_len = (4 - remainder + 1) % 4
        pad_frames = video_tensor[:, -1:, :, :, :].repeat(1, pad_len, 1, 1, 1)
        video_tensor = torch.cat([video_tensor, pad_frames], dim=1)
    return video_tensor, T

def get_window_index(T, window_size, overlap):
    if T <= window_size:
        return [(0, T)]
    res = [(0, window_size)]; start = window_size - overlap
    while start < T:
        end = start + window_size
        if end < T:
            res.append((start, end)); start += window_size - overlap
        else:
            start = max(0, T - window_size); res.append((start, T)); break
    return res

def generate_depth_sliced(model, input_rgb, window_size=45, overlap=9, scale_only=False):
    B, T, C, H, W = input_rgb.shape
    depth_windows = get_window_index(T, window_size, overlap)
    print(f"depth_windows {depth_windows}")
    depth_res_list = []
    for start, end in depth_windows:
        _input_rgb_slice = input_rgb[:, start:end]
        _input_rgb_slice, origin_T = pad_time_mod4(_input_rgb_slice)
        _input_frame = _input_rgb_slice.shape[1]
        _input_height, _input_width = _input_rgb_slice.shape[-2:]
        outputs = model.pipe(prompt=[""] * B, negative_prompt=[""] * B, mode=model.args.mode, height=_input_height, width=_input_width,
                             num_frames=_input_frame, batch_size=B, input_image=_input_rgb_slice[:, 0], extra_images=_input_rgb_slice,
                             extra_image_frame_index=torch.ones([B, _input_frame]), input_video=_input_rgb_slice, cfg_scale=1, seed=0,
                             tiled=False, denoise_step=model.args.denoise_step)
        depth_res_list.append(outputs['depth'][:, :origin_T])
    model.raw_windows = [d.copy() for d in depth_res_list]
    depth_list_aligned = None; prev_end = None
    for i, (t, (start, end)) in enumerate(zip(depth_res_list, depth_windows)):
        if i == 0:
            depth_list_aligned = t; prev_end = end; continue
        curr_start = start; real_overlap = prev_end - curr_start
        if real_overlap > 0:
            ref_frames = depth_list_aligned[:, -real_overlap:]; curr_frames = t[:, :real_overlap]
            if scale_only:
                scale = np.sum(curr_frames * ref_frames) / (np.sum(curr_frames * curr_frames) + 1e-6); shift = 0.0
            else:
                scale, shift = compute_scale_and_shift(curr_frames, ref_frames)
            scale = np.clip(scale, 0.7, 1.5)
            aligned_t = t * scale + shift
            aligned_t[aligned_t < 0] = 0
            print(f"[Overlap {i}] real_overlap = {real_overlap} scale = {scale:.8f}, shift = {shift:.8f}")
            alpha = np.linspace(0, 1, real_overlap, dtype=np.float32).reshape(1, real_overlap, 1, 1, 1)
            smooth_overlap = (1 - alpha) * ref_frames + alpha * aligned_t[:, :real_overlap]
            depth_list_aligned = np.concatenate([depth_list_aligned[:, :-real_overlap], smooth_overlap, aligned_t[:, real_overlap:]], axis=1)
        else:
            depth_list_aligned = np.concatenate([depth_list_aligned, t], axis=1)
        prev_end = end
    return depth_list_aligned[:, :T]
# ---------------------------------------------------------------------------------------------------------------------


class FakePipe:
    """What WanVideoPipeline.__call__ does for mode='regression' (units: InputVideoEmbedder -> vae.encode of the video;
    PromptEmbedder -> zeros(512, 4096); ImageEmbedder -> nothing, the T2V DiT has no image input; one model_fn call at
    timestep 1000 * denoise_step; scheduler.step returns the model output (target 'x'); vae.decode; (x + 1) / 2 clipped)."""
    def __init__(self, dit, vae, device, dtype=torch.bfloat16):
        self.dit, self.vae, self.device, self.dtype = dit, vae, device, dtype
        self.forward_seconds = []

    @torch.inference_mode()
    def __call__(self, input_video, denoise_step, **kw):
        video = (input_video.to(self.device, self.dtype) * 2 - 1).permute(0, 2, 1, 3, 4).contiguous()  # B C T H W in [-1, 1]
        latents = self.vae.encode(video, device=self.device).to(self.dtype)
        context = torch.zeros([latents.shape[0], 512, 4096], dtype=self.dtype, device=self.device)
        timestep = torch.tensor([1000 * denoise_step], dtype=self.dtype, device=self.device)
        torch.cuda.synchronize(); t0 = time.time()
        depth_latents = self.dit(latents.to(self.dit.patch_embedding.weight.dtype), timestep.to(self.dit.patch_embedding.weight.dtype), context.to(self.dit.patch_embedding.weight.dtype))
        torch.cuda.synchronize(); self.forward_seconds.append(time.time() - t0)
        depth = self.vae.decode(depth_latents.to(self.dtype), device=self.device)  # B 3 T H W, clamped to [-1, 1]
        depth = depth.permute(0, 2, 3, 4, 1).to("cpu", torch.float32).numpy()
        return {"depth": ((depth + 1.0) / 2.0).clip(0.0, 1.0)}


class FakeModel:
    def __init__(self, pipe):
        self.pipe = pipe
        self.args = types.SimpleNamespace(mode="regression", denoise_step=0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["unit", "video"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video"); ap.add_argument("--frames", type=int, default=181)
    ap.add_argument("--height", type=int, default=480); ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--window", type=int, default=81); ap.add_argument("--overlap", type=int, default=21)
    ap.add_argument("--merged", help="ComfyUI-layout merged file to load into the reference model instead of merging")
    args = ap.parse_args()
    dev = "cuda"
    if args.mode == "unit":
        torch.manual_seed(0)
        x = torch.randn(1, 16, 21, 60, 80)
        ctx = torch.zeros(1, 512, 4096); t = torch.tensor([500.0])
        out = {"x": x.numpy()}
        for label, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            dit = build_dit(args.ckpt, dev, dtype)
            with torch.inference_mode():
                torch.cuda.synchronize(); t0 = time.time()
                y = dit(x.to(dev, dtype), t.to(dev, dtype), ctx.to(dev, dtype))
                torch.cuda.synchronize()
            print(f"reference {label}: forward {time.time() - t0:.2f}s, out {tuple(y.shape)} mean {y.float().mean():.4f} std {y.float().std():.4f}")
            out[f"y_{label}"] = y.float().cpu().numpy()
            if label == "fp32" and args.merged:
                dit2 = build_dit(args.ckpt, dev, dtype, merged_sd=load_file(args.merged))
                with torch.inference_mode():
                    y2 = dit2(x.to(dev, dtype), t.to(dev, dtype), ctx.to(dev, dtype))
                out["y_fp32_convertedweights"] = y2.float().cpu().numpy()
                # merged weight comparison: reference (peft) vs converter
                ref_sd = {k.replace(".base_layer", ""): v for k, v in dit.state_dict().items() if "lora_" not in k}
                conv = load_file(args.merged)
                worst = max((ref_sd[k].float() - conv[k].float().to(dev)).abs().max().item() for k in conv)
                rel = max(((ref_sd[k].float() - conv[k].float().to(dev)).norm() / ref_sd[k].float().norm()).item() for k in conv if conv[k].ndim == 2)
                print(f"merged weights: peft(bf16 add) vs converter(fp32 add): max abs {worst:.3e}, worst rel fro {rel:.3e}")
                del dit2
            del dit; torch.cuda.empty_cache()
        np.savez(args.out, **out)
    else:
        frames, fps, frames_u8 = read_video(args.video)
        frames = frames[: args.frames].unsqueeze(0); frames_u8 = frames_u8[: args.frames]
        print("video", frames.shape, "fps", fps)
        small, orig = resize_for_training_scale(frames, args.height, args.width)
        print("resized", small.shape)
        pipe = FakePipe(build_dit(args.ckpt, dev, torch.bfloat16), build_vae(args.ckpt, dev), dev)
        model = FakeModel(pipe)
        t0 = time.time()
        depth = generate_depth_sliced(model, small, args.window, args.overlap)[0]
        total = time.time() - t0
        print(f"reference: {len(model.raw_windows)} windows in {total:.1f}s (DiT forwards {[round(s, 2) for s in pipe.forward_seconds]})")
        depth_full = resize_depth_back(depth, orig)
        print(f"depth range {depth_full.min():.4f} .. {depth_full.max():.4f}, shape {depth_full.shape}")
        np.savez(args.out, frames=frames_u8, depth_model_res=depth.astype(np.float32), depth=depth_full.astype(np.float32), fps=fps,
                 windows=np.array(get_window_index(frames.shape[1], args.window, args.overlap)),
                 **{f"raw_{i}": w[0].mean(-1).astype(np.float32) for i, w in enumerate(model.raw_windows)})


if __name__ == "__main__":
    main()
