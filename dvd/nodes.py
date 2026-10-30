"""DVD Video Depth: the node."""

from comfy.ldm.colormap import turbo
from comfy_api.latest import IO

from . import depth as D
from .pipeline import estimate


class DVDVideoDepth(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="DVDVideoDepth",
            display_name="DVD Video Depth",
            category="image/geometry estimation",
            search_aliases=["video depth", "deterministic video depth", "wan depth", "depth estimation video"],
            description="Temporally stable relative depth for a video with DVD (Deterministic Video Depth, EnVision "
                        "Research): Wan 2.1 1.3B retrained to regress depth in a single forward pass, no sampling. "
                        "Frames are processed in overlapping windows that are aligned (scale + shift) and cross-faded, "
                        "so clips of any length keep one consistent scale. Near is bright.",
            inputs=[
                IO.Model.Input("model", tooltip="Load Diffusion Model: dvd_1.1_bf16.safetensors (DVD is a Wan 2.1 T2V 1.3B model)."),
                IO.Vae.Input("vae", tooltip="Load VAE: wan_2.1_vae.safetensors."),
                IO.Image.Input("images", tooltip="The frames (Load Video → Get Video Components, or any image batch; a single image works too)."),
                IO.Int.Input("height", default=480, min=64, max=4096, step=16,
                             tooltip="Processing size: frames are scaled, aspect kept, to cover this box (rounded up to 16) and the depth is "
                                     "resized back. 480 x 640 is DVD's reference setting; larger is slower and off the training distribution."),
                IO.Int.Input("width", default=640, min=64, max=4096, step=16, tooltip="See height."),
                IO.Int.Input("window_size", default=81, min=5, max=401, step=4,
                             tooltip="Frames per forward pass (4n+1). 81 is the reference; smaller needs less VRAM and makes more windows."),
                IO.Int.Input("overlap", default=21, min=0, max=200,
                             tooltip="Frames shared by consecutive windows, where they are aligned and cross-faded. DVD's demo uses 21; "
                                     "more is steadier across windows and slower."),
                IO.Combo.Input("normalize", options=["video", "frame", "none"], default="video",
                               tooltip="video: min/max over the whole clip to 0..1 (DVD's visualisation, keeps the clip's single scale). "
                                       "frame: per frame. none: the decoded values as they are (relative, 0..1)."),
                IO.Combo.Input("colormap", options=["grayscale", "turbo"], default="grayscale"),
                IO.Boolean.Input("invert", default=False, tooltip="Flip near and far (DVD predicts near = bright)."),
                IO.Combo.Input("output_size", options=["input", "model"], default="input",
                               tooltip="input: depth at the frames' size. model: keep the processing size (far less memory on long clips)."),
                IO.Float.Input("denoise_step", default=0.5, min=0.0, max=1.0, step=0.01, advanced=True,
                               tooltip="The fixed timestep DVD regresses at (timestep = 1000 x this). The released weights are trained at 0.5."),
                IO.Boolean.Input("scale_only", default=False, advanced=True,
                                 tooltip="Align consecutive windows with a scale only, no shift."),
            ],
            outputs=[IO.Image.Output(display_name="depth")],
        )

    @classmethod
    def execute(cls, model, vae, images, height, width, window_size, overlap, normalize, colormap, invert, output_size,
                denoise_step, scale_only) -> IO.NodeOutput:
        depth = estimate(model, vae, images, height, width, window_size, overlap, denoise_step, scale_only,
                         resize_back=output_size == "input")
        depth = D.normalize(depth, normalize).clamp_(0.0, 1.0)
        if invert:
            depth = 1.0 - depth
        if colormap == "turbo":
            image = turbo(depth)
        else:
            image = depth.unsqueeze(-1).expand(*depth.shape, 3).contiguous()
        return IO.NodeOutput(image.float())


NODES = [DVDVideoDepth]
