"""ComfyUI-DVDepth: native ComfyUI support for DVD (EnVision-Research/DVD), deterministic video depth estimation.

DVD is Wan 2.1 T2V 1.3B with a merged LoRA, run as a regressor: clean video latents in at a fixed timestep,
depth latents out, one forward pass, an all-zero text context. ComfyUI implements Wan 2.1 natively, so the
converted weights (scripts/convert.py, or the files on nynxz/DVD) load with Load Diffusion Model, the stock Wan 2.1
VAE does the encoding and decoding, and the one node here, DVD Video Depth, runs the pass through ComfyUI's sampler
machinery with DVD's windowing and window alignment on top. No text encoder, no extra requirements.
"""

from comfy_api.latest import ComfyExtension
from typing_extensions import override

from .dvd.nodes import NODES


class DVDExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return NODES


async def comfy_entrypoint():
    return DVDExtension()
