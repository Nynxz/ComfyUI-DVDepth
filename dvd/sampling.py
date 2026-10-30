"""DVD's deterministic pass, run through ComfyUI's sampler machinery.

DVD (Deterministic Video Depth) is Wan 2.1 T2V 1.3B retrained as a regressor: the DiT gets the clean video latent
at one fixed timestep (1000 x denoise_step, 500 for the released weights) with an all-zero text context, and its
output *is* the depth latent. No noise, no ODE, one forward pass. ComfyUI's model_sampling classes describe how a
model's output relates to its input; the one here says "nothing is scaled, the output is the answer", and a
one-call sampler feeds it through CFGGuider so model management, offloading, LoRA/weight patches, attention
backends and torch.compile all apply as for any other Wan model.
"""

import comfy.model_sampling
import comfy.sample
import comfy.samplers
import torch

CONTEXT_TOKENS = 512  # DVD's pipeline hands the DiT torch.zeros(512, 4096) in place of a text embedding
CONTEXT_DIM = 4096


class Regression:
    def calculate_input(self, sigma, noise):
        return noise

    def calculate_denoised(self, sigma, model_output, model_input):
        return model_output

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        return latent_image

    def inverse_noise_scaling(self, sigma, latent):
        return latent


class ModelSamplingRegression(comfy.model_sampling.ModelSamplingDiscreteFlow, Regression):
    """timestep(sigma) = sigma * 1000, so a sigma of denoise_step is DVD's training timestep."""


def regression_model(model):
    """A clone of the Wan model whose sampling is DVD's regression (the loaded file is detected as a plain Wan 2.1 T2V)."""
    m = model.clone()
    sampling = ModelSamplingRegression(model.model.model_config)
    sampling.set_parameters(shift=1.0, multiplier=1000)
    m.add_object_patch("model_sampling", sampling)
    return m


def empty_context(batch_size=1):
    """The CONDITIONING DVD was trained with: zeros, no text encoder needed."""
    return [[torch.zeros(batch_size, CONTEXT_TOKENS, CONTEXT_DIM), {}]]


def _one_pass(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """A sampler with a single model call whose denoised prediction is the result."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    denoised = model(x, sigmas[0] * s_in, **extra_args)
    if callback is not None:
        callback({"x": x, "i": 0, "sigma": sigmas[0], "sigma_hat": sigmas[0], "denoised": denoised})
    return denoised


def predict(model, latent, conditioning=None, denoise_step=0.5, seed=0):
    """Video latents (as VAE Encode makes them, B x 16 x T x H x W) -> depth latents, same layout, for VAE Decode.

    `model` must come from regression_model(). The latent normalisation (Wan's per-channel mean/std) is applied by
    the sampler on the way in and undone on the way out, like for any sampled latent.
    """
    if conditioning is None:
        conditioning = empty_context(latent.shape[0])
    sigmas = torch.tensor([float(denoise_step), 0.0])
    sampler = comfy.samplers.KSAMPLER(_one_pass)
    noise = torch.zeros_like(latent)  # ignored by Regression.noise_scaling; shapes the memory estimate
    return comfy.sample.sample_custom(model, noise, 1.0, sampler, sigmas, conditioning, conditioning, latent,
                                      disable_pbar=True, seed=seed)
