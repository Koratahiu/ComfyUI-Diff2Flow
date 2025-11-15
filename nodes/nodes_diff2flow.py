import torch
import numpy as np
from types import MethodType
from functools import partial
from typing import Any

try:
    import torchdiffeq
except ImportError:
    print("Warning: torchdiffeq is not installed. The advanced ODE solvers will not be available.")
    print("Please install it by running: pip install torchdiffeq")
    torchdiffeq = None

import comfy.model_sampling
import comfy.samplers
import comfy.sample
from comfy.ldm.modules.diffusionmodules.util import make_beta_schedule
from tqdm.auto import trange, tqdm
import latent_preview

# Diff2Flow Core Logic

def _d2f_extract_into_tensor(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple[int, ...]) -> torch.Tensor:
    b, *_ = t.shape
    t = t.to(a.device)
    t = t.clamp(0, a.shape[-1] - 1)
    left_idx = t.long()
    right_idx = (left_idx + 1).clamp(max=a.shape[-1] - 1)
    left_val = a.gather(-1, left_idx)
    right_val = a.gather(-1, right_idx)
    t_ = t - left_idx.float()
    out = left_val * (1 - t_) + right_val * t_
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def _d2f_convert_fm_t_to_dm_t(self, t: torch.Tensor) -> torch.Tensor:
    rectified_alphas_cumprod_full = self.df_rectified_alphas_cumprod_full.clone().to(t.device)
    rectified_alphas_cumprod_full = torch.flip(rectified_alphas_cumprod_full, [0])
    max_idx = rectified_alphas_cumprod_full.shape[0] - 1
    
    # Find the indices for interpolation, and clamp them to be within the valid range.
    right_index = torch.searchsorted(rectified_alphas_cumprod_full, t, right=True).clamp(max=max_idx)
    left_index = (right_index - 1).clamp(min=0)

    right_value = rectified_alphas_cumprod_full[right_index]
    left_value = rectified_alphas_cumprod_full[left_index]
    
    # Prevent division by zero if left and right values are the same.
    denom = right_value - left_value
    interp_ratio = torch.where(denom > 0, (t - left_value) / denom, torch.zeros_like(t))

    dm_t = left_index.float() + interp_ratio
    dm_t = self.model_sampling.num_timesteps - dm_t
    return dm_t

def _d2f_convert_fm_xt_to_dm_xt(self, fm_xt: torch.Tensor, fm_t: torch.Tensor) -> torch.Tensor:
    scale = self.df_sqrt_alphas_cumprod_full + self.df_sqrt_one_minus_alphas_cumprod_full
    dm_t = self._df_convert_fm_t_to_dm_t(fm_t)
    dm_t_left_index = torch.floor(dm_t)
    dm_t_right_index = torch.ceil(dm_t)
    max_idx = scale.shape[-1] - 1
    dm_t_left_index = dm_t_left_index.clamp(0, max_idx).long()
    dm_t_right_index = dm_t_right_index.clamp(0, max_idx).long()
    dm_t_left_value = scale[dm_t_left_index]
    dm_t_right_value = scale[dm_t_right_index]
    scale_t = dm_t_left_value + (dm_t - dm_t_left_index.float()) * (dm_t_right_value - dm_t_left_value)
    scale_t = scale_t.view(-1, 1, 1, 1)
    dm_xt = fm_xt * scale_t
    return dm_xt

def _d2f_predict_start_from_z_and_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return (
        self._df_extract_into_tensor(self.df_sqrt_alphas_cumprod, t, x_t.shape) * x_t -
        self._df_extract_into_tensor(self.df_sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
    )

def _d2f_predict_eps_from_z_and_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return (
        self._df_extract_into_tensor(self.df_sqrt_alphas_cumprod, t, x_t.shape) * v +
        self._df_extract_into_tensor(self.df_sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t
    )

def _d2f_predict_start_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    return (
        self._df_extract_into_tensor(self.df_sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
        self._df_extract_into_tensor(self.df_sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
    )

def _d2f_get_vector_field_from_v(self, v: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    z_pred = self._df_predict_start_from_z_and_v(x_t, t, v)
    eps_pred = self._df_predict_eps_from_z_and_v(x_t, t, v)
    vector_field = z_pred - eps_pred
    return vector_field

def _d2f_get_vector_field_from_eps(self, noise: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    z_pred = self._df_predict_start_from_eps(x_t, t, noise)
    eps_pred = noise
    vector_field = z_pred - eps_pred
    return vector_field

@torch.no_grad()
def _d2f_get_diff2flow_velocity(self, fm_x: torch.Tensor, fm_t: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """
    The main inference/sampling function.
    Predicts the vector field (Diff2Flow velocity) using the underlying UNet model.
    """
    dm_t_continuous = self._df_convert_fm_t_to_dm_t(fm_t)
    dm_x = self._df_convert_fm_xt_to_dm_xt(fm_x, fm_t)

    model_device = self.device
    model_dtype = self.get_dtype()

    unet_kwargs = kwargs.copy()
    if 'c_crossattn' in unet_kwargs:
        unet_kwargs['context'] = unet_kwargs.pop('c_crossattn')

    # Cast inputs to the correct device and dtype for the model
    dm_x = dm_x.to(model_device, model_dtype)
    dm_t_continuous = dm_t_continuous.to(model_device) # Timesteps are usually float32
    for key, value in unet_kwargs.items():
        if hasattr(value, "to") and key not in ['transformer_options', 'control']:
            unet_kwargs[key] = value.to(model_device, model_dtype)

    model_pred = self.diffusion_model(
        dm_x,
        dm_t_continuous,
        **unet_kwargs
    )

    model_pred = model_pred.float()

    # Handle NaN prediction (as in original code)
    if torch.isnan(model_pred).any():
        model_pred[torch.isnan(model_pred)] = 0

    dm_x_float = dm_x.float()
    dm_t_float = dm_t_continuous.float()

    vector_field = None
    if isinstance(self.model_sampling, comfy.model_sampling.V_PREDICTION):
        vector_field = self._df_get_vector_field_from_v(model_pred, dm_x_float, dm_t_float)
    elif isinstance(self.model_sampling, comfy.model_sampling.EPS):
        vector_field = self._df_get_vector_field_from_eps(model_pred, dm_x_float, dm_t_float)
    else:
        raise NotImplementedError(f"Diff2Flow is not implemented for prediction type: {type(self.model_sampling)}")

    return vector_field

def enable_diff2flow(model):
    """
    Modifies the model in-place to enable training and inference
    with the Diff2Flow methodology.
    The logic based on the paper:
    "Diff2Flow: Training Flow Matching Models via Diffusion Model Alignment"
    Modified from:
    https://github.com/CompVis/diff2flow/blob/33239aa0c02c554ee0b3fff5c5f0167a8dabdf6a/diff2flow/flow_obj.py

    ADAPTED FOR COMFYUI.
    """
    if hasattr(model, 'get_diff2flow_velocity'):
        return  # Already enabled

    device = model.device

    sampling_settings = model.model_config.sampling_settings
    beta_schedule_name = sampling_settings.get("beta_schedule", "linear")
    if beta_schedule_name == "scaled_linear":
        beta_schedule_name = "linear"

    betas_tensor = make_beta_schedule(
        beta_schedule_name,
        model.model_sampling.num_timesteps,
        linear_start=model.model_sampling.linear_start,
        linear_end=model.model_sampling.linear_end
    )
    betas = betas_tensor.cpu().numpy()

    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_full = np.append(1., alphas_cumprod)

    to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

    # Register all constants as buffers instead of simple attributes
    model.register_buffer('df_betas', to_torch(betas))
    model.register_buffer('df_alphas_cumprod', to_torch(alphas_cumprod))
    model.register_buffer('df_alphas_cumprod_full', to_torch(alphas_cumprod_full))

    model.register_buffer('df_sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
    model.register_buffer('df_sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))

    model.register_buffer('df_sqrt_alphas_cumprod_full', to_torch(np.sqrt(alphas_cumprod_full)))
    model.register_buffer('df_sqrt_one_minus_alphas_cumprod_full', to_torch(np.sqrt(1. - alphas_cumprod_full)))

    model.register_buffer('df_sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
    model.register_buffer('df_sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

    # These are calculated from tensors that will now be buffers
    rectified_alphas = model.df_sqrt_alphas_cumprod_full / (model.df_sqrt_alphas_cumprod_full + model.df_sqrt_one_minus_alphas_cumprod_full)
    rectified_sqrt_alphas = model.df_sqrt_one_minus_alphas_cumprod_full / (model.df_sqrt_alphas_cumprod_full + model.df_sqrt_one_minus_alphas_cumprod_full)

    model.register_buffer('df_rectified_alphas_cumprod_full', rectified_alphas)
    model._df_extract_into_tensor = MethodType(_d2f_extract_into_tensor, model)
    model._df_convert_fm_t_to_dm_t = MethodType(_d2f_convert_fm_t_to_dm_t, model)
    model._df_convert_fm_xt_to_dm_xt = MethodType(_d2f_convert_fm_xt_to_dm_xt, model)
    model._df_predict_start_from_z_and_v = MethodType(_d2f_predict_start_from_z_and_v, model)
    model._df_predict_eps_from_z_and_v = MethodType(_d2f_predict_eps_from_z_and_v, model)
    model._df_predict_start_from_eps = MethodType(_d2f_predict_start_from_eps, model)
    model._df_get_vector_field_from_v = MethodType(_d2f_get_vector_field_from_v, model)
    model._df_get_vector_field_from_eps = MethodType(_d2f_get_vector_field_from_eps, model)
    model.get_diff2flow_velocity = MethodType(_d2f_get_diff2flow_velocity, model)

# ODE Solver Logic

class ODEFunction:
    def __init__(self, model, t_min, t_max, n_steps, is_adaptive, extra_args=None, callback=None):
        self.model = model
        self.extra_args = {} if extra_args is None else extra_args
        self.callback = callback
        self.t_min = t_min.item()
        self.t_max = t_max.item()
        self.n_steps = n_steps
        self.is_adaptive = is_adaptive
        self.step = 0
        self.last_denoised = None
        self.is_diff2flow = hasattr(self.model.inner_model, 'get_diff2flow_velocity')
        if is_adaptive: self.pbar = tqdm(total=100, desc="solve", unit="%", leave=False, position=1)
        else: self.pbar = tqdm(total=n_steps, desc="solve", leave=False, position=1)

    def __call__(self, t, y):
        # Add batch dimension for the model
        y_in = y.unsqueeze(0)

        # prepare timestep for the model
        timestep_in = t.unsqueeze(0) if t.ndim == 0 else t

        if self.is_diff2flow:
            base_model = self.model.inner_model
            cond_scale = self.model.cfg
            positive_conds = self.model.conds.get("positive")
            negative_conds = self.model.conds.get("negative")
            model_options = self.extra_args.get('model_options', {})

            original_apply_model = base_model.apply_model
            try:
                t_in = timestep_in.repeat(y_in.shape[0])

                def diff2flow_apply_model(x_in, sigma_in, **c):
                    return base_model.get_diff2flow_velocity(fm_x=x_in, fm_t=t_in, **c)

                base_model.apply_model = diff2flow_apply_model
                
                dm_t_continuous = base_model._df_convert_fm_t_to_dm_t(t_in)
                current_sigma_for_conds = base_model.model_sampling.sigma(dm_t_continuous)

                velocity_cond, velocity_uncond = comfy.samplers.calc_cond_batch(base_model, [positive_conds, negative_conds], y_in, current_sigma_for_conds, model_options)
                
                velocity = velocity_uncond + cond_scale * (velocity_cond - velocity_uncond)
                self.last_denoised = y_in 
                return velocity.squeeze(0)
            finally:
                base_model.apply_model = original_apply_model
        else:
            if t <= 1e-5: 
                return torch.zeros_like(y)
            denoised = self.model(y_in, timestep_in, **self.extra_args)
            self.last_denoised = denoised
            # Calculate derivative
            derivative = (y_in - denoised) / t
            # Remove batch dimension for torchdiffeq
            return derivative.squeeze(0)

    def _callback(self, t0, y0, step):
        if self.callback is not None:
            sigma_for_callback = t0
            denoised_for_preview = self.last_denoised if self.last_denoised is not None else y0.unsqueeze(0)
            
            if self.is_diff2flow:
                base_model = self.model.inner_model
                dm_t = base_model._df_convert_fm_t_to_dm_t(t0)
                sigma_for_callback = base_model.model_sampling.sigma(dm_t)
                denoised_for_preview = y0.unsqueeze(0)

            self.callback({
                "x": y0.unsqueeze(0),
                "i": step,
                "sigma": sigma_for_callback,
                "sigma_hat": sigma_for_callback,
                "denoised": denoised_for_preview
            })

    def callback_step(self, t0, y0, dt):
        if self.is_adaptive: return
        self._callback(t0, y0, self.step)
        self.pbar.update(1)
        self.step += 1

    def callback_accept_step(self, t0, y0, dt):
        if not self.is_adaptive: return
        progress = (self.t_max - t0.item()) / (self.t_max - self.t_min)
        self._callback(t0, y0, round((self.n_steps - 1) * progress))
        new_step = round(100 * progress)
        self.pbar.update(new_step - self.step)
        self.step = new_step

    def reset(self):
        self.step = 0
        self.pbar.reset()
        self.last_denoised = None

class ODESampler:
    def __init__(self, solver, rtol, atol, max_steps):
        if torchdiffeq is None:
            raise ImportError("torchdiffeq is not installed. Please install it to use ODE solvers.")
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps
        self.ADAPTIVE_SOLVERS = { "dopri8", "dopri5", "bosh3", "fehlberg2", "adaptive_heun" }

    @torch.no_grad()
    def __call__(self, model, x, sigmas, extra_args=None, callback=None, disable=None):
        cfg_guider_model = model.inner_model
        is_diff2flow = hasattr(cfg_guider_model.inner_model, 'get_diff2flow_velocity')
        is_adaptive = self.solver in self.ADAPTIVE_SOLVERS

        if is_diff2flow:
            # Un-scale the initial noise.
            # Flow Matching's ODE must start from unscaled N(0,I) noise, but KSampler prepares x as scaled noise.
            initial_sigma = sigmas[0]
            if initial_sigma > 0:
                x = x / initial_sigma

            t_start = torch.tensor(0.0, device=x.device)
            t_end = torch.tensor(1.0, device=x.device)

            if is_adaptive:
                t = torch.stack([t_start, t_end])
            else:
                t = torch.linspace(t_start.item(), t_end.item(), len(sigmas), device=x.device)
        else:
            t_max = sigmas[0]
            t_min = sigmas[-1]
            t = torch.stack([t_max, t_min]) if is_adaptive else sigmas.to(x.device)

        # The ODE function needs a model that returns denoised x0. The CFG guider does exactly that.
        # We must clean up extra_args to remove arguments not expected by the CFGGuider's predict function.
        ode_extra_args = extra_args.copy()
        ode_extra_args.pop('denoise_mask', None)
        ode = ODEFunction(cfg_guider_model, t.min(), t.max(), len(sigmas), is_adaptive, ode_extra_args, callback)

        samples = torch.empty_like(x)
        for i in trange(x.shape[0], desc=self.solver, disable=disable):
            ode.reset()
            samples[i] = torchdiffeq.odeint(ode, x[i], t, rtol=self.rtol, atol=self.atol, method=self.solver, options={"max_num_steps": self.max_steps})[-1]

        return samples

# Node Definition

class Diff2FlowODESampler:
    # Build solver list dynamically
    if torchdiffeq is not None:
        ADAPTIVE_SOLVERS = { "dopri8", "dopri5", "bosh3", "fehlberg2", "adaptive_heun" }
        FIXED_SOLVERS = { "euler", "midpoint", "rk4", "heun3", "explicit_adams", "implicit_adams" }
        ALL_ODE_SOLVERS = sorted(list(ADAPTIVE_SOLVERS | FIXED_SOLVERS))
        SOLVERS = ALL_ODE_SOLVERS
    else:
        raise ImportError("torchdiffeq is not installed. Please install it to use this node.")

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "model": ("MODEL",),
                "solver": (s.SOLVERS, {"default": "euler"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 5, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "latent_image": ("LATENT", ),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default": "simple"}),
            }
        }
        # The optional inputs for advanced ODE solvers
        if torchdiffeq is not None:
            inputs["optional"] = {
                "log_relative_tolerance": ("FLOAT", { "min": -7, "max": 0, "default": -2.5, "step": 0.1 }),
                "log_absolute_tolerance": ("FLOAT", { "min": -7, "max": 0, "default": -3.5, "step": 0.1 }),
                "max_steps": ("INT", { "min": 1, "max": 500, "default": 50, "step": 1 }),
            }
        return inputs

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/custom_sampling"

    def sample(self, model, solver, steps, cfg, denoise, seed, positive, negative, latent_image, scheduler, disable_pbar=False, **kwargs):
        if torchdiffeq is None:
            raise ImportError("torchdiffeq is not installed. Please install it to use this solver.")

        # Clone model and enable Diff2Flow
        model_patched = model.clone()
        enable_diff2flow(model_patched.model)

        log_relative_tolerance = kwargs.get("log_relative_tolerance", -2.5)
        log_absolute_tolerance = kwargs.get("log_absolute_tolerance", -3.5)
        max_steps_ode = kwargs.get("max_steps", 50)
        
        rtol = 10 ** log_relative_tolerance
        atol = 10 ** log_absolute_tolerance
        sampler_function = ODESampler(solver, rtol, atol, max_steps_ode)

        sampler = comfy.samplers.KSAMPLER(sampler_function)

        # Prepare sigmas based on steps and denoise value
        # This standardizes the process for all samplers
        model_sampling = model_patched.model.model_sampling

        if denoise == 1.0:
            sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps)
        else:
            raise NotImplementedError ("denoise < 1.0 is not implemented yet")
            total_steps = int(steps / denoise) if denoise > 0 else steps
            sigmas_full = comfy.samplers.calculate_sigmas(model_sampling, scheduler, total_steps)
            sigmas = sigmas_full[-(steps + 1):]

        # Prepare noise and call the main sample function
        noise = comfy.sample.prepare_noise(latent_image['samples'], seed)

        # Create the callback function that the UI uses for previews
        callback = latent_preview.prepare_callback(model, steps)

        samples = comfy.samplers.sample(model_patched, noise, positive, negative, cfg,
                                        device=model_patched.load_device,
                                        sampler=sampler,
                                        sigmas=sigmas.to(model_patched.load_device),
                                        model_options=model_patched.model_options,
                                        latent_image=latent_image['samples'],
                                        denoise_mask=None, # ODE samplers don't support masks well
                                        callback=callback,
                                        disable_pbar=disable_pbar,
                                        seed=seed)

        # Format and return the output
        out_latent = latent_image.copy()
        out_latent["samples"] = samples
        return (out_latent,)


NODE_CLASS_MAPPINGS = {
    "Diff2FlowODESampler": Diff2FlowODESampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Diff2FlowODESampler": "Diff2Flow ODE KSampler",
}