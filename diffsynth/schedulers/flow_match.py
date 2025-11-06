import torch



class FlowMatchScheduler():

    def __init__(self, num_inference_steps=100, num_train_timesteps=1000, shift=3.0, sigma_max=1.0, sigma_min=0.003/1.002, inverse_timesteps=False, extra_one_step=False, reverse_sigmas=False):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)


    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, shift=None):
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing
            self.training = True
        else:
            self.training = False


    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        # Determine the device that should be used for the computation. In the
        # scheduler we expect ``sample`` and ``model_output`` to live on the
        # same device, but we defensively prioritise the tensor that provides a
        # valid device attribute.
        target_device = None
        if isinstance(sample, torch.Tensor) and sample.device.type != "meta":
            target_device = sample.device
        if (
            target_device is None
            and isinstance(model_output, torch.Tensor)
            and model_output.device.type != "meta"
        ):
            target_device = model_output.device

        # Pick a dtype for the arithmetic that is able to represent the model
        # output. Float8 tensors cannot be used directly in arithmetic with
        # higher precision dtypes, therefore we always upcast to the model
        # output dtype (or float32 as a safe fallback).
        float8_dtypes = {
            dtype
            for dtype in [
                getattr(torch, "float8_e4m3fn", None),
                getattr(torch, "float8_e5m2", None),
            ]
            if dtype is not None
        }
        if isinstance(model_output, torch.Tensor) and model_output.dtype not in float8_dtypes:
            computation_dtype = model_output.dtype
        else:
            computation_dtype = torch.float32

        if target_device is not None:
            sigmas = self.sigmas.to(device=target_device, dtype=computation_dtype)
        else:
            sigmas = self.sigmas.to(dtype=computation_dtype)
        sigma = sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = torch.tensor(
                1 if (self.inverse_timesteps or self.reverse_sigmas) else 0,
                device=target_device,
                dtype=computation_dtype,
            )
        else:
            sigma_ = sigmas[timestep_id + 1]

        if isinstance(model_output, torch.Tensor):
            model_output = model_output.to(device=target_device, dtype=computation_dtype)

        if isinstance(sample, torch.Tensor):
            sample_converted = sample.to(device=target_device, dtype=computation_dtype)
            prev_sample = sample_converted + model_output * (sigma_ - sigma)
            return prev_sample.to(sample.dtype)

        # Fallback for non tensor inputs (should not happen in practice).
        prev_sample = torch.as_tensor(sample, device=target_device, dtype=computation_dtype)
        prev_sample = prev_sample + model_output * (sigma_ - sigma)
        return prev_sample
    

    def return_to_timestep(self, timestep, sample, sample_stablized):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output
    
    
    def add_noise(self, original_samples, noise, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample
    

    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target
    

    def training_weight(self, timestep):
        timestep_id = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
        weights = self.linear_timesteps_weights[timestep_id]
        return weights
