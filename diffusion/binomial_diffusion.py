import enum
import math

import numpy as np
import torch as th
from torch.distributions.binomial import Binomial

from model.basic_module import mean_flat
from loss.losses import binomial_kl, binomial_log_likelihood


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine2":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2),
        )
    elif schedule_name == "alpha_bar_linear":
        beta = []
        for i in range(num_diffusion_timesteps):
            t = i + 1
            beta.append(1/(num_diffusion_timesteps - t + 1))
        return np.array(beta)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class LossType(enum.Enum):
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB
    BCE = enum.auto()  # use raw BCE loss
    MIX = enum.auto()  # combine BCE loss and kl loss

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class BinomialDiffusion:
    """
    Utilities for training and sampling diffusion models.

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at 1 and going to T.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        self.alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
    
    def q_mean(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: Binomial distribution parameters, of x_start's shape.
        """
        mean = _extract_into_tensor(self.alphas_cumprod, t, x_start.shape) * x_start 
        + (1 - _extract_into_tensor(self.alphas_cumprod, t, x_start.shape)) / 2
        
        return mean
    
    def q_sample(self, x_start, t):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A noisy version of x_start.
        """

        mean = self.q_mean(x_start, t)
        return Binomial(1, mean).sample()
    
    def q_posterior_mean(self, x_start, x_t, t):
        """
        Get the distribution q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape

        theta_1 = (_extract_into_tensor(self.alphas, t, x_start.shape) * (1-x_t) + (1 - _extract_into_tensor(self.alphas, t, x_start.shape)) / 2) * (_extract_into_tensor(self.alphas_cumprod, t-1, x_start.shape) * (1-x_start) + (1 - _extract_into_tensor(self.alphas_cumprod, t-1, x_start.shape)) / 2)
        theta_2 = (_extract_into_tensor(self.alphas, t, x_start.shape) * x_t + (1 - _extract_into_tensor(self.alphas, t, x_start.shape)) / 2) * (_extract_into_tensor(self.alphas_cumprod, t-1, x_start.shape) * x_start + (1 - _extract_into_tensor(self.alphas_cumprod, t-1, x_start.shape)) / 2)

        posterior_mean = theta_2 / (theta_1 + theta_2)

        return posterior_mean
    
    def p_mean(
        self, model, x, t, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            return x
        
        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output

        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean = self.q_posterior_mean(
                x_start=pred_xstart, x_t=x, t=t
            )
            model_mean = th.where((t == 0)[:,None, None, None], pred_xstart, model_mean)
        else:
            raise NotImplementedError(self.model_mean_type)
        return {
            "mean": model_mean,
            "pred_xstart": pred_xstart,
        }
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            th.abs(x_t - eps).to(device=t.device).float()
        )
    
    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        A = (_extract_into_tensor(self.alphas, t, x_t.shape) * (1-x_t) + (1 - _extract_into_tensor(self.alphas, t, x_t.shape)) / 2)
        B = (_extract_into_tensor(self.alphas, t, x_t.shape) * x_t + (1 - _extract_into_tensor(self.alphas, t, x_t.shape)) / 2)
        C = (1 - _extract_into_tensor(self.alphas_cumprod, t-1, x_t.shape)) / 2
        numerator = A * C * xprev + B * C * (xprev -  1) + A * xprev * _extract_into_tensor(self.alphas_cumprod, t-1, x_t.shape)
        denominator = (B  + A  * xprev - B * xprev) * _extract_into_tensor(self.alphas_cumprod, t-1, x_t.shape)
        return (numerator / denominator)
    
    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t
    
    def p_sample(
        self, model, x, t, denoised_fn=None, model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean(
            model,
            x,
            t,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        sample = Binomial(1, out["mean"]).sample()
        if t[0] != 0:
            return {"sample": sample, "pred_xstart": out["pred_xstart"]}
        else:
            # return {"sample": sample, "pred_xstart": out["pred_xstart"]}
            return {"sample": out["mean"], "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = Binomial(1, th.ones(*shape)/2).sample().to(device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]
    
    def ddim_sample(
        self,
        model,
        x,
        t,
        denoised_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean(
            model,
            x,
            t,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if t[0] != 0:
            alpha_bar_t_1 = _extract_into_tensor(self.alphas_cumprod, t-1, x.shape)
            alpha_bar_t = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
            sigma = (1 - alpha_bar_t_1) / (1 - alpha_bar_t)
            mean = sigma * x + (alpha_bar_t_1 - sigma * alpha_bar_t) * out["pred_xstart"]
            # sample = Binomial(1, th.clip(mean, min=0, max=1)).sample()
            sample = Binomial(1, mean).sample()
            return {"sample": sample, "pred_xstart": out["pred_xstart"]}
        else:
            return {"sample": out["mean"], "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]
    
    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = Binomial(1, th.ones(*shape)/2).sample().to(device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean = self.q_posterior_mean(x_start=x_start, x_t=x_t, t=t)
        out = self.p_mean(
            model, x_t, t, model_kwargs=model_kwargs
        )
        kl = binomial_kl(true_mean, out["mean"])

        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -binomial_log_likelihood(x_start, means=out["mean"])
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}
    
    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean = self.q_mean(x_start, t)
        kl_prior = binomial_kl(
            mean1=qt_mean, mean2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)
    
    def training_losses(self, model, x_start, t, model_kwargs=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        x_t = self.q_sample(x_start, t)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL or self.loss_type == LossType.MIX:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
            if self.loss_type == LossType.MIX:
                target = {
                    ModelMeanType.PREVIOUS_X: self.q_posterior_mean(
                        x_start=x_start, x_t=x_t, t=t
                    ),
                    ModelMeanType.START_X: x_start,
                    ModelMeanType.EPSILON: self._predict_xstart_from_eps(x_t=x_t, t=t, eps=x_start),
                }[self.model_mean_type]
                model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
                terms["bce"] = mean_flat(-binomial_log_likelihood(target, means=model_output)) / np.log(2.0)
                terms["vb"] = terms["loss"]
                terms["loss"] = terms["vb"] + terms["bce"]
        elif self.loss_type == LossType.BCE:
            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean(
                    x_start=x_start, x_t=x_t, t=t
                ),
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: self._predict_xstart_from_eps(x_t=x_t, t=t, eps=x_start),
            }[self.model_mean_type]
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
            terms["loss"] = mean_flat(-binomial_log_likelihood(target, means=model_output)) / np.log(2.0)
        else:
            raise NotImplementedError(self.loss_type)

        return terms

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
