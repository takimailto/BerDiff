import io
import argparse
import blobfile as bf

import torch as th

from diffusion import binomial_diffusion as bd
from diffusion.respace import SpacedDiffusion, space_timesteps
from model.unet import SegmentationModel


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    return dict(
        image_size=128,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.0,
        img_channels=1,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        ltype="bce",  # bce, kl, mix
        mean_type="xstart",
        rescale_timesteps=True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
    )


def create_model_and_diffusion(
    image_size,
    img_channels,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    ltype,
    mean_type,
    rescale_timesteps,
    use_checkpoint,
    use_scale_shift_norm,
):
    model = create_model(
        image_size,
        img_channels,
        num_channels,
        num_res_blocks,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
    )
    
    diffusion = create_binomial_diffusion(
        steps=diffusion_steps,
        noise_schedule=noise_schedule,
        ltype=ltype,
        mean_type=mean_type,
        rescale_timesteps=rescale_timesteps,
        timestep_respacing=timestep_respacing,
    )

    return model, diffusion


def create_model(
    image_size,
    img_channels,
    num_channels,
    num_res_blocks,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
):
    if image_size == 512:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    out_channels=in_channels=1
    
    
    model = SegmentationModel(
        in_channels=in_channels,
        img_channels=img_channels,
        model_channels=num_channels,
        out_channels=out_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=None,
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )

    return model

def create_binomial_diffusion(
    *,
    steps=1000,
    noise_schedule="linear",
    ltype="bce",
    mean_type="xstart",
    rescale_timesteps=False,
    timestep_respacing="",
):
    betas = bd.get_named_beta_schedule(noise_schedule, steps)
    if ltype == "rescale_kl":
        loss_type =bd.LossType.RESCALED_KL
    elif ltype == "kl":
        loss_type = bd.LossType.KL
    elif ltype == "bce":
        loss_type = bd.LossType.BCE
    elif ltype == "mix":
        loss_type = bd.LossType.MIX
    else:
        raise NotImplementedError(f"unknown LossType: {ltype}")
    if not timestep_respacing:
        timestep_respacing = [steps]
    if mean_type == "xstart":
        model_mean = bd.ModelMeanType.START_X
    elif mean_type == "epsilon":
        model_mean = bd.ModelMeanType.EPSILON
    elif mean_type == "previous":
        model_mean = bd.ModelMeanType.PREVIOUS_X
    else:
        raise NotImplementedError(f"unknown ModelMeanType: {mean_type}")

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=model_mean,
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
    

def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    mpigetrank=0
    if mpigetrank==0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
    return th.load(io.BytesIO(data), **kwargs)

def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")