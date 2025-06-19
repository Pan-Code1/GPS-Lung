

import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.functional import gaussian_blur


_MODELS = {}


def register_model(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _MODELS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _MODELS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_model(name):
  return _MODELS[name]


def get_sigmas(config):
  """Get sigmas --- the set of noise levels for SMLD from config files.
  Args:
    config: A ConfigDict object parsed from the config file
  Returns:
    sigmas: a jax numpy arrary of noise levels
  """
  sigmas = np.exp(
    np.linspace(np.log(config.model.sigma_max), np.log(config.model.sigma_min), config.model.num_scales))

  return sigmas


def get_ddpm_params(config):
  """Get betas and alphas --- parameters used in the original DDPM paper."""
  num_diffusion_timesteps = 1000
  # parameters need to be adapted if number of time steps differs from 1000
  beta_start = config.model.beta_min / config.model.num_scales
  beta_end = config.model.beta_max / config.model.num_scales
  betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

  alphas = 1. - betas
  alphas_cumprod = np.cumprod(alphas, axis=0)
  sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
  sqrt_1m_alphas_cumprod = np.sqrt(1. - alphas_cumprod)

  return {
    'betas': betas,
    'alphas': alphas,
    'alphas_cumprod': alphas_cumprod,
    'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
    'sqrt_1m_alphas_cumprod': sqrt_1m_alphas_cumprod,
    'beta_min': beta_start * (num_diffusion_timesteps - 1),
    'beta_max': beta_end * (num_diffusion_timesteps - 1),
    'num_diffusion_timesteps': num_diffusion_timesteps
  }


def create_model(config):
  """Create the score model."""
  model_name = config.model.name
  score_model = get_model(model_name)(config)
  score_model = score_model.to(config.device)
  score_model = torch.nn.DataParallel(score_model)
  return score_model


def get_model_fn(model, train=False):
  """Create a function to give the output of the score-based model.

  Args:
    model: The score model.
    train: `True` for training and `False` for evaluation.

  Returns:
    A model function.
  """

  def model_fn(x, labels):
    """Compute the output of the score-based model.

    Args:
      x: A mini-batch of input data.
      labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
        for different models.

    Returns:
      A tuple of (model output, new mutable states)
    """
    if not train:
      model.eval()
      return model(x, labels)
    else:
      model.train()
      return model(x, labels)

  return model_fn




def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))


def sobel_filter(t, eta=1):
  channels = t.shape[1]
  device = t.device
  t = gaussian_blur(t, kernel_size=5, sigma=1)
  sobel_x = torch.tensor([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]], dtype=torch.float, requires_grad=False, device=t.device).view(1,1,3,3)
  sobel_y = torch.tensor([[-1, -2, -1],
                      [0, 0, 0],
                      [1, 2, 1]], dtype=torch.float, requires_grad=False, device=t.device).view(1,1,3,3)
  
  gx = F.conv2d(t, sobel_x, stride=1, padding=1)
  gy = F.conv2d(t, sobel_y, stride=1, padding=1)

  i_mag = torch.sqrt((torch.pow(gx, 2) + torch.pow(gy, 2)))
  i_mag[i_mag < eta] = 0
  i_mag = i_mag / torch.max_pool2d(i_mag, t.shape[-1])

  i_mag += torch.randn_like(t)*t.std()+t.mean()

  # i_mag = gaussian_blur(i_mag, kernel_size=7, sigma=5)
  # i_mag[i_mag >0] = 1

  # hpf = t * i_mag

  return i_mag