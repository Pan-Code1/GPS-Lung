import os
import h5py
from flask import Flask, request

import torch
import torchvision
import numpy as np
import json

from tqdm import tqdm

import torchvision.transforms as transforms

from backbones.ncsnpp_generator_adagn import NCSNpp
from config import Config, Arguments
cfg = Config()
args = Arguments()

app = Flask(__name__)

def load_checkpoint(checkpoint_dir, netG, name_of_network, epoch,device = 'cuda:0'):
    checkpoint_file = checkpoint_dir.format(name_of_network, epoch)  

    checkpoint = torch.load(checkpoint_file, map_location=device)
    ckpt = checkpoint
   
    for key in list(ckpt.keys()):
         ckpt[key[7:]] = ckpt.pop(key)
    netG.load_state_dict(ckpt)
    netG.eval()

# load model
gen_diffusive_2 = NCSNpp(args)
checkpoint_file = cfg.model_path + "/{}_{}.pth"
epoch_chosen = args.which_epoch
load_checkpoint(checkpoint_file, gen_diffusive_2,'gen_diffusive_2',epoch=str(epoch_chosen), device = cfg.device)
gen_diffusive_2.to(cfg.device)

@app.route('/CBCT2CT', methods=['POST'])
def CBCT2CT():
    msg = request.data.decode("utf-8") 
    msg = json.loads(msg)
    print(msg)
    try:   
        generate(msg)
        return "generate done"
    except Exception as e:
        print(e)
        return "generate failed"

def load_data(temp_file):
    f = h5py.File(os.path.join(cfg.temp_path, temp_file), "r")
    dicom_src = f["dicom_src"][()]
    data_arr = f["dicom_data"][:]
    f.close()

    return dicom_src, data_arr


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out


def sample_posterior(coefficients, x_0,x_t, t):
    
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped
    
  
    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        
        noise = torch.randn_like(x_t)
        
        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:,None,None,None] * torch.exp(0.5 * log_var) * noise
            
    sample_x_pos = p_sample(x_0, x_t, t)
    
    return sample_x_pos


def sample_from_model(coefficients, generator, n_time, x_init, T, opt):
    x = x_init[:,[0],:]
    source = x_init[:,[1],:]
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
          
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)#.to(x.device)
            x_0 = generator(torch.cat((x,source),axis=1), t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0[:,[0],:], x, t)
            x = x_new.detach()
        
    return x


#%% Diffusion coefficients 
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var


def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)


def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3
   
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    
    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    
    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas**0.5
    a_s = torch.sqrt(1-betas)
    return sigmas, a_s, betas


class Posterior_Coefficients():
    def __init__(self, args, device):
        
        _, _, self.betas = get_sigma_schedule(args, device=device)
        
        #we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]
        
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
                                    (torch.tensor([1.], dtype=torch.float32,device=device), self.alphas_cumprod[:-1]), 0
                                        )               
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))


class ConditionalVerticalFlip(object):
    def __init__(self, flag):
        self.flag = flag

    def __call__(self, img):
        if self.flag:
            return transforms.functional.vflip(img)
        else:
            return img


def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small)  + eps_small
    return t.to(device)


def save_results(results, temp_file):
    with h5py.File(os.path.join(cfg.temp_path, temp_file), "a") as f:
        f.create_dataset('synthesis_data', data=results)
        f.close()    
    return 0


def generate(msg):
    temp_file = msg["temp_file"]
    flip = msg["flip"]
    dicom_src, data_arr = load_data(temp_file)
    # print(type(data_arr))
    print(f"data from {dicom_src}")
    print(f"data shape {data_arr.shape}")
    
    torch.manual_seed(42)
    
    T = get_time_schedule(args, cfg.device)
    pos_coeff = Posterior_Coefficients(args, cfg.device)

    to_range_0_1 = lambda x: (x + 1.) / 2.
    to_range_minus1_1 = lambda x: (x - 0.5) / 0.5
    # crop = transforms.CenterCrop((224, 224))

    preprocess = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        ConditionalVerticalFlip(flip)
    ])
    postprocess = transforms.Compose([
        transforms.Resize((cfg.input_size)),
        ConditionalVerticalFlip(flip)
    ])

    results = []
    with tqdm(total=data_arr.shape[0]) as pbar:
        for iteration, x in enumerate(data_arr): 
            x = x[None, None, :, :]
            x = preprocess(torch.Tensor(x))
            # print(x.shape)

            source_data = x.to(cfg.device, non_blocking=True)
            source_data = to_range_minus1_1(source_data)
            
            x2_t = torch.cat((torch.randn_like(source_data),source_data),axis=1)
            #diffusion steps
            with torch.no_grad():
                fake_sample2 = sample_from_model(pos_coeff, gen_diffusive_2, args.num_timesteps, x2_t, T, args)
        
            fake_sample2 = postprocess(to_range_0_1(fake_sample2)) 
            source_data = postprocess(to_range_0_1(source_data))
            if 0:
                fake_sample2 = torch.cat((source_data, fake_sample2),axis=-1)
                torchvision.utils.save_image(fake_sample2, '{}/{}_samples2_{}.jpg'.format("debug", 'CBCT', iteration), normalize=True)
            
            results.append(fake_sample2.cpu().detach().squeeze(0).numpy())

            pbar.update(1)
    
    results = np.concatenate(results, axis=0)
    save_results(results, temp_file)
    print("Generate done")

    return 0

if __name__ == '__main__':
    app.run(host=cfg.host, port=cfg.port)
