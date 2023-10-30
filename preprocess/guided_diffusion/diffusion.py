import os
import logging
import time
import glob
import sys

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from data import get_dataset, data_transform, inverse_data_transform
from preprocess.functions.ckpt_util import get_ckpt_path, download
from preprocess.functions.svd_ddnm import ddnm_diffusion, ddnm_plus_diffusion

import torchvision.utils as tvu

from preprocess.guided_diffusion.models import Model
from preprocess.guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict
import random

from scipy.linalg import orth

def get_gaussian_noisy_img(img, noise_level, device):
    return img + torch.randn_like(img).to(device) * noise_level

def MeanUpsample(x, scale):
    n, c, h, w = x.shape
    out = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n,c,h,1,w,1)
    out = out.view(n, c, scale*h, scale*w)
    return out

def color2gray(x):
    coef=1/3
    x = x[:,0,:,:] * coef + x[:,1,:,:]*coef +  x[:,2,:,:]*coef
    return x.unsqueeze(1)

def gray2color(x):
    x = x[:,0,:,:]
    coef=1/3
    base = coef**2 + coef**2 + coef**2
    return torch.stack((x*coef/base, x*coef/base, x*coef/base), 1)   


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, type, device=None):
        self.args = args
        self.config = config
        self.type = type
        if device is None:
            device = (
                torch.device(f"{args.gpu}")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device
        print(f"The diffusion model is running on device:{device}")

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def _sample(self, simplified):
        cls_fn = None
        if self.config.model.type == 'simple':
            model = Model(self.config)

            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            elif self.config.data.dataset == 'CelebA_HQ':
                name = 'celeba_hq'
            elif self.config.data.dataset == 'Attack':
                name = 'celeba_hq'
            else:
                raise ValueError
            if name != 'celeba_hq':
                ckpt = get_ckpt_path(f"ema_{name}", prefix=self.args.exp)
                print("Loading checkpoint {}".format(ckpt))
            elif name == 'celeba_hq':
                ckpt = os.path.join(self.args.exp, "logs/celeba/celeba_hq.ckpt")
                if not os.path.exists(ckpt):
                    download('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt',
                             ckpt)
            else:
                raise ValueError
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            #model = torch.nn.DataParallel(model)
            print('Model is loaded')

        elif self.config.model.type == 'openai':
            config_dict = vars(self.config.model)
            model = create_model(**config_dict)
            if self.config.model.use_fp16:
                model.convert_to_fp16()
            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_diffusion.pt' % (
                self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt' % (
                        self.config.data.image_size, self.config.data.image_size), ckpt)
            else:
                ckpt = os.path.join(self.args.exp, "logs/imagenet/256x256_diffusion_uncond.pt")
                if not os.path.exists(ckpt):
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt',
                        ckpt)

            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model.eval()
            model = torch.nn.DataParallel(model)
            
            
    def sample(self, simplified, test_dataset):
        cls_fn = None
        if self.args.dataset == 'CelebA':
            name = 'celeba_hq'
            self.config.model.type == 'CelebA'
            model = Model(self.config)

            if name == 'celeba_hq':
                ckpt = os.path.join(self.args.exp, "logs/celeba/celeba_hq.ckpt")
                if not os.path.exists(ckpt):
                    download('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt',
                                ckpt)
            else:
                raise ValueError
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model, device_ids=self.args.gpulist)
            print(f'{self.args.dataset} Model is loaded')

        else:
            config_dict = vars(self.config.model)
            model = create_model(**config_dict)
            if self.config.model.use_fp16:
                model.convert_to_fp16()
            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_diffusion.pt' % (
                self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion.pt' % (
                        self.config.data.image_size, self.config.data.image_size), ckpt)
            else:
                ckpt = os.path.join(self.args.exp, "logs/imagenet/256x256_diffusion_uncond.pt")
                if not os.path.exists(ckpt):
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt',
                        ckpt)

            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model.eval()
            model = torch.nn.DataParallel(model, device_ids=self.args.gpulist)
        


        if simplified:
            print('Run ZIP.',
                  f'{self.config.time_travel.T_sampling} sampling steps.',
                  f'travel_length = {self.config.time_travel.travel_length},',
                  f'travel_repeat = {self.config.time_travel.travel_repeat}.',
                  f'Task: {self.args.deg}.'
                 )
            self.simplified_purify(model, test_dataset)
        else:
            pass
      
      
      
    def simplified_purify(self, model, test_dataset):
        args, config = self.args, self.config

        if self.type == "train":
            for n_class in test_dataset.classes:
                #os.makedirs(os.path.join(self.args.image_folder, f"{n_class}"), exist_ok=True)
                os.makedirs(os.path.join(self.args.train_image_folder_apy, f"{n_class}"), exist_ok=True)
                os.makedirs(os.path.join(self.args.train_image_folder_orig, f"{n_class}"), exist_ok=True)
        elif self.type == "test":
            for n_class in test_dataset.classes:
                os.makedirs(os.path.join(self.args.test_image_folder, f"{n_class}"), exist_ok=True)
                os.makedirs(os.path.join(self.args.test_image_folder_apy, f"{n_class}"), exist_ok=True)
                os.makedirs(os.path.join(self.args.test_image_folder_orig, f"{n_class}"), exist_ok=True)
        elif self.type == "test_pois":
            for n_class in test_dataset.classes:
                #os.makedirs(os.path.join(self.args.test_image_folder_pois, f"{n_class}"), exist_ok=True)
                os.makedirs(os.path.join(self.args.test_pois_image_folder_apy, f"{n_class}"), exist_ok=True)
                os.makedirs(os.path.join(self.args.test_pois_image_folder_orig, f"{n_class}"), exist_ok=True)


        print(f'Dataset has size {len(test_dataset)}')

        def seed_worker(worker_id):
            worker_seed = args.seed % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

        A1 = lambda z: color2gray(z)
        Ap1 = lambda z: gray2color(z)
        
        scale=round(args.deg_scale)
        A2 = torch.nn.AdaptiveAvgPool2d((256//scale,256//scale))
        Ap2 = lambda z: MeanUpsample(z,scale)
        
        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        pbar = tqdm.tqdm(val_loader)
        for x_orig, n_class in pbar:
            #print("n_class:", n_class)
            x_orig = x_orig.to(self.device)
            x_orig = data_transform(self.config, x_orig)

            y1 = A1(x_orig)
            y2 = A2(x_orig)

            Apy1 = Ap1(y1)
            Apy2 = Ap2(y2)
            
            if self.type == "train":
                for i in range(len(Apy1)):
                    class_name = test_dataset.classes[n_class['label_pois'][i].item()] # for easier access, we just put them in the poisoned label folder
                    tvu.save_image(
                        inverse_data_transform(config, Apy1[i]), os.path.join(self.args.train_image_folder_apy,f"{class_name}", f"{class_name}_{idx_so_far + i}_1.png")
                    )
                    tvu.save_image(
                        inverse_data_transform(config, Apy2[i]), os.path.join(self.args.train_image_folder_apy,f"{class_name}", f"{class_name}_{idx_so_far + i}_2.png")
                    )
                    tvu.save_image(
                        inverse_data_transform(config, x_orig[i]), os.path.join(self.args.train_image_folder_orig,f"{class_name}", f"{class_name}_{idx_so_far + i}.png")
                    )
            elif self.type == "test":
                for i in range(len(Apy1)):
                    class_name = test_dataset.classes[n_class[i].item()]
                    tvu.save_image(
                        inverse_data_transform(config, Apy1[i]), os.path.join(self.args.test_image_folder_apy,f"{class_name}", f"{class_name}_{idx_so_far + i}_1.png")
                    )
                    tvu.save_image(
                        inverse_data_transform(config, Apy2[i]), os.path.join(self.args.test_image_folder_apy,f"{class_name}", f"{class_name}_{idx_so_far + i}_2.png")
                    )
                    tvu.save_image(
                        inverse_data_transform(config, x_orig[i]), os.path.join(self.args.test_image_folder_orig,f"{class_name}", f"{class_name}_{idx_so_far + i}.png")
                    )
            elif self.type == "test_pois":
                for i in range(len(Apy1)):
                    class_name = test_dataset.classes[n_class['label_pois'][i].item()]
                    tvu.save_image(
                        inverse_data_transform(config, Apy1[i]), os.path.join(self.args.test_pois_image_folder_apy,f"{class_name}", f"{class_name}_{idx_so_far + i}_1.png")
                    )
                    tvu.save_image(
                        inverse_data_transform(config, Apy2[i]), os.path.join(self.args.test_pois_image_folder_apy,f"{class_name}", f"{class_name}_{idx_so_far + i}_2.png")
                    )
                    tvu.save_image(
                        inverse_data_transform(config, x_orig[i]), os.path.join(self.args.test_pois_image_folder_orig,f"{class_name}", f"{class_name}_{idx_so_far + i}.png")
                    )
            
             
            # init x_T
            x = torch.randn(
                y1.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

            with torch.no_grad():
                skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
                n = x.size(0)
                x0_preds = []
                xs = [x]
                
                times = get_schedule_jump(config.time_travel.T_sampling, 
                                               config.time_travel.travel_length, 
                                               config.time_travel.travel_repeat,
                                              )
                time_pairs = list(zip(times[:-1], times[1:]))
                
                
                # reverse diffusion sampling
                for i, j in tqdm.tqdm(time_pairs):
                    i, j = i*skip, j*skip
                    if j<0: j=-1 

                    #if j < i: # normal sampling 
                    t = (torch.ones(n) * i).to(x.device)
                    next_t = (torch.ones(n) * j).to(x.device)
                    at = compute_alpha(self.betas, t.long())
                    at_next = compute_alpha(self.betas, next_t.long())
                    sigma_t = ((1 - at_next)/(1-at)).sqrt()*(1-at/at_next).sqrt()
                    xt = xs[-1].to('cuda')

                    et = model(xt, t)
                    if et.size(1) == 6:
                        et = et[:, :3]
                    xt_hat1 = at.sqrt()*Apy1  + xt - Ap1(A1(xt)) + (1 - at).sqrt() * Ap1(A1(et))
                    xt_hat2 = at.sqrt()*Apy2  + xt - Ap2(A2(xt)) + (1 - at).sqrt() * Ap2(A2(et))
                    xt_hat  = (xt_hat1 + xt_hat2)/2
                    xt_hat = (1-at**self.args.at_threshold)*xt_hat + at**self.args.at_threshold*xt
                    x0_t_hat = (xt_hat - et * (1 - at).sqrt()) / at.sqrt()
                    xt_next = at_next.sqrt() * x0_t_hat + (1-at_next-sigma_t**2).sqrt()*et + sigma_t * torch.randn_like(xt)

                    xs.append(xt_next.to('cpu'))                  

                x = xs[-1]
                
            x = [inverse_data_transform(config, xi) for xi in x]

            #class_name = test_dataset.classes[n_class.item()]
            if self.type == "train":
                for i in range(len(x)):
                    clean_class_name = test_dataset.classes[n_class['label_orig'][i].item()]
                    pois_class_name = test_dataset.classes[n_class['label_pois'][i].item()]
                    os.makedirs(os.path.join(self.args.image_folder, f"{clean_class_name}", f"{pois_class_name}"), exist_ok=True)
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{clean_class_name}", f"{pois_class_name}", f"{clean_class_name}_{pois_class_name}_{idx_so_far + i}.png")
                    )
            elif self.type == "test":
                for i in range(len(x)):
                    clean_class_name = test_dataset.classes[n_class[i].item()]
                    #pois_class_name = test_dataset.classes[n_class['label_pois'][i].item()]
                    #os.makedirs(os.path.join(self.args.test_image_folder, f"{clean_class_name}", f"{pois_class_name}"), exist_ok=True)
                    tvu.save_image(
                        x[i], os.path.join(self.args.test_image_folder, f"{clean_class_name}", f"{clean_class_name}_{idx_so_far + i}.png")
                    )
            elif self.type == "test_pois":
                for i in range(len(x)):
                    clean_class_name = test_dataset.classes[n_class['label_orig'][i].item()]
                    pois_class_name = test_dataset.classes[n_class['label_pois'][i].item()]
                    os.makedirs(os.path.join(self.args.test_image_folder_pois, f"{clean_class_name}", f"{pois_class_name}"), exist_ok=True)
                    tvu.save_image(
                        x[i], os.path.join(self.args.test_image_folder_pois, f"{clean_class_name}", f"{pois_class_name}", f"{clean_class_name}_{pois_class_name}_{idx_so_far + i}.png")
                    )
            orig = inverse_data_transform(config, x_orig[0])
            mse = torch.mean((x[0].to(self.device) - orig) ** 2)
            psnr = 10 * torch.log10(1 / mse)
            avg_psnr += psnr

            idx_so_far += y1.shape[0]   
            
    
# Code form RePaint   
def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)
    return ts

def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)
        
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a
