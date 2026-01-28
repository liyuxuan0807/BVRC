"""
BVRC-D: Diffusion Model Regressor for BVRC Algorithm
Requires a pre-trained diffusion model and CLIP feature encoder.
"""

import os
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, models
from transformers import AutoModel, AutoProcessor


def make_beta_schedule(num_timesteps=1000):
    max_beta, cosine_s = 0.999, 0.008
    return torch.tensor([
        min(1 - (math.cos(((i + 1) / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2) /
            (math.cos((i / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2), max_beta)
        for i in range(num_timesteps)
    ])


def extract(input, t, x):
    out = torch.gather(input, 0, t.to(input.device))
    return out.reshape([t.shape[0]] + [1] * (len(x.shape) - 1))


def q_sample(y, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, t, noise=None):
    if noise is None:
        noise = torch.randn_like(y)
    return extract(alphas_bar_sqrt, t, y) * y + extract(one_minus_alphas_bar_sqrt, t, y) * noise


class ViTWrapper(nn.Module):
    def __init__(self, model_path, device='cuda'):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.model.eval()
        self.dim = self.model.config.projection_dim
        cfg = self.processor.image_processor
        size = (cfg.crop_size['height'], cfg.crop_size['width'])
        self.transform = transforms.Compose([
            transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor(),
            transforms.Normalize(mean=cfg.image_mean, std=cfg.image_std),
        ])
    
    def forward(self, x):
        with torch.no_grad():
            return self.model.get_image_features(pixel_values=x).float()


class ResNetEncoder(nn.Module):
    def __init__(self, feature_dim=1024):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Linear(resnet.fc.in_features, feature_dim)
    
    def forward(self, x):
        return self.projection(torch.flatten(self.encoder(x), 1))


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super().__init__()
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()
    
    def forward(self, x, t):
        return self.embed(t).view(-1, self.lin.out_features) * self.lin(x)


class ConditionalModel(nn.Module):
    def __init__(self, n_steps, fp_dim=128, feature_dim=1024):
        super().__init__()
        self.lin1 = ConditionalLinear(1 + fp_dim, feature_dim, n_steps + 1)
        self.norm1 = nn.BatchNorm1d(feature_dim)
        self.lin2 = ConditionalLinear(feature_dim, feature_dim, n_steps + 1)
        self.norm2 = nn.BatchNorm1d(feature_dim)
        self.lin3 = ConditionalLinear(feature_dim, feature_dim, n_steps + 1)
        self.norm3 = nn.BatchNorm1d(feature_dim)
        self.lin4 = nn.Linear(feature_dim, 1)
    
    def forward(self, x_embed, y, t, fp_x):
        y = torch.cat([y.view(-1, 1), fp_x], dim=-1)
        y = F.softplus(self.norm1(self.lin1(y, t))) * x_embed
        y = F.softplus(self.norm2(self.lin2(y, t)))
        y = F.softplus(self.norm3(self.lin3(y, t)))
        return self.lin4(y)


class Diffusion(nn.Module):
    def __init__(self, fp_encoder, num_timesteps=1000, fp_dim=512, device='cuda', feature_dim=1024, ddim_steps=10):
        super().__init__()
        self.device = device
        self.num_timesteps = num_timesteps
        self.fp_encoder = fp_encoder.eval()
        
        betas = make_beta_schedule(num_timesteps).float().to(device)
        alphas = 1.0 - betas
        self.alphas_cumprod = alphas.cumprod(dim=0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_cumprod)
        
        self.diffusion_encoder = ResNetEncoder(feature_dim).to(device)
        self.model = ConditionalModel(num_timesteps, fp_dim=fp_dim, feature_dim=feature_dim).to(device)
        
        c = num_timesteps // ddim_steps
        self.ddim_timesteps = np.asarray(list(range(0, num_timesteps, c))) + 1
        alphas_ddim = self.alphas_cumprod[self.ddim_timesteps]
        alphas_prev = torch.tensor([self.alphas_cumprod[0]] + self.alphas_cumprod[self.ddim_timesteps[:-1]].tolist()).to(device)
        self.register_buffer('ddim_alphas', alphas_ddim)
        self.register_buffer('ddim_alphas_prev', alphas_prev)
        self.register_buffer('ddim_sigmas', torch.zeros_like(alphas_ddim))
    
    def forward_t(self, y_0, x, t, fp_x):
        e = torch.randn_like(y_0)
        y_t = q_sample(y_0, self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt, t, e)
        return self.model(self.diffusion_encoder(x), y_t, t, fp_x), e
    
    def reverse_ddim(self, x, fp_x=None):
        with torch.no_grad():
            if fp_x is None:
                fp_x = self.fp_encoder(x)
            x_embed = self.diffusion_encoder(x)
            y_t = torch.randn([x.shape[0], 1]).to(self.device)
            
            for i, step in enumerate(np.flip(self.ddim_timesteps)):
                idx = len(self.ddim_timesteps) - i - 1
                t = torch.full((x.shape[0],), step, device=self.device, dtype=torch.long)
                e_t = self.model(x_embed, y_t, t, fp_x).detach()
                
                a_t = self.ddim_alphas[idx].view(1, 1)
                a_prev = self.ddim_alphas_prev[idx].view(1, 1)
                sqrt_one_minus = torch.sqrt(1. - a_t)
                
                y_0 = (y_t - sqrt_one_minus * e_t) / a_t.sqrt()
                y_t = a_prev.sqrt() * y_0 + (1. - a_prev).sqrt() * e_t
            
            return y_t


class DiffusionRegressor:
    """Diffusion model regressor with unified interface."""
    
    def __init__(self, clip_path=None, model_path=None, device='cuda:0', num_timesteps=1000,
                 feature_dim=1024, ddim_steps=10, finetune_epochs=5, finetune_lr=5e-5,
                 y_mean=None, y_std=None, **kwargs):
        self.clip_path = clip_path
        self.model_path = model_path
        self.device = torch.device(device)
        self.num_timesteps = num_timesteps
        self.feature_dim = feature_dim
        self.ddim_steps = ddim_steps
        self.finetune_epochs = finetune_epochs
        self.finetune_lr = finetune_lr
        self.y_mean = y_mean
        self.y_std = y_std
        self.fp_encoder = None
        self.diffusion_model = None
        self._initialized = False
        self._images_cache = None
    
    def _init_model(self):
        if self._initialized:
            return
        if self.clip_path is None:
            raise ValueError("clip_path required for DiffusionRegressor")
        
        self.fp_encoder = ViTWrapper(self.clip_path, device=self.device)
        self.diffusion_model = Diffusion(self.fp_encoder, self.num_timesteps, self.fp_encoder.dim,
                                         self.device, self.feature_dim, self.ddim_steps)
        
        if self.model_path and os.path.exists(self.model_path):
            states = torch.load(self.model_path, map_location=self.device, weights_only=False)
            self.diffusion_model.model.load_state_dict(states[0])
            self.diffusion_model.diffusion_encoder.load_state_dict(states[1])
        self._initialized = True
    
    def set_images(self, images):
        self._images_cache = images
    
    def set_normalization_stats(self, y_mean, y_std):
        self.y_mean, self.y_std = y_mean, y_std
    
    def fit(self, X, y):
        self._init_model()
        if self._images_cache is None:
            raise RuntimeError("Call set_images() first")
        
        y_flat = y.ravel() if hasattr(y, 'ravel') else y
        train_images = self._images_cache[X] if isinstance(X, np.ndarray) and X.dtype == np.int64 else self._images_cache
        
        optimizer = optim.Adam(list(self.diffusion_model.model.parameters()) +
                              list(self.diffusion_model.diffusion_encoder.parameters()), lr=self.finetune_lr)
        criterion = nn.MSELoss()
        y_norm = (torch.tensor(y_flat, dtype=torch.float32) - self.y_mean) / self.y_std
        
        self.diffusion_model.model.train()
        bs = min(64, len(y_flat))
        
        for _ in range(self.finetune_epochs):
            for i in range(0, len(y_flat), bs):
                idx = np.random.choice(len(y_flat), min(bs, len(y_flat) - i), replace=False)
                x_b = train_images[idx].to(self.device)
                y_b = y_norm[idx].view(-1, 1).to(self.device)
                t = torch.randint(0, self.num_timesteps, (len(idx),), device=self.device)
                
                pred, true = self.diffusion_model.forward_t(y_b, x_b, t, self.fp_encoder(x_b))
                optimizer.zero_grad()
                criterion(pred, true).backward()
                optimizer.step()
        return self
    
    def predict(self, X):
        self._init_model()
        if self._images_cache is None:
            raise RuntimeError("Call set_images() first")
        
        images = self._images_cache[X] if isinstance(X, np.ndarray) and X.dtype == np.int64 else self._images_cache
        self.diffusion_model.model.eval()
        
        preds = []
        for i in range(0, len(images), 64):
            x_b = images[i:i+64].to(self.device)
            p = self.diffusion_model.reverse_ddim(x_b).cpu().numpy() * self.y_std + self.y_mean
            preds.append(p.flatten())
        return np.concatenate(preds)
    
    def clone(self):
        new = DiffusionRegressor(self.clip_path, None, str(self.device), self.num_timesteps,
                                 self.feature_dim, self.ddim_steps, self.finetune_epochs,
                                 self.finetune_lr, self.y_mean, self.y_std)
        new._images_cache = self._images_cache
        if self._initialized:
            new._init_model()
            new.diffusion_model.model.load_state_dict(copy.deepcopy(self.diffusion_model.model.state_dict()))
            new.diffusion_model.diffusion_encoder.load_state_dict(copy.deepcopy(self.diffusion_model.diffusion_encoder.state_dict()))
        return new
    
    @staticmethod
    def get_name():
        return "Diffusion"
