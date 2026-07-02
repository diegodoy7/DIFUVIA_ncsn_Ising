import torch.nn as nn
import torch
import numpy as np
from . import get_sigmas
from .layers import *
from .normalization import get_normalization

# temperature embedding with fourier features inspired by "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains" (Tancik et al., NeurIPS 2020)
class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Se fija aleatoriamente una vez y no se entrena (Base de Fourier)
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    
    def forward(self, x):
        # x: (Batch,) -> (Batch, 1)
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class NCSNv2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.logit_transform = config.data.logit_transform
        self.rescaled = config.data.rescaled
        self.norm = get_normalization(config, conditional=False) 
        self.ngf = ngf = config.model.ngf
        self.num_classes = num_classes = config.model.num_classes

        self.act = act = get_act(config)
        self.register_buffer('sigmas', get_sigmas(config))
        self.config = config

        # --- 2. MODIFICACIÓN: EMBEDDING DE TEMPERATURA ---
        # Definimos el tamaño del embedding de temperatura (ej. 64 dimensiones)
        self.temp_embed_dim = 64
        self.temp_projector = nn.Sequential(
            GaussianFourierProjection(embed_dim=self.temp_embed_dim),
            nn.Linear(self.temp_embed_dim, self.temp_embed_dim),
            self.act
        )

        # --- 3. MODIFICACIÓN: CANALES DE ENTRADA ---
        # La primera convolución ahora recibe: Canales de Imagen (1) + Canales de Temp (64)
        input_channels = config.data.channels + self.temp_embed_dim
        
        self.begin_conv = nn.Conv2d(input_channels, ngf, 3, stride=1, padding=1, padding_mode='circular')

        # El resto se mantiene igual...
        self.normalizer = self.norm(ngf, self.num_classes)
        self.end_conv = nn.Conv2d(ngf, config.data.channels, 3, stride=1, padding=1, padding_mode='circular')

        self.res1 = nn.ModuleList([
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act, normalization=self.norm),
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act, normalization=self.norm)]
        )

        self.res2 = nn.ModuleList([
            ResidualBlock(self.ngf, 2 * self.ngf, resample='down', act=act, normalization=self.norm),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act, normalization=self.norm)]
        )

        self.res3 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act, normalization=self.norm, dilation=2),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act, normalization=self.norm, dilation=2)]
        )

        if config.data.image_size == 28:
            self.res4 = nn.ModuleList([
                ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act, normalization=self.norm, adjust_padding=True, dilation=4),
                ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act, normalization=self.norm, dilation=4)]
            )
        else:
            self.res4 = nn.ModuleList([
                ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act, normalization=self.norm, adjust_padding=False, dilation=4),
                ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act, normalization=self.norm, dilation=4)]
            )

        self.refine1 = RefineBlock([2 * self.ngf], 2 * self.ngf, act=act, start=True)
        self.refine2 = RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, act=act)
        self.refine3 = RefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, act=act)
        self.refine4 = RefineBlock([self.ngf, self.ngf], self.ngf, act=act, end=True)

    def _compute_cond_module(self, module, x):
        for m in module:
            x = m(x)
        return x

    def forward(self, x, y, y_temp=None):
        # x: (Batch, 1, H, W)
        # y: (Batch,) -> Indices de ruido
        # y_temp: (Batch,) -> Temperaturas continuas

        if not self.logit_transform and not self.rescaled:
            h = 2 * x - 1.
        else:
            h = x
            
        # --- 4. MODIFICACIÓN: INYECCIÓN DE TEMPERATURA ---
        if y_temp is not None:
            # A. Proyectar T a un vector denso (B, 64)
            temp_emb = self.temp_projector(y_temp)
            
            # B. Expandir espacialmente para que coincida con la imagen (B, 64, H, W)
            # .view(B, 64, 1, 1) -> crea dimensiones
            # .expand(...) -> repite los valores en ancho y alto
            temp_map = temp_emb.view(temp_emb.shape[0], temp_emb.shape[1], 1, 1).expand(
                temp_emb.shape[0], temp_emb.shape[1], h.shape[2], h.shape[3]
            )
            
            # C. Concatenar: Ahora la entrada tiene 1 + 64 canales
            h = torch.cat([h, temp_map], dim=1)

        # La convolución inicial ahora espera los canales extra
        output = self.begin_conv(h)

        # El resto sigue igual...
        layer1 = self._compute_cond_module(self.res1, output)
        layer2 = self._compute_cond_module(self.res2, layer1)
        layer3 = self._compute_cond_module(self.res3, layer2)
        layer4 = self._compute_cond_module(self.res4, layer3)

        ref1 = self.refine1([layer4], layer4.shape[2:])
        ref2 = self.refine2([layer3, ref1], layer3.shape[2:])
        ref3 = self.refine3([layer2, ref2], layer2.shape[2:])
        output = self.refine4([layer1, ref3], layer1.shape[2:])

        output = self.normalizer(output)
        output = self.act(output)
        output = self.end_conv(output)

        used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))

        output = output / used_sigmas

        return output