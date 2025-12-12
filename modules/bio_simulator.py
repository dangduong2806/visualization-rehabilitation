import torch
import torch.nn as nn
import numpy as np
import math

class BioSimulator(nn.Module):
    def __init__(self, device=torch.device('cpu'), grid_shape=(32, 32), output_size=(256, 256)):
        super(BioSimulator, self).__init__()
        self.device = device
        self.output_size = output_size
        self.grid_shape = grid_shape

        self.k, self.a, self.b = 17.3, 0.75, 120.0
        
        self.spread = 675.0e-6 
        self.r2s = 0.5
        
        self.slope = 19152642.5
        self.half = 1.057e-7      
        self.rheo = 23.9e-6       
        self.freq = 300.0             
        self.pw = 170.0e-6   

        # mapping
        xc = torch.linspace(-15, 15, grid_shape[0], device=device)
        yc = torch.linspace(-15, 15, grid_shape[1], device=device)
        gx, gy = torch.meshgrid(xc, yc, indexing='xy')
        
        w = torch.complex(gx, gy)
        ewk = torch.exp(w / self.k)
        z = (self.a * self.b * (ewk - 1)) / (self.b - self.a * ewk)
        
        self.vx = z.real.flatten()
        self.vy = z.imag.flatten()
        
        max_ecc = max(self.vx.abs().max(), self.vy.abs().max()).item()
        self.fov = max_ecc * 1.1
        print(f"vùng nhìn rộng +/- {self.fov:.2f} độ")

        valid = (torch.abs(self.vx) < 90)
        
        r = torch.abs(z).flatten()
        self.M = self.k * (1.0 / (r + self.a) - 1.0 / (r + self.b))
        
        self.vx = self.vx[valid]
        self.vt = self.vy[valid]
        self.M = self.M[valid]
        self.idx = torch.nonzero(valid.flatten()).squeeze()
        
        rx, ry = output_size
        
        xs = torch.linspace(-self.fov, self.fov, rx, device=device)
        ys = torch.linspace(-self.fov, self.fov, ry, device=device)
        
        self.px, self.py = torch.meshgrid(xs, ys, indexing='xy')
        self.px = self.px.view(1, 1, rx, ry)
        self.py = self.py.view(1, 1, rx, ry)
        
        self.vx = self.vx.view(1, -1, 1, 1)
        self.vy = self.vy.view(1, -1, 1, 1)
        self.M = self.M.view(1, -1, 1, 1)

    def sigmoid(self, x):
        return 1.0 / (1.0 + torch.exp(-x))

    def forward(self, stimulation):
        batch_size = stimulation.shape[0]
        
        flat = stimulation.view(batch_size, -1)
        I = flat[:, self.idx] * 80.0e-6 
        
        I_eff = torch.relu(I - self.rheo)
        Q = I_eff * self.pw * self.freq
        
        B_logit = self.slope * (Q - self.half)
        B = self.sigmoid(B_logit)
        B = B.view(batch_size, -1, 1, 1)
        
        size_base = torch.sqrt(I / self.spread)
        
        sigmas = size_base.view(batch_size, -1, 1, 1) * (self.r2s / self.M)
        
        deg2pix = self.output_size[0] / (self.fov * 2)
        sigma_px = sigmas * deg2pix
        
        sigma_px = torch.clamp(sigma_px, min=1.0)

        diff_x = (self.px - self.vx) * deg2pix
        diff_y = (self.py - self.vy) * deg2pix
        dist2 = diff_x**2 + diff_y**2
        
        gauss = torch.exp(-dist2 / (2 * sigma_px**2))
        
        out = torch.sum(gauss * B, dim=1)
        out = out.unsqueeze(1)
        
        out = out * 2.0
        
        return torch.clamp(out, 0, 1)