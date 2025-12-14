import torch
import torch.nn as nn
import numpy as np
import math

class BioSimulatorHILO(nn.Module):
    
    def __init__(self, device=torch.device('cpu'), grid_shape=(32, 32), output_size=(256, 256)):
        super(BioSimulatorHILO, self).__init__()
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

        xc = torch.linspace(-15, 15, grid_shape[0], device=device)
        yc = torch.linspace(-15, 15, grid_shape[1], device=device)
        gx, gy = torch.meshgrid(xc, yc, indexing='xy')
        
        self.register_buffer('gx_base', gx.flatten().view(1, -1))
        self.register_buffer('gy_base', gy.flatten().view(1, -1))
        
        w = torch.complex(gx, gy)
        ewk = torch.exp(w / self.k)
        z = (self.a * self.b * (ewk - 1)) / (self.b - self.a * ewk)
        
        vx_default = z.real.flatten()
        vy_default = z.imag.flatten()
        
        max_ecc = max(vx_default.abs().max(), vy_default.abs().max()).item()
        self.fov = max_ecc * 1.1
        
        valid = (torch.abs(vx_default) < 90)
        r = torch.abs(z).flatten()
        M_default = self.k * (1.0 / (r + self.a) - 1.0 / (r + self.b))
        
        self.register_buffer('idx', torch.nonzero(valid.flatten()).squeeze())
        self.n_valid = self.idx.shape[0]
        
        rx, ry = output_size
        xs = torch.linspace(-self.fov, self.fov, rx, device=device)
        ys = torch.linspace(-self.fov, self.fov, ry, device=device)
        
        px, py = torch.meshgrid(xs, ys, indexing='xy')
        self.register_buffer('px', px.view(1, 1, rx, ry))
        self.register_buffer('py', py.view(1, 1, rx, ry))
        
        self.deg2pix = output_size[0] / (self.fov * 2)
        
        # default patient params
        default_phi = torch.zeros(1, 13, device=device)
        default_phi[0, 3] = 1.0
        default_phi[0, 4] = 1.0
        default_phi[0, 5] = 1.0
        default_phi[0, 6] = 1.0
        default_phi[0, 7] = 1.0
        self.register_buffer('default_phi', default_phi)

    def sigmoid(self, x):
        return 1.0 / (1.0 + torch.exp(-x))
    
    def _apply_implant_geometry(self, phi):
        """        
        Args:
            phi: (B, 13) patient parameters
            
        Returns:
            gx, gy: (B, N) tọa độ điện cực đã transform
        """
        batch_size = phi.shape[0]
        
        x_shift = phi[:, 0].view(batch_size, 1)  # mm → độ
        y_shift = phi[:, 1].view(batch_size, 1)
        rotation = phi[:, 2].view(batch_size, 1)  # độ
        
        dx = x_shift * 3.5
        dy = y_shift * 3.5
        
        theta = torch.deg2rad(rotation)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        
        # Affine transform
        # gx_base, gy_base: (1, N)
        gx = self.gx_base * cos_t - self.gy_base * sin_t + dx
        gy = self.gx_base * sin_t + self.gy_base * cos_t + dy
        
        return gx, gy  # (B, N)
    
    def _compute_cortical_mapping(self, gx, gy):
        """
        Retino-Cortical
        
        Args:
            gx, gy: (B, N) tọa độ điện cực
            
        Returns:
            vx, vy: (B, N) tọa độ trên vỏ não
            M: (B, N) hệ số phóng đại
        """
        w = torch.complex(gx, gy)
        ewk = torch.exp(w / self.k)
        z = (self.a * self.b * (ewk - 1)) / (self.b - self.a * ewk)
        
        vx = z.real
        vy = z.imag
        r = torch.abs(z)
        M = self.k * (1.0 / (r + self.a) - 1.0 / (r + self.b))
        
        return vx, vy, M

    def forward(self, stimulation, phi=None):
        """
        Forward pass của simulator
        
        Args:
            stimulation: (B, 1, 32, 32) output từ encoder
            phi: (B, 13) patient parameters, None = dùng default
            
        Returns:
            phosphene: (B, 1, H, W) ảnh mô phỏng
        """
        batch_size = stimulation.shape[0]
        
        if phi is None:
            phi = self.default_phi.expand(batch_size, -1)
        
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)
        if phi.shape[0] == 1 and batch_size > 1:
            phi = phi.expand(batch_size, -1)
            
        spread_scale = phi[:, 3].view(batch_size, 1).clamp(min=0.1, max=10.0)
        brightness_scale = phi[:, 4].view(batch_size, 1).clamp(min=0.1, max=5.0)
        size_scale = phi[:, 5].view(batch_size, 1).clamp(min=0.1, max=5.0)
        threshold_scale = phi[:, 6].view(batch_size, 1).clamp(min=0.1, max=5.0)
        contrast = phi[:, 7].view(batch_size, 1).clamp(min=0.1, max=5.0)
        
        has_geometry_change = (phi[:, 0:3].abs().sum() > 1e-6)
        
        if has_geometry_change:
            gx, gy = self._apply_implant_geometry(phi)
            vx, vy, M = self._compute_cortical_mapping(gx, gy)
            
            # Filter valid points (trong FOV)
            vx = vx.view(batch_size, -1, 1, 1)
            vy = vy.view(batch_size, -1, 1, 1)
            M = M.view(batch_size, -1, 1, 1)
            
            flat = stimulation.view(batch_size, -1)
        else:
            gx, gy = self.gx_base, self.gy_base
            vx, vy, M = self._compute_cortical_mapping(gx, gy)
            
            vx = vx[:, self.idx].view(batch_size, -1, 1, 1)
            vy = vy[:, self.idx].view(batch_size, -1, 1, 1)
            M = M[:, self.idx].view(batch_size, -1, 1, 1)
            
            flat = stimulation.view(batch_size, -1)
            flat = flat[:, self.idx]
        
        I = flat * 80.0e-6
        
        rheo_adjusted = self.rheo * threshold_scale
        I_eff = torch.relu(I - rheo_adjusted)
        Q = I_eff * self.pw * self.freq
        
        B_logit = self.slope * (Q - self.half)
        B = self.sigmoid(B_logit)
        
        B = B * brightness_scale
        B = torch.pow(B, 1.0 / contrast.clamp(min=0.5))
        B = B.view(batch_size, -1, 1, 1)
        
        spread_adjusted = self.spread * spread_scale
        size_base = torch.sqrt(I / spread_adjusted)
        
        sigmas = size_base.view(batch_size, -1, 1, 1) * (self.r2s / (M + 1e-9)) * size_scale.view(batch_size, 1, 1, 1)
        
        sigma_px = sigmas * self.deg2pix
        sigma_px = torch.clamp(sigma_px, min=1.0)
        
        diff_x = (self.px - vx) * self.deg2pix
        diff_y = (self.py - vy) * self.deg2pix
        dist2 = diff_x**2 + diff_y**2
        
        gauss = torch.exp(-dist2 / (2 * sigma_px**2))
        
        out = torch.sum(gauss * B, dim=1)
        out = out.unsqueeze(1)
        
        out = out * 2.0
        
        return torch.clamp(out, 0, 1)


# === HELPER FUNCTIONS ===

def create_default_phi(batch_size=1, device='cpu'):
    phi = torch.zeros(batch_size, 13, device=device)
    phi[:, 3] = 1.0  # spread_scale
    phi[:, 4] = 1.0  # brightness_scale
    phi[:, 5] = 1.0  # size_scale
    phi[:, 6] = 1.0  # threshold_scale
    phi[:, 7] = 1.0  # contrast
    return phi


def create_random_phi(batch_size=1, device='cpu', seed=None):
    """
    Tạo patient params ngẫu nhiên cho training
    
    Ranges:
        x_shift, y_shift: [-2, 2] mm
        rotation: [-15, 15] độ
        spread_scale: [0.5, 2.0]
        brightness_scale: [0.5, 2.0]
        size_scale: [0.5, 2.0]
        threshold_scale: [0.5, 2.0]
        contrast: [0.5, 2.0]
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    phi = torch.zeros(batch_size, 13, device=device)
    
    # Geometry
    phi[:, 0] = torch.rand(batch_size, device=device) * 4 - 2      # x_shift: [-2, 2]
    phi[:, 1] = torch.rand(batch_size, device=device) * 4 - 2      # y_shift: [-2, 2]
    phi[:, 2] = torch.rand(batch_size, device=device) * 30 - 15    # rotation: [-15, 15]
    
    # Scales (0.5 to 2.0)
    phi[:, 3] = torch.rand(batch_size, device=device) * 1.5 + 0.5  # spread_scale
    phi[:, 4] = torch.rand(batch_size, device=device) * 1.5 + 0.5  # brightness_scale
    phi[:, 5] = torch.rand(batch_size, device=device) * 1.5 + 0.5  # size_scale
    phi[:, 6] = torch.rand(batch_size, device=device) * 1.5 + 0.5  # threshold_scale
    phi[:, 7] = torch.rand(batch_size, device=device) * 1.5 + 0.5  # contrast
    
    return phi


def expand_phi_to_feature_maps(phi, height, width):
    """    
    Args:
        phi: (B, 13) patient parameters
        height, width: kích thước ảnh
        
    Returns:
        phi_maps: (B, 13, H, W)
    """
    batch_size = phi.shape[0]
    phi_maps = phi.view(batch_size, 13, 1, 1).expand(-1, -1, height, width)
    return phi_maps