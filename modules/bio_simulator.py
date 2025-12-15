import torch
import torch.nn as nn
import numpy as np
import math

class BioSimulatorHILO(nn.Module):
    
    PARAM_RANGES = {
        'rho': (1.5, 8.0),
        'lambda': (0.45, 0.98),
        'omega': (0.9, 1.1),
        'a0': (0.27, 0.57),
        'a1': (0.42, 0.62),
        'a2': (0.005, 0.025),
        'a3': (0.2, 0.7),
        'a4': (-0.5, -0.1),
        'od_x': (3700.0, 4700.0),
        'od_y': (0.0, 1000.0),
        'x_imp': (-500.0, 500.0),
        'y_imp': (-500.0, 500.0),
        'rot': (-30.0, 30.0)
    }
    
    PARAM_KEYS = ['rho', 'lambda', 'omega', 'a0', 'a1', 'a2', 'a3', 'a4',
                  'od_x', 'od_y', 'x_imp', 'y_imp', 'rot']
    
    def __init__(self, device=torch.device('cpu'), grid_shape=(32, 32), output_size=(256, 256)):
        super(BioSimulatorHILO, self).__init__()
        self.device = device
        self.output_size = output_size
        self.grid_shape = grid_shape

        self.k, self.a, self.b = 17.3, 0.75, 120.0
        self.spread_base = 675.0e-6 
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
        self.register_buffer('idx', torch.nonzero(valid.flatten()).squeeze())
        self.n_valid = self.idx.shape[0]
        
        rx, ry = output_size
        xs = torch.linspace(-self.fov, self.fov, rx, device=device)
        ys = torch.linspace(-self.fov, self.fov, ry, device=device)
        
        px, py = torch.meshgrid(xs, ys, indexing='xy')
        self.register_buffer('px', px.view(1, 1, rx, ry))
        self.register_buffer('py', py.view(1, 1, rx, ry))
        
        self.deg2pix = output_size[0] / (self.fov * 2)
        
        default_phi = self._create_default_params()
        self.register_buffer('default_phi', default_phi)

    def _create_default_params(self):
        default = torch.zeros(1, 13, device=self.device)
        for i, key in enumerate(self.PARAM_KEYS):
            low, high = self.PARAM_RANGES[key]
            default[0, i] = (low + high) / 2.0
        return default

    def sigmoid(self, x):
        return 1.0 / (1.0 + torch.exp(-x))
    
    def _unpack_params(self, phi):
        """       
        Args:
            phi: (B, 13) real values
            
        Returns:
            dict chứa các tham số đã reshape
        """
        batch_size = phi.shape[0]
        
        return {
            'rho': phi[:, 0].view(batch_size, 1),           # điện trở suất
            'lambda_': phi[:, 1].view(batch_size, 1),       # decay factor
            'omega': phi[:, 2].view(batch_size, 1),         # scale factor
            # polynomial coefficients - shape (B, 1, 1, 1) để broadcast với (B, 1, H, W)
            'a0': phi[:, 3].view(batch_size, 1, 1, 1),
            'a1': phi[:, 4].view(batch_size, 1, 1, 1),
            'a2': phi[:, 5].view(batch_size, 1, 1, 1),
            'a3': phi[:, 6].view(batch_size, 1, 1, 1),
            'a4': phi[:, 7].view(batch_size, 1, 1, 1),
            'od_x': phi[:, 8].view(batch_size, 1),          # optic disc position
            'od_y': phi[:, 9].view(batch_size, 1),
            'x_imp': phi[:, 10].view(batch_size, 1),        # implant shift
            'y_imp': phi[:, 11].view(batch_size, 1),
            'rot': phi[:, 12].view(batch_size, 1),          # rotation
        }
    
    def _apply_implant_geometry(self, phi_params):
        """        
        Args:
            phi_params: dict từ _unpack_params()
            
        Returns:
            gx, gy: (B, N) tọa độ điện cực đã transform
        """
        batch_size = phi_params['x_imp'].shape[0]
        
        x_imp = phi_params['x_imp']
        y_imp = phi_params['y_imp']
        rot = phi_params['rot']
        
        MICRON_PER_DEGREE = 280.0
        dx = x_imp / MICRON_PER_DEGREE
        dy = y_imp / MICRON_PER_DEGREE
        
        theta = torch.deg2rad(rot)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        
        gx = self.gx_base * cos_t - self.gy_base * sin_t + dx
        gy = self.gx_base * sin_t + self.gy_base * cos_t + dy
        
        return gx, gy  # (B, N)
    
    def _compute_cortical_mapping(self, gx, gy):
        """
        Retino-Cortical
        """
        w = torch.complex(gx, gy)
        ewk = torch.exp(w / self.k)
        z = (self.a * self.b * (ewk - 1)) / (self.b - self.a * ewk)
        
        vx = z.real
        vy = z.imag
        r = torch.abs(z)
        M = self.k * (1.0 / (r + self.a) - 1.0 / (r + self.b))
        
        return vx, vy, M
    
    def _apply_polynomial_brightness(self, out, phi_params):
        """
        out_final = a0 + a1*out + a2*out^2 + a3*out^3 + a4*out^4
        """
        a0 = phi_params['a0']
        a1 = phi_params['a1']
        a2 = phi_params['a2']
        a3 = phi_params['a3']
        a4 = phi_params['a4']
        
        out_poly = (a0 + 
                    a1 * out + 
                    a2 * (out ** 2) + 
                    a3 * (out ** 3) + 
                    a4 * (out ** 4))
        
        return out_poly

    def forward(self, stimulation, phi=None):
        """
        Forward pass của simulator
        
        Args:
            stimulation: (B, 1, 32, 32) output từ encoder
            phi: (B, 13) patient parameters (REAL values, chưa chuẩn hóa)
                 None = dùng default (giá trị trung bình)
            
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
        
        params = self._unpack_params(phi)
        
        has_geometry_change = (
            phi[:, 10].abs().sum() > 1e-3 or
            phi[:, 11].abs().sum() > 1e-3 or
            phi[:, 12].abs().sum() > 1e-3
        )
        
        if has_geometry_change:
            gx, gy = self._apply_implant_geometry(params)
            vx, vy, M = self._compute_cortical_mapping(gx, gy)
            
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
            
        # Bắt đầu sửa đổi an toàn
        I = flat * 80.0e-6
        
        I_eff = torch.relu(I - self.rheo)
        Q = I_eff * self.pw * self.freq
        
        B_logit = self.slope * (Q - self.half)
        B = self.sigmoid(B_logit)
        
        # áp dụng omega (scale factor) và lambda (decay)
        omega = params['omega']
        lambda_ = params['lambda_']
        
        B = B * omega
        B = B.view(batch_size, -1, 1, 1)
        
        rho = params['rho'].view(batch_size, 1)
        
        rho_normalized = rho / 4.75
        spread_adjusted = self.spread_base * rho_normalized
        
        size_base = torch.sqrt(I / (spread_adjusted + 1e-8))
        sigmas = size_base.view(batch_size, -1, 1, 1) * (self.r2s / (M + 1e-8))
        
        sigma_px = sigmas * self.deg2pix
        sigma_px = torch.clamp(sigma_px, min=0.5, max=50.0)
        
        diff_x = (self.px - vx) * self.deg2pix
        diff_y = (self.py - vy) * self.deg2pix
        dist2 = diff_x**2 + diff_y**2
        
        # decay = lambda_.view(batch_size, 1, 1, 1)
        
        gauss = torch.exp(-dist2 / (2 * sigma_px**2 + 1e-8))
        
        out = torch.sum(gauss * B, dim=1)
        out = out.unsqueeze(1)
        
        out = out * 2.0
        
        # [SAFETY 4] QUAN TRỌNG NHẤT: Clamp giá trị trước khi đưa vào đa thức bậc 4
        # Nếu out > 2 hoặc 3, out^4 sẽ cực lớn -> Gradient Explosion -> Loss NaN
        out_clamped = torch.clamp(out, 0.0, 3.0)
        
        out = self._apply_polynomial_brightness(out_clamped, params)
        
        return torch.clamp(out, 0, 1)



PARAM_RANGES = {
    'rho': (1.5, 8.0),
    'lambda': (0.45, 0.98),
    'omega': (0.9, 1.1),
    'a0': (0.27, 0.57),
    'a1': (0.42, 0.62),
    'a2': (0.005, 0.025),
    'a3': (0.2, 0.7),
    'a4': (-0.5, -0.1),
    'od_x': (3700.0, 4700.0),
    'od_y': (0.0, 1000.0),
    'x_imp': (-500.0, 500.0),
    'y_imp': (-500.0, 500.0),
    'rot': (-30.0, 30.0)
}

PARAM_KEYS = ['rho', 'lambda', 'omega', 'a0', 'a1', 'a2', 'a3', 'a4',
              'od_x', 'od_y', 'x_imp', 'y_imp', 'rot']


def generate_random_params(batch_size, device='cpu'):
    params_list = []
    for k in PARAM_KEYS:
        low, high = PARAM_RANGES[k]
        val = torch.rand(batch_size, 1, device=device) * (high - low) + low
        params_list.append(val)
    
    params = torch.cat(params_list, dim=1)
    return params


def normalize_params(params_tensor):
    normalized_list = []
    for i, k in enumerate(PARAM_KEYS):
        low, high = PARAM_RANGES[k]
        normalized_val = (params_tensor[:, i:i+1] - low) / (high - low)
        normalized_list.append(normalized_val)
    normalized_params = torch.cat(normalized_list, dim=1)
    return normalized_params


def denormalize_params(normalized_tensor):
    denormalized_list = []
    for i, k in enumerate(PARAM_KEYS):
        low, high = PARAM_RANGES[k]
        real_val = normalized_tensor[:, i:i+1] * (high - low) + low
        denormalized_list.append(real_val)
    denormalized_params = torch.cat(denormalized_list, dim=1)
    return denormalized_params


def create_default_params(batch_size=1, device='cpu'):
    params = torch.zeros(batch_size, 13, device=device)
    for i, k in enumerate(PARAM_KEYS):
        low, high = PARAM_RANGES[k]
        params[:, i] = (low + high) / 2.0
    return params


def create_baseline_params(batch_size=1, device='cpu'):
    params = torch.zeros(batch_size, 13, device=device)
    
    params[:, 0] = 4.75    # rho
    params[:, 1] = 0.98    # lambda
    params[:, 2] = 1.0     # omega
    
    # Polynomial: a0 + a1*x ≈ x (identity-ish)
    params[:, 3] = 0.0     # a0: offset nhỏ
    params[:, 4] = 0.5     # a1: linear term
    params[:, 5] = 0.0     # a2: no quadratic
    params[:, 6] = 0.0     # a3: no cubic
    params[:, 7] = 0.0     # a4: no quartic
    
    # Geometry: no shift, no rotation
    params[:, 8] = 4200.0  # od_x: trung bình
    params[:, 9] = 500.0   # od_y: trung bình
    params[:, 10] = 0.0    # x_imp: no shift
    params[:, 11] = 0.0    # y_imp: no shift
    params[:, 12] = 0.0    # rot: no rotation
    
    return params


def expand_phi_to_feature_maps(phi, height, width):
    """    
    Args:
        phi: (B, 13) patient parameters (normalized)
        height, width: kích thước ảnh
        
    Returns:
        phi_maps: (B, 13, H, W)
    """
    batch_size = phi.shape[0]
    phi_maps = phi.view(batch_size, 13, 1, 1).expand(-1, -1, height, width)
    return phi_maps
