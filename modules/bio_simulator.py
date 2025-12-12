import torch
import torch.nn as nn
import numpy as np

class BioSimulator(nn.Module):
    def __init__(self, device=torch.device('cpu'), grid_shape=(32, 32), output_size=(256, 256)):
        super(BioSimulator, self).__init__()
        self.device = device
        self.output_size = output_size
        self.grid_shape = grid_shape
        
        # --- CONSTANTS (Giữ nguyên từ file gốc) ---
        self.k, self.a, self.b = 17.3, 0.75, 120.0
        self.slope = 19152642.5
        self.half = 1.057e-7      
        self.rheo = 23.9e-6       
        self.freq = 300.0             
        self.pw = 170.0e-6   
        self.r2s = 0.5

        # --- 1. BASE GRID (Lưới điện cực gốc - chưa biến đổi) ---
        # Tọa độ gốc (đơn vị độ thị giác - dva)
        xc = torch.linspace(-15, 15, grid_shape[0], device=device)
        yc = torch.linspace(-15, 15, grid_shape[1], device=device)
        
        # Meshgrid và Flatten ngay từ đầu để dùng cho batch
        self.gx_base, self.gy_base = torch.meshgrid(xc, yc, indexing='xy')
        self.gx_base = self.gx_base.flatten().view(1, -1) # Shape: (1, N)
        self.gy_base = self.gy_base.flatten().view(1, -1)

        # --- 2. OUTPUT PIXEL GRID (Lưới pixel màn hình) ---
        rx, ry = output_size
        self.fov = 30.0 
        xs = torch.linspace(-self.fov, self.fov, rx, device=device)
        ys = torch.linspace(-self.fov, self.fov, ry, device=device)
        
        self.px, self.py = torch.meshgrid(xs, ys, indexing='xy')
        # Shape: (1, 1, H, W) để broadcast với Batch
        self.px = self.px.view(1, 1, rx, ry) 
        self.py = self.py.view(1, 1, rx, ry)
        
        # Hệ số chuyển đổi độ sang pixel
        self.deg2pix = output_size[0] / (self.fov * 2)

        # Biến lưu trạng thái (State) sau khi Config
        self.state = {}

    # =========================================================================
    # LUỒNG 1: BƯỚC CẤU HÌNH (CONFIGURATION STEP)
    # Nhiệm vụ: Xây dựng hình học mắt và Implant từ tham số bệnh nhân
    # =========================================================================
    def configure_model(self, patient_params):
        """
        Input: patient_params (Batch, 13) - Giá trị thực
        Output: Không trả về (hoặc trả về self), nhưng cập nhật self.state
        """
        batch_size = patient_params.shape[0]

        # 1.1. Giải nén tham số (Unpack)
        # Nhóm tham số vật lý (lưu lại để dùng cho bước Stimulation)
        self.state['rho'] = patient_params[:, 0].view(batch_size, 1, 1, 1) # (B, 1, 1, 1)
        
        # Nhóm hệ số độ sáng đa thức (a0 - a4)
        self.state['coeffs'] = {
            'a0': patient_params[:, 3].view(batch_size, 1, 1),
            'a1': patient_params[:, 4].view(batch_size, 1, 1),
            'a2': patient_params[:, 5].view(batch_size, 1, 1),
            'a3': patient_params[:, 6].view(batch_size, 1, 1),
            'a4': patient_params[:, 7].view(batch_size, 1, 1)
        }

        # Nhóm tham số hình học (Implant Geometry)
        x_imp = patient_params[:, 10].view(batch_size, 1) # Shift x (microns)
        y_imp = patient_params[:, 11].view(batch_size, 1) # Shift y (microns)
        rot   = patient_params[:, 12].view(batch_size, 1) # Rotation (degrees)

        # 1.2. Biến đổi Affine (Xoay & Tịnh tiến Implant)
        theta = torch.deg2rad(rot)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        # Chuyển đổi micron sang độ (giả sử 1 độ ~ 300 micron)
        dx = x_imp / 300.0
        dy = y_imp / 300.0

        # Công thức xoay vector [x, y]
        gx_new = self.gx_base * cos_t - self.gy_base * sin_t + dx
        gy_new = self.gx_base * sin_t + self.gy_base * cos_t + dy

        # 1.3. Ánh xạ Retino-Cortical (Model Mắt)
        # Mapping từ Võng mạc (w) -> Vỏ não (z)
        w = torch.complex(gx_new, gy_new) 
        ewk = torch.exp(w / self.k)
        z = (self.a * self.b * (ewk - 1)) / (self.b - self.a * ewk)
        
        # Lấy tọa độ trung tâm phosphene trên vỏ não
        vx = z.real.view(batch_size, -1, 1, 1) # (B, N, 1, 1)
        vy = z.imag.view(batch_size, -1, 1, 1)

        # Tính hệ số phóng đại vỏ não (Cortical Magnification)
        r = torch.abs(z)
        M = self.k * (1.0 / (r + self.a) - 1.0 / (r + self.b))
        self.state['M'] = M.view(batch_size, -1, 1, 1)

        # 1.4. Tính trước khoảng cách (Pre-compute Distance Matrix)
        # Tính khoảng cách từ mọi pixel màn hình đến mọi tâm phosphene
        # (x - vx)^2 + (y - vy)^2
        diff_x = (self.px - vx) * self.deg2pix
        diff_y = (self.py - vy) * self.deg2pix
        
        # Lưu dist^2 để bước sau chỉ cần exp()
        self.state['dist2'] = diff_x**2 + diff_y**2
        
        return self

    # =========================================================================
    # LUỒNG 2: BƯỚC KÍCH THÍCH (STIMULATION STEP)
    # Nhiệm vụ: Tính toán phản ứng điện và vẽ ảnh dựa trên Cấu hình đã có
    # =========================================================================
    def produce_stimulation(self, stimulation_pattern):
        """
        Input: stimulation_pattern (Batch, 1, 32, 32) - Đầu ra Encoder
        Output: Phosphene Image (Batch, 1, H, W)
        """
        if not self.state:
            raise RuntimeError("Model chưa được cấu hình! Hãy gọi configure_model() trước.")

        batch_size = stimulation_pattern.shape[0]
        flat_stim = stimulation_pattern.view(batch_size, -1) # (B, N)
        
        # 2.1. Mô phỏng Điện sinh học (Bio-physics)
        # Từ Biên độ kích thích -> Dòng điện -> Độ sáng chủ quan
        I = flat_stim * 80.0e-6 
        I_eff = torch.relu(I - self.rheo)
        Q = I_eff * self.pw * self.freq
        
        B_logit = self.slope * (Q - self.half)
        B = self.sigmoid(B_logit)
        B = B.view(batch_size, -1, 1, 1) # (B, N, 1, 1)

        # 2.2. Tính kích thước phosphene (Sigma)
        # Sigma phụ thuộc vào cường độ I và tham số Rho của bệnh nhân
        rho = self.state['rho']
        M = self.state['M']
        
        # Công thức: spread càng lớn (rho nhỏ) -> size càng to? 
        # Hoặc rho là bán kính lan tỏa. Tùy định nghĩa dataset.
        # Ở đây giả định rho là spread factor.
        size_base = torch.sqrt(I / (rho + 1e-9)) 
        size_base = size_base.view(batch_size, -1, 1, 1)
        
        # Điều chỉnh theo độ phóng đại vỏ não
        sigmas = size_base * (self.r2s / (M + 1e-9))
        
        # Đổi sang pixel
        sigma_px = sigmas * self.deg2pix
        sigma_px = torch.clamp(sigma_px, min=0.5)

        # 2.3. Render ảnh (Gaussian Rendering)
        dist2 = self.state['dist2']
        
        # Công thức Gaussian: exp(-dist^2 / 2*sigma^2)
        gauss = torch.exp(-dist2 / (2 * sigma_px**2))
        
        # Tổng hợp: Sum(Gaussian * Brightness) trên trục N (số điện cực)
        out = torch.sum(gauss * B, dim=1) # Kết quả: (B, H, W)
        out = out.unsqueeze(1) # (B, 1, H, W)

        # 2.4. Hậu xử lý (Brightness Modulation)
        # Áp dụng đa thức a0...a4
        c = self.state['coeffs']
        out_poly = c['a0'] + c['a1']*out + c['a2']*(out**2) + c['a3']*(out**3) + c['a4']*(out**4)

        return torch.clamp(out_poly, 0, 1)

    def sigmoid(self, x):
        return 1.0 / (1.0 + torch.exp(-x))

    # Hàm Forward chuẩn của nn.Module (Kết nối 2 luồng)
    def forward(self, stimulation, patient_params):
        # Bước 1: Cấu hình
        self.configure_model(patient_params)
        
        # Bước 2: Kích thích
        return self.produce_stimulation(stimulation)