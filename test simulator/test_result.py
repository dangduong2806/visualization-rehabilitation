import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from datetime import datetime
from torchvision import transforms

from modules.models import Representer, E2E_PhospheneSimulator_jaap
from modules.utils import get_pMask_jaap 
from configs import IMG_SIZE
from utils import normalize_params
from modules.bio_simulator import BioSimulatorHILO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "D:/Tinh-toan-khtk/E2E-Point-SPV/Point-SPV/res/models_reg/enc2_20251215_124028_1"
IMAGE_PATH = r"D:/Tinh-toan-khtk/E2E-Point-SPV/imagenette2-320/val/n02979186/ILSVRC2012_val_00008651.JPEG"

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Thêm hàm sinh tham số
def generate_test_params(device):
    """
    Tạo một bộ tham số cụ thể để test (ví dụ: xoay 15 độ, rho=2.0)
    """
    # 1. Khởi tạo tham số thực (Real values)
    real = torch.zeros(1, 13, device=device)
    
    # Cài đặt giá trị mặc định (theo logic create_default_params)
    # Hoặc cài đặt thủ công các hiệu ứng muốn test:
    real[:, 0] = 2.0   # Rho (độ lan tỏa)
    real[:, 1] = 0.98  # Lambda
    real[:, 2] = 1.0   # Omega
    real[:, 3] = 0.0   # a0
    real[:, 4] = 0.5   # a1
    # ... các a khác để 0
    real[:, 8] = 4200.0 # od_x
    real[:, 9] = 500.0  # od_y
    real[:, 10] = 0.0   # x_imp
    real[:, 11] = 0.0   # y_imp
    real[:, 12] = 15.0  # Rot: Xoay 15 độ để thấy rõ hiệu ứng
    
    # 2. Chuẩn hóa tham số (dùng hàm import từ bio_simulator để đảm bảo nhất quán với lúc train)
    norm = normalize_params(real)
    
    return real, norm
# -------------------------------------------------------

def load_image_robust(path):
    if os.path.exists(path):
        try:
            stream = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception as e:
            print(f"lỗi đọc file: {e}")
    return np.zeros((256, 256, 3), dtype=np.uint8)

def visualize_phosphene():
    try:
        # LƯU Ý QUAN TRỌNG: arryaOut phải khớp với lúc train (Bio model dùng True để ra vector 1024)
        model = Representer(n_channels=3, n_patient_params=13, arryaOut=True) 
        
        # Load weight (cần strict=False nếu có sự thay đổi nhỏ về tên layer, hoặc True nếu khớp hoàn toàn)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
        model.to(DEVICE)
        model.eval()
        print("Encoder loaded successfully")
    except Exception as e:
        print(f"error loading model: {e}")
        return
    
    # --- 2. LOAD SIMULATOR CŨ (Để so sánh - Tùy chọn) ---
    try:
        pMask = get_pMask_jaap().to(DEVICE)
        sim_old = E2E_PhospheneSimulator_jaap(pMask=pMask, device=DEVICE)
        sim_old.to(DEVICE)
        sim_old.eval()
        print("regular loaded")
    except:
        sim_old = None
        print("lỗi regular")

    # --- 3. LOAD BIO SIMULATOR MỚI ---
    # Dùng đúng tên class BioSimulatorHILO
    sim_new = BioSimulatorHILO(device=DEVICE, grid_shape=(32,32), output_size=(IMG_SIZE, IMG_SIZE))
    sim_new.eval()
    print("BioSimulatorHILO loaded")

    original_img = load_image_robust(IMAGE_PATH)

    # Transform chuẩn ImageNet
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)), # Resize cứng về 256x256
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(original_img).unsqueeze(0).to(DEVICE)

    # --- CẬP NHẬT 2: Sinh tham số ---
    real_params, norm_params = generate_test_params(DEVICE)
    print(f"Testing with params: Rho={real_params[0,0].item()}, Rot={real_params[0,12].item()}")

    with torch.no_grad():
        # Encoder: Nhận normalized params
        encoder_out = model(input_tensor, patient_params=norm_params)
        
        if sim_old:
            try:
                # Regular Sim thường cần input dạng ảnh (spatial), nếu encoder_out là vector thì phải reshape
                if encoder_out.dim() == 2: # (B, 1024)
                    spat_map = encoder_out.view(-1, 1, 32, 32)
                    out_old = sim_old(spat_map)
                else:
                    out_old = sim_old(encoder_out)
            except Exception as e:
                print(f"Lỗi chạy sim cũ: {e}")
                out_old = torch.zeros_like(input_tensor)
        else:
            out_old = torch.zeros_like(input_tensor)

        # Sim mới (Bio): Nhận real params, tham số tên là 'phi'
        out_new = sim_new(encoder_out, phi=real_params)
    
    # --- 5. HIỂN THỊ KẾT QUẢ ---
    # Encoder output visualization (nếu là vector thì reshape để xem dạng map)
    if encoder_out.dim() == 2:
        enc_vis = encoder_out.view(32, 32).cpu().numpy()
    else:
        enc_vis = encoder_out.squeeze().cpu().numpy()
    
    # [FIX 3] Xử lý hiển thị ảnh (Transpose Channel)
    def process_for_plot(tensor_img):
        img = tensor_img.squeeze().cpu().numpy() # (3, 256, 256) hoặc (256, 256)
        if img.ndim == 3:
            img = img.transpose(1, 2, 0) # Chuyển thành (256, 256, 3)
        # Scale về 0-1 nếu cần, hoặc để matplotlib tự lo
        return img
    
    old_img_plot = process_for_plot(out_old)
    new_img_plot = process_for_plot(out_new)

    plt.figure(figsize=(20, 6))

    plt.subplot(1, 4, 1)
    plt.title("Input")
    plt.imshow(original_img)
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("Encoder Output")
    plt.imshow(enc_vis, cmap='viridis', interpolation='nearest')

    plt.subplot(1, 4, 3)
    plt.title("Regular Sim")
    # cmap='gray' chỉ hoạt động tốt nếu ảnh là 2D hoặc 3D (H,W,C)
    if old_img_plot.ndim == 3:
        plt.imshow(old_img_plot) # RGB
    else:
        plt.imshow(old_img_plot, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("Bio Sim")
    if new_img_plot.ndim == 3:
        plt.imshow(new_img_plot)
    else:
        plt.imshow(new_img_plot, cmap='gray', vmin=0, vmax=1, origin='lower')
    plt.axis('off')

    output_folder = "phosphenes_result"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    save_path = os.path.join(output_folder, f"comparison_{timestamp}.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Lưu kết quả tại: {save_path}")
    plt.show()

if __name__ == "__main__":
    visualize_phosphene()
