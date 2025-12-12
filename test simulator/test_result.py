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
from modules.bio_simulator import BioSimulator
from modules.utils import get_pMask_jaap 
from configs import IMG_SIZE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "res/models_reg/enc2_20251209_064446_0"
IMAGE_PATH = r"C:\Users\Dell\OneDrive\Máy tính\Study\project\Point-SPV\Point-SPV\imagenette2-320\imagenette2-320\train\n01440764\ILSVRC2012_val_00009346.JPEG"

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Thêm hàm sinh tham số
def generate_and_norm_params(device):
    # Cài đặt nhanh để test, lý tưởng nên import từ file huấn luyện
    # 1. Sinh tham số thực
    real = torch.zeros(1, 13, device=device)
    real[:, 0] = 2.0  # Rho
    real[:, 1] = 0.5  # Lambda
    real[:, 3] = 0.0  # a0
    real[:, 4] = 1.0  # a1
    real[:, 12] = 15.0 # Rot 15 độ
    
    # 2. Sinh tham số chuẩn hóa (giả sử min-max đơn giản)
    # Lưu ý: Cần dùng đúng công thức chuẩn hóa như lúc train
    norm = real.clone() 
    # Ví dụ demo: scale rho từ [1.5, 8] về [0, 1]
    norm[:, 0] = (real[:, 0] - 1.5) / (8.0 - 1.5)
    
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
        # load encoder model
        model = Representer(n_channels=3, n_patient_params=13, arrayOut=False)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print("encoder loaded")
    except Exception as e:
        print(f"error loading model: {e}")
        return
    
    try:
        pMask = get_pMask_jaap().to(DEVICE)
        sim_old = E2E_PhospheneSimulator_jaap(pMask=pMask, device=DEVICE)
        sim_old.eval()
        print("regular loaded")
    except:
        sim_old = None
        print("lỗi regular")

    sim_new = BioSimulator(device=DEVICE, grid_shape=(32,32), output_size=(IMG_SIZE, IMG_SIZE))
    sim_new.eval()
    print("Bio loaded")

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
    real_params, norm_params = generate_and_norm_params(DEVICE)
    print(f"Testing with params: Rho={real_params[0,0].item()}, Rot={real_params[0,12].item()}")

    with torch.no_grad():
        # encoder nhận normalized params
        encoder_out = model(input_tensor, patient_params=norm_params)
        
        if sim_old:
            out_old = sim_old(encoder_out)
        else:
            out_old = torch.zeros_like(input_tensor)

        # simulator mới nhận real params
        out_new = sim_new(encoder_out, patient_params=real_params)
    
    enc_img = encoder_out.squeeze().cpu().numpy()
    old_img = out_old.squeeze().cpu().numpy()
    new_img = out_new.squeeze().cpu().numpy()

    plt.figure(figsize=(20, 6))

    plt.subplot(1, 4, 1)
    plt.title("Input")
    plt.imshow(original_img)
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("Encoder Output")
    plt.imshow(enc_img, cmap='viridis', interpolation='nearest')
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 4, 3)
    plt.title("Regular")
    plt.imshow(old_img, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("Bio")
    plt.imshow(new_img, cmap='gray', vmin=0, vmax=1, origin='lower')
    plt.axis('off')

    output_folder = "phosphenes_result"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    output_filename = f"comparison_{timestamp}.png"
    save_path = os.path.join(output_folder, output_filename)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f" lưu tại {save_path}")
    
    plt.show()

if __name__ == "__main__":
    try:
        visualize_phosphene()
    except Exception as e:
        print(f"error{e}")
        import traceback
        traceback.print_exc()