import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

from modules.bio_simulator import BioSimulator
from configs import IMG_SIZE

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hàm tiện ích để tạo các tham số giả lập bệnh nhân
def create_dummy_params(batch_size, device, rot=0, rho=2.0, x_shift=0, y_shift=0):
    # Tạo vector 13 tham số
    # [rho, lambda, omega, a0, a1, a2, a3, a4, OD_x, OD_y, x_imp, y_imp, rot]
    params = torch.zeros(batch_size, 13, device=device)
    
    params[:, 0] = rho      # Rho
    params[:, 3] = 0.0      # a0 (bias độ sáng)
    params[:, 4] = 1.0      # a1 (linear scaling)
    params[:, 10] = x_shift # x implant
    params[:, 11] = y_shift # y implant
    params[:, 12] = rot     # rotation
    
    return params

#test cấu trúc simulator

def create_shape(shape_type="square"):
    img = np.zeros((32, 32), dtype=np.float32)
    
    if shape_type == "square":
        img[8:24, 8:24] = 1.0
        
    elif shape_type == "circle":
        cv2.circle(img, (16, 16), 10, 1.0, -1)
        
    elif shape_type == "bar":
        img[5:27, 14:18] = 1.0

    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)

def check_shapes():
    
    sim = BioSimulator(device=DEVICE, grid_shape=(32, 32), output_size=(IMG_SIZE, IMG_SIZE))
    
    shapes = ["square", "circle", "bar"]

    # Tạo tham số thử nghiệm: Xoay 30 độ và dịch chuyển nhẹ
    # Để kiểm chứng xem Simulator có thực sự xoay implant không
    test_params = create_dummy_params(1, DEVICE, rot=30, x_shift=100, rho=3.0)
    
    plt.figure(figsize=(10, 8))
    
    for i, shape_name in enumerate(shapes):
        input_tensor = create_shape(shape_name)
        
        with torch.no_grad():
            # Truyền thêm patient params vào simulator
            output_tensor = sim(input_tensor, patient_params=test_params)
            
        plt.subplot(3, 2, i*2 + 1)
        plt.imshow(input_tensor.squeeze().cpu().numpy(), cmap='gray')
        plt.title(f"Encoder Output: {shape_name}")
        plt.axis('off')
        
        plt.subplot(3, 2, i*2 + 2)
        plt.imshow(output_tensor.squeeze().cpu().numpy(), cmap='gray', origin='lower')
        plt.title(f"Bio-Sim Perception")
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig("check_geometry.png")
    plt.show()

if __name__ == "__main__":
    check_shapes()