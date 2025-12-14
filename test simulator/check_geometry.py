import torch
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "./test_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

from modules.bio_simulator import BioSimulatorHILO, create_default_phi, create_random_phi
from modules.bio_simulatorver0 import BioSimulator


def create_test_input(shape_type="square", device='cpu'):
    img = np.zeros((32, 32), dtype=np.float32)
    
    if shape_type == "square":
        img[8:24, 8:24] = 1.0
    elif shape_type == "circle":
        y, x = np.ogrid[:32, :32]
        mask = (x - 16)**2 + (y - 16)**2 <= 10**2
        img[mask] = 1.0
    elif shape_type == "cross":
        img[14:18, 4:28] = 1.0
        img[4:28, 14:18] = 1.0
    elif shape_type == "letter_T":
        img[4:8, 4:28] = 1.0
        img[4:28, 14:18] = 1.0
    elif shape_type == "diagonal":
        for i in range(32):
            if i < 32:
                img[i, max(0, i-2):min(32, i+3)] = 1.0
        
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)


def test_basic_comparison():
    print("="*60)
    print("TEST 1: So sánh original vs HILO (default params)")
    print("="*60)
    
    device = torch.device('cpu')
    
    sim_original = BioSimulator(device=device)
    sim_hilo = BioSimulatorHILO(device=device)
    
    shapes = ["square", "circle", "cross", "letter_T"]
    
    fig, axes = plt.subplots(len(shapes), 3, figsize=(12, 16))
    
    for i, shape in enumerate(shapes):
        stim = create_test_input(shape, device)
        phi_default = create_default_phi(batch_size=1, device=device)
        
        with torch.no_grad():
            out_original = sim_original(stim)
            out_hilo = sim_hilo(stim, phi_default)
        
        diff = torch.abs(out_original - out_hilo).max().item()
        
        axes[i, 0].imshow(stim.squeeze().numpy(), cmap='gray')
        axes[i, 0].set_title(f"Input: {shape}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(out_original.squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f"Original\nmax={out_original.max():.3f}")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(out_hilo.squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title(f"HILO (default)\nmax={out_hilo.max():.3f}\ndiff={diff:.6f}")
        axes[i, 2].axis('off')
        
        print(f"  {shape}: max_diff = {diff:.8f} {'✓ OK' if diff < 0.01 else '✗ KHÁC'}")
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/test1_basic_comparison.png", dpi=150)
    plt.close()
    print(f"\nĐã lưu: {OUTPUT_DIR}/test1_basic_comparison.png")


def test_geometry_effects():
    print("\n" + "="*60)
    print("TEST 2: Hiệu ứng hình học của HILO")
    print("="*60)
    
    device = torch.device('cpu')
    sim_hilo = BioSimulatorHILO(device=device)
    
    stim = create_test_input("cross", device)
    
    configs = []
    
    phi_default = create_default_phi(1, device)
    configs.append(("Default", phi_default))
    
    phi_rot15 = create_default_phi(1, device)
    phi_rot15[0, 2] = 15.0
    configs.append(("Xoay 15°", phi_rot15))
    
    phi_rot_neg15 = create_default_phi(1, device)
    phi_rot_neg15[0, 2] = -15.0
    configs.append(("Xoay -15°", phi_rot_neg15))
    
    phi_shift_x = create_default_phi(1, device)
    phi_shift_x[0, 0] = 1.0
    configs.append(("Dịch X +1mm", phi_shift_x))
    
    phi_shift_y = create_default_phi(1, device)
    phi_shift_y[0, 1] = 1.0
    configs.append(("Dịch Y +1mm", phi_shift_y))
    
    phi_combo = create_default_phi(1, device)
    phi_combo[0, 0] = 0.5
    phi_combo[0, 1] = 0.5
    phi_combo[0, 2] = 10.0
    configs.append(("Xoay 10° + Dịch", phi_combo))
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    axes[0].imshow(stim.squeeze().numpy(), cmap='gray')
    axes[0].set_title("INPUT: Cross")
    axes[0].axis('off')
    
    for idx, (name, phi) in enumerate(configs):
        with torch.no_grad():
            out = sim_hilo(stim, phi)
        
        axes[idx + 1].imshow(out.squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[idx + 1].set_title(f"{name}")
        axes[idx + 1].axis('off')
        
        print(f"  {name}: output range [{out.min():.3f}, {out.max():.3f}]")
    
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/test2_geometry_effects.png", dpi=150)
    plt.close()
    print(f"\nĐã lưu: {OUTPUT_DIR}/test2_geometry_effects.png")


def test_scale_effects():
    print("\n" + "="*60)
    print("TEST 3: scale parameters")
    print("="*60)
    
    device = torch.device('cpu')
    sim_hilo = BioSimulatorHILO(device=device)
    
    stim = create_test_input("square", device)
    
    params_to_test = [
        ("spread_scale", 3),
        ("brightness_scale", 4),
        ("size_scale", 5),
    ]
    
    scale_values = [0.5, 1.0, 2.0]
    
    fig, axes = plt.subplots(len(params_to_test), len(scale_values) + 1, figsize=(14, 10))
    
    for i, (param_name, param_idx) in enumerate(params_to_test):
        axes[i, 0].imshow(stim.squeeze().numpy(), cmap='gray')
        axes[i, 0].set_title(f"Input")
        axes[i, 0].axis('off')
        
        for j, scale in enumerate(scale_values):
            phi = create_default_phi(1, device)
            phi[0, param_idx] = scale
            
            with torch.no_grad():
                out = sim_hilo(stim, phi)
            
            axes[i, j + 1].imshow(out.squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[i, j + 1].set_title(f"{param_name}={scale}")
            axes[i, j + 1].axis('off')
        
        print(f"  Tested: {param_name}")
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/test3_scale_effects.png", dpi=150)
    plt.close()
    print(f"\nĐã lưu: {OUTPUT_DIR}/test3_scale_effects.png")


def test_random_patients():
    print("\n" + "="*60)
    print("TEST 4: Batch với random patient params")
    print("="*60)
    
    device = torch.device('cpu')
    sim_hilo = BioSimulatorHILO(device=device)
    
    batch_size = 4
    
    stim_single = create_test_input("circle", device)
    stim_batch = stim_single.expand(batch_size, -1, -1, -1)
    
    phi_batch = create_random_phi(batch_size, device, seed=42)
    
    print("\nPatient parameters:")
    print(f"  x_shift:    {[f'{x:.2f}' for x in phi_batch[:, 0].tolist()]}")
    print(f"  y_shift:    {[f'{x:.2f}' for x in phi_batch[:, 1].tolist()]}")
    print(f"  rotation:   {[f'{x:.1f}' for x in phi_batch[:, 2].tolist()]}")
    print(f"  brightness: {[f'{x:.2f}' for x in phi_batch[:, 4].tolist()]}")
    
    with torch.no_grad():
        out_batch = sim_hilo(stim_batch, phi_batch)
    
    fig, axes = plt.subplots(2, batch_size + 1, figsize=(16, 8))
    
    for i in range(batch_size + 1):
        if i == 0:
            axes[0, i].imshow(stim_single.squeeze().numpy(), cmap='gray')
            axes[0, i].set_title("INPUT\n(same for all)")
        else:
            axes[0, i].imshow(stim_single.squeeze().numpy(), cmap='gray')
            axes[0, i].set_title(f"Patient {i}")
        axes[0, i].axis('off')
    
    axes[1, 0].text(0.5, 0.5, "OUTPUTS\n(different)", ha='center', va='center', fontsize=12)
    axes[1, 0].axis('off')
    
    for i in range(batch_size):
        axes[1, i + 1].imshow(out_batch[i].squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[1, i + 1].set_title(f"rot={phi_batch[i, 2]:.1f}°\nshift=({phi_batch[i, 0]:.2f}, {phi_batch[i, 1]:.2f})")
        axes[1, i + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/test4_random_patients.png", dpi=150)
    plt.close()
    print(f"\nĐã lưu: {OUTPUT_DIR}/test4_random_patients.png")


def test_full_comparison():
    print("\n" + "="*60)
    print("TEST TỔNG HỢP: Input | Original | HILO")
    print("="*60)
    
    device = torch.device('cpu')
    
    sim_original = BioSimulator(device=device)
    sim_hilo = BioSimulatorHILO(device=device)
    
    shapes = ["square", "circle", "cross", "letter_T", "diagonal"]
    
    phi_varied = create_default_phi(1, device)
    phi_varied[0, 2] = 10.0   # xoay 10 độ
    phi_varied[0, 4] = 1.2    # brightness_scale = 1.2
    phi_varied[0, 5] = 0.8    # size_scale = 0.8
    
    fig, axes = plt.subplots(len(shapes), 3, figsize=(12, 20))
    
    fig.suptitle("So sánh: Input | Original Simulator | HILO (xoay 10°, brightness×1.2, size×0.8)", 
                 fontsize=14, fontweight='bold')
    
    for i, shape in enumerate(shapes):
        stim = create_test_input(shape, device)
        
        with torch.no_grad():
            out_original = sim_original(stim)
            out_hilo = sim_hilo(stim, phi_varied)
        
        # Input
        axes[i, 0].imshow(stim.squeeze().numpy(), cmap='gray')
        axes[i, 0].set_title(f"Input: {shape}" if i == 0 else shape)
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_ylabel("INPUT", fontsize=12, fontweight='bold')
        
        # Original
        axes[i, 1].imshow(out_original.squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title("Original" if i == 0 else "")
        axes[i, 1].axis('off')
        
        # HILO
        axes[i, 2].imshow(out_hilo.squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title("HILO (varied)" if i == 0 else "")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/test_full_comparison.png", dpi=150)
    plt.close()
    print(f"\nĐã lưu: {OUTPUT_DIR}/test_full_comparison.png")

if __name__ == "__main__":
    print("checking hilo hehe\n")
    
    test_basic_comparison()
    
    test_geometry_effects()
    
    test_scale_effects()
    
    test_random_patients()
    
    test_full_comparison()
    
    print("\n" + "="*60)
    print("done")
    print("="*60)
    print(f"\nCác file output trong thư mục: {OUTPUT_DIR}/")
    print("  - test1_basic_comparison.png")
    print("  - test2_geometry_effects.png")
    print("  - test3_scale_effects.png")
    print("  - test4_random_patients.png")
    print("  - test_full_comparison.png")