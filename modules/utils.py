import matplotlib.pyplot as plt
import numpy as np
import torch
import noise
from configs import FIGS_DIR, SIMULATOR

####### REPORT FUNCTIONS 
def visualizeBatch(imgs, activMaps, spvs, plt, ep):
    # 1. Lấy số lượng ảnh thực tế trong batch (Ví dụ: 4)
    n_samples = imgs.shape[0]
    
    # 2. Giới hạn chỉ vẽ tối đa 5 ảnh để hình không quá to (nếu batch là 32 thì chỉ vẽ 5)
    num_rows = min(n_samples, 5)
    
    # 3. Tạo subplot động
    # squeeze=False để đảm bảo axarr luôn là mảng 2 chiều [row, col] ngay cả khi num_rows=1
    f, axarr = plt.subplots(num_rows, 3, figsize=(10, 3 * num_rows), squeeze=False)

    for i in range(num_rows):
        # --- Cột 1: Input Image ---
        img = (imgs[i].clamp(0.0, 1.0).detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        axarr[i, 0].imshow(img)
        axarr[i, 0].axis('off') # Tắt trục tọa độ cho đẹp
        if i == 0: axarr[i, 0].set_title("Input")

        # --- Cột 2: Simulator Output ---
        spv = (spvs[i].clamp(0.0, 1.0).detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        axarr[i, 1].imshow(spv, cmap="gray")
        axarr[i, 1].axis('off')
        if i == 0: axarr[i, 1].set_title("Sim Output")

        # --- Cột 3: Activation Map (Chỉ cho Regular Simulator) ---
        if SIMULATOR != "biological":
            try:
                # Chỉ vẽ nếu activMaps là dạng ảnh (C, H, W). Biological output là vector (1024) nên ko vẽ được
                activeMap = (activMaps[i].clamp(0.0, 1.0).detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
                axarr[i, 2].imshow(activeMap, cmap="gray")
            except Exception:
                pass # Bỏ qua nếu lỗi kích thước
        
        axarr[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR + '/res_'+str(ep)+'.png')
    plt.close()



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_pMask_jaap(size=(256,256),phosphene_density=32,seed=1,
              jitter_amplitude=0., intensity_var=0.,
              dropout=False,perlin_noise_scale=.4):

    # Define resolution and phosphene_density
    [nx,ny] = size
    n_phosphenes = phosphene_density**2 # e.g. n_phosphenes = 32 x 32 = 1024
    pMask = torch.zeros(size)


    # Custom 'dropout_map'
    p_dropout = perlin_noise_map(shape=size,scale=perlin_noise_scale*size[0],seed=seed)
    np.random.seed(seed)

    for p in range(n_phosphenes):
        i, j = divmod(p, phosphene_density)
       
        jitter = np.round(np.multiply(np.array([nx,ny])//phosphene_density,
                                      jitter_amplitude * (np.random.rand(2)-.5))).astype(int)
        rx = (j*nx//phosphene_density) + nx//(2*phosphene_density) + jitter[0]
        ry = (i*ny//phosphene_density) + ny//(2*phosphene_density) + jitter[1]

        rx = np.clip(rx,0,nx-1)
        ry = np.clip(ry,0,ny-1)
        
        intensity = intensity_var*(np.random.rand()-0.5)+1.
        if dropout==True:
            pMask[rx,ry] = np.random.choice([0.,intensity], p=[p_dropout[rx,ry],1-p_dropout[rx,ry]])
        else:
            pMask[rx,ry] = intensity
            
    return pMask    

def perlin_noise_map(seed=0,shape=(256,256),scale=100,octaves=6,persistence=.5,lacunarity=2.):
    out = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            out[i][j] = noise.pnoise2(i/scale, 
                                        j/scale, 
                                        octaves=octaves, 
                                        persistence=persistence, 
                                        lacunarity=lacunarity, 
                                        repeatx=shape[0], 
                                        repeaty=shape[1], 
                                        base=seed)
    out = (out-out.min())/(out.max()-out.min())
    return out


