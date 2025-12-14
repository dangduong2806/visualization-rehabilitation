import matplotlib.pyplot as plt
import numpy as np
import torch
import noise
from configs import FIGS_DIR, SIMULATOR

####### REPORT FUNCTIONS 
def visualizeBatch(imgs, activMaps, spvs, plt, ep):
  
  f, axarr = plt.subplots(5,3)

  for i in range(0,5):
      
    img = (imgs[i].clamp_(0.0, 1.0).detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    # repr = (activMaps[i].clamp_(0.0, 1.0).detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    if SIMULATOR != "biological": activeMap = (activMaps[i].clamp_(0.0, 1.0).detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    spv = (spvs[i].clamp_(0.0, 1.0).detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    # img = torch.reshape(torch.squeeze(outputs), (56, 56, 3)).detach().numpy()
    axarr[i,0].imshow(img)
    axarr[i, 1].imshow(spv, cmap="gray")

    if SIMULATOR != "biological": axarr[i, 2].imshow(activeMap, cmap="gray")
    
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


