import torch
import random

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

NUM_PARAMS = len(PARAM_RANGES)

def generate_random_params(batch_size, device):
    """
    Sinh tham số ngẫu nhiên (Giá trị thực - Physical Values)
    Output shape: (Batch_Size, 13)
    """
    params_list = []
    keys = ['rho', 'lambda', 'omega', 'a0', 'a1', 'a2', 'a3', 'a4', 
            'od_x', 'od_y', 'x_imp', 'y_imp', 'rot']
    for k in keys:
        low, high = PARAM_RANGES[k]
        # sinh random unifrorm trong khoảng [low, high] 
        val = torch.rand(batch_size, 1, device=device) * (high - low) + low
        params_list.append(val)
    
    params = torch.cat(params_list, dim=1)  # shape: (Batch_Size, 13)
    return params

def normalize_params(params_tensor):
    """
    Chuẩn hóa tham số về khoảng [0, 1] để đưa vào Encoder
    Input: Tensor giá trị thực (Batch, 13)
    """

    keys = ['rho', 'lambda', 'omega', 'a0', 'a1', 'a2', 'a3', 'a4', 
            'od_x', 'od_y', 'x_imp', 'y_imp', 'rot']
    normalized_list = []
    for i, k in enumerate(keys):
        low, high = PARAM_RANGES[k]
        normalized_val = (params_tensor[:, i:i+1] - low) / (high - low)
        normalized_list.append(normalized_val)
    normalized_params = torch.cat(normalized_list, dim=1)
    return normalized_params
