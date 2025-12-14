import torch
from torch import nn
import torch.nn.functional as F
import math


######### MODELS DEFINITIONS
class SightedRecognizer(nn.Module):
    def __init__(self, Model, numOfLayers):
        super(SightedRecognizer, self).__init__()
        self.features = nn.Sequential(*list(Model.children())[:numOfLayers])
        
    def forward(self, x):
        x = self.features(x)
        return x


class BlindRecognizer_Feat(nn.Module):
    def __init__(self, Model):
        super(BlindRecognizer_Feat, self).__init__()
        # self.features = nn.Sequential(*list(Model.children())[:numOfLayers])
        self.feature = Model
        self.channelAdj = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1)
    def forward(self, x):
        # x = x.type(torch.LongTensor)
        x = self.channelAdj(x)
        x = self.feature(x)
        # x = self.softmax(x)
        return x


class BlindRecognizer_Classifier(nn.Module):
    def __init__(self, Model, numOfLayers, numOfOutputs):
        super(BlindRecognizer_Classifier, self).__init__()

        # self.classifierModel = nn.Sequential(*list(Model.children())[numOfLayers:])
        self.classifierModel = Model
        self.classChooser = nn.Linear(1000, numOfOutputs)
        # self.softmax = torch.nn.Softmax()
        
    def forward(self, x):
        x = self.classifierModel(x)
        x = self.classChooser(x)
        # x = self.softmax(x)
        return x

# U-NET
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Dùng Bilinear upsampling thay vì TransposeConv để giảm checkerboard artifacts (thường tốt hơn cho bài toán này)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Xử lý trường hợp kích thước không khớp do padding (nếu có)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Skip connection: Nối x2 (từ encoder) vào x1 (từ decoder)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

### UPDATE ENCODER
class Representer(nn.Module):
    # Thêm n_patient_params vào init để rõ ràng hơn, mặc định 13
    def __init__(self, n_channels=3, n_patient_params=13, n_classes=1, arryaOut=False):
        super(Representer, self).__init__()
        # Tổng số kênh đầu vào thực tế
        total_input_channels = n_channels + n_patient_params

        self.n_channels = n_channels
        self.n_patient_params = n_patient_params
        self.n_classes = n_classes
        self.arryaOut = arryaOut

        # --- ENCODER (Contracting Path) ---
        self.inc = DoubleConv(total_input_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512) # Có thể mở rộng nếu ảnh lớn (256x256 trở lên)

        # --- DECODER (Expanding Path) ---
        self.up1 = Up(512+256, 256)
        self.up2 = Up(256 + 128, 128) # +128 do nối với output của down2
        self.up3 = Up(128 + 64, 64)   # +64 do nối với output của down1
        self.up4 = Up(64 + 32, 32)    # +32 do nối với output của inc

        # CHỈ CHO SIMULATOR HIỆN TẠI
        # --- THÊM MỚI: Lớp ép kích thước về 32x32 --- 
        # Bất kể ảnh đầu vào là 224x224 hay 256x256, qua đây đều thành 32x32
        self.final_pool = nn.AdaptiveAvgPool2d((32, 32)) 
        # --------------------------------------------
        
        # Output layer
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

        # --- Xử lý cho trường hợp arryaOut=True (Biological setting) ---
        # Nếu output là vector, ta dùng Global Average Pooling để giảm chiều
        if self.arryaOut:
            self.flat = nn.Flatten()
            # Giả sử kích thước ảnh chuẩn, cần tính toán lại in_features nếu đổi size ảnh
            # Tuy nhiên, cách an toàn là dùng AdaptiveAvgPool trước khi Flatten
            self.gap = nn.AdaptiveAvgPool2d((4, 4)) 
            self.lin1 = nn.Linear(32 * 4 * 4, 1024) 
            self.lin2 = nn.Linear(1024, 1024)

        self.tan = nn.Tanh()

    def expand_params(self, params, target_shape):
        """
        Biến vector (B, 13) -> Feature Maps (B, 13, H, W)
        """
        H, W = target_shape[2], target_shape[3]
        params_reshaped = params.view(params.size(0), params.size(1), 1, 1)
        params_expanded = params_reshaped.expand(-1, -1, H, W)
        return params_expanded

    # update forward để thêm patient params
    def forward(self, x, patient_params):
        """
        x: Ảnh đầu vào (Batch, 3, H, W)
        patient_params: Vector tham số ĐÃ CHUẨN HÓA (Batch, 13)
        """
        # Mở rộng patient_params thành feature maps
        param_maps = self.expand_params(patient_params, x.shape)
        # Concatenate ảnh với param maps
        x_input = torch.cat([x, param_maps], dim=1)  # Kích thước: (Batch, 3+13, H, W)

        # 1. Encoder
        x1 = self.inc(x_input)        # Giữ lại x1 để skip connection
        x2 = self.down1(x1)     # Giữ lại x2
        x3 = self.down2(x2)     # Giữ lại x3
        x4 = self.down3(x3)     # Đáy chữ U (Bottleneck)
        x5 = self.down4(x4)

        # 2. Decoder (với Skip Connections)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)    # Nối với x3
        x = self.up3(x, x2)     # Nối với x2
        x = self.up4(x, x1)     # Nối với x1
        
        # 3. Output logic
        if not self.arryaOut:
            # Resize về 32x32
            x = self.final_pool(x) 
            # Tại đây, kích thước x là [Batch, 32, 32, 32]
            
            # Regular Grid Simulation: Trả về ảnh (Batch, 1, H, W)
            x = self.outc(x)
        else:
            # Biological Simulation: Trả về vector
            # Lưu ý: U-Net thường output ảnh, nên nếu cần vector ta lấy feature map cuối cùng
            x = self.gap(x) # [Batch, 32, 4, 4]
            x = self.flat(x)
            x = self.lin1(x)
            x = self.lin2(x)

        # 4. Activation & Binarization (Quan trọng: Giữ nguyên logic cũ)
        x = self.tan(x)
        # Straight-Through Estimator (STE) cho binarization
        x = x + torch.sign(x).detach() - x.detach()
        # Scale về [0, 1]
        x = 0.5 * (x + 1)
        
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, n_channels, stride=1, resample_out=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(n_channels)
        self.resample_out = resample_out
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        if self.resample_out:
            out = self.resample_out(out)
        return out
    


#### SIMULATOR MODEL FROM EXPERIMENT 4 of Jaap de Ruyter van Steveninck, 2022
class Simulator_exp4(object):
    """ Modular phosphene simulator that is used in experiment 4. Requires a predefined phosphene mapping. e.g. Tensor of 650 X 256 X 256 where 650 is the number of phosphenes and 256 X 256 is the resolution of the output image."""
    def __init__(self,pMap=None, device='cpu',pMap_from_file='/home/burkuc/viseon_a/training_configuration/phosphene_map_exp4.pt'):
        # Phospene mapping (should be of shape: n_phosphenes, res_x, res_y)
        if pMap is not None:
            self.pMap = pMap
        else:
            #self.pMap = torch.load(pMap_from_file, map_location=torch.device('cpu'))
            self.pMap = torch.load(pMap_from_file, map_location=torch.device(device))


        self.n_phosphenes = self.pMap.shape[0]
    
    def __call__(self,stim):
        return torch.einsum('ij, jkl -> ikl', stim, self.pMap).unsqueeze(dim=1) 

    def get_center_of_phosphenes(self):
        pMap = torch.nn.functional.interpolate(self.pMap.unsqueeze(dim=1),size=(128,128))  #650,1,128,128
        pLocs = pMap.view(self.n_phosphenes,-1).argmax(dim=-1) #650
        self.plocs = pLocs // 128, pLocs % 128 # y and x coordinates of the center of each phosphene
        return pLocs
    


    
class E2E_PhospheneSimulator_jaap(nn.Module):
    """ Uses three steps to convert  the stimulation vectors to phosphene representation:
    1. Resizes the feature map (default: 32x32) to SVP template (256x256)
    2. Uses pMask to sample the phosphene locations from the SVP activation template
    2. Performs convolution with gaussian kernel for realistic phosphene simulations
    """
    def __init__(self,pMask,scale_factor=8, sigma=1.5,kernel_size=11, intensity=15, device=torch.device('cuda:0')):
        super(E2E_PhospheneSimulator_jaap, self).__init__()
        
        # Device
        self.device = device
        
        # Phosphene grid
        self.pMask = pMask.to(self.device)
        self.up = nn.Upsample(mode="nearest",scale_factor=scale_factor)
        self.gaussian = self.get_gaussian_layer(kernel_size=kernel_size, sigma=sigma, channels=1)
        self.intensity = intensity 
    
    def get_gaussian_layer(self, kernel_size, sigma, channels):
        """non-trainable Gaussian filter layer for more realistic phosphene simulation"""

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = (1./(2.*math.pi*variance)) *\
                          torch.exp(
                              -torch.sum((xy_grid - mean)**2., dim=-1) /\
                              (2*variance)
                          )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels, bias=False)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False

        return gaussian_filter    

    def forward(self, stimulation):
        
        # Phosphene simulation
        phosphenes = self.up(stimulation)*self.pMask
        phosphenes = self.gaussian(F.pad(phosphenes, (5,5,5,5), mode='constant', value=0)) 
        return self.intensity*phosphenes    
    

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x
    

def divideModel(resNetModel2, FEAT_LAYERS_N):

    modules = list(resNetModel2.children())[:FEAT_LAYERS_N]
    blindUnit_feat_pre = nn.Sequential(*modules)

    modules2 = list(resNetModel2.children())[FEAT_LAYERS_N:-1]
    blindUnit_classifier_pre = nn.Sequential(*[*modules2, Flatten(), list(resNetModel2.children())[-1]])

    return blindUnit_feat_pre, blindUnit_classifier_pre