# This code has been developed at the Department of Machine Learning and Neural Computing, Donders Institute of Brain
# 
###############
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torchvision.transforms as T

# LOAD MODULES AND GLOBAL VARIABLES
from modules.models import SightedRecognizer, BlindRecognizer_Feat, BlindRecognizer_Classifier, Representer, E2E_PhospheneSimulator_jaap, divideModel, Simulator_exp4
from configs import BATCH_SIZE, LAMBDA, LEARNING_RATE, EPOCHS, LOG_ON_BATCH, DATASET, FEAT_LAYERS_N, NUM_OF_CLASSES, SIMULATOR, EXEC_UNIT, MODELS_DIR, LOG_DIR, IMG_SIZE
from modules.dataloaders import imageNettePrep, CocoPrep
from modules.utils import *

# import tqdm
import tqdm

# IMPORT THE NEW FUNCTION
from utils import generate_random_params, normalize_params

from modules.bio_simulator import BioSimulatorHILO, generate_random_params, normalize_params

DEVICE = torch.device('cuda:0')


########### DATA LOADING

# Initialize the datasets
if DATASET == "imageNette":
    trainDataset, validDataset = imageNettePrep()
elif DATASET == "COCO":
    trainDataset, validDataset = CocoPrep()
else:
    print("wrong dataset name in the config file")


img, target = validDataset[0]  # test reading one sample

# Initialize data loaders
train_dataloader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(validDataset, batch_size=BATCH_SIZE, shuffle=True)



######### INITIALIZAING THE MODELS

resNetModel = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
resNetModel2 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")

# Create Sighted Unit useing first N layers of resNet
sightedUnit = SightedRecognizer(resNetModel, FEAT_LAYERS_N)

# freeze the weights Sighted Unit to be used for perceptual loss
for param in sightedUnit.parameters():
    param.requires_grad = False

encoder = Representer(
    n_channels=3,
    n_patient_params=13,
    arryaOut=(SIMULATOR=="biological"))   # initialize the encoder

# divide a resnet model into first N layers and after that. Use them to create to parts of the blind unit
blindUnit_feat_pre, blindUnit_classifier_pre = divideModel(resNetModel2, FEAT_LAYERS_N)
blindUnit_feat = BlindRecognizer_Feat(blindUnit_feat_pre)    
blindUnit_classifier = BlindRecognizer_Classifier(blindUnit_classifier_pre, FEAT_LAYERS_N, NUM_OF_CLASSES)

# Initialize the simulator (from Jaap de ruyter van steveninck et al. 2022)


if SIMULATOR=="regular":
    pMask = get_pMask_jaap()
    simulator = E2E_PhospheneSimulator_jaap(pMask=pMask, device=DEVICE)
elif SIMULATOR=="biological":
    # --- CẬP NHẬT MỚI ---
    # Khởi tạo simulator chuẩn HILO (Human-In-The-Loop / Patient-Specific)
    simulator = BioSimulatorHILO(
        device=DEVICE, 
        grid_shape=(32, 32), 
        output_size=(IMG_SIZE, IMG_SIZE)
    )
    print("Đã khởi tạo BioSimulatorHILO với Patient Parameters.")

# log the number of parameters for training
print("Num of params for sightedUnit:")
print(count_parameters(sightedUnit))
print("Num of params for encoder:")
print(count_parameters(encoder))
print("Num of params for blindUnit_feat:")
print(count_parameters(blindUnit_feat))
print("Num of params for blindUnit_classifier:")
print(count_parameters(blindUnit_classifier))


############# TRAINNG
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

optimizerencoder = torch.optim.Adam(list(encoder.parameters()) + list(blindUnit_feat.parameters()), LEARNING_RATE)
optimizerAll = torch.optim.Adam(list(encoder.parameters()) + list(blindUnit_feat.parameters()) + list(blindUnit_classifier.parameters()), LEARNING_RATE)

CE_loss = nn.CrossEntropyLoss().cuda()
RECON_loss = nn.MSELoss().cuda()

if EXEC_UNIT=="GPU":
    sightedUnit.cuda()
    encoder.cuda()
    blindUnit_classifier.cuda()
    blindUnit_feat.cuda()
    if SIMULATOR!="biological":
        simulator.cuda()

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(train_dataloader) instead of
    # iter(train_dataloader) so that we can track the batch
    # index and do some intra-epoch reporting
    loop = tqdm.tqdm(train_dataloader, desc="training", colour="blue")
    for i, data in enumerate(loop):
        # Every data instance is an input + label pair

        inputs, target = data
        
        if EXEC_UNIT == "GPU":
            inputs = inputs.cuda()
            # target = target.cuda()
            target_lbl = target["label"].cuda()
            target_mi = target["masked_img"].cuda()
            target_gaze = target["gaze"].cuda()
        else: 
            target_lbl = target["label"]
            target_mi = target["masked_img"]
            target_gaze = target["gaze"]
        
        # Sinh patient parameters ngẫu nhiên cho batch
        batch_size = inputs.shape[0]
        real_params = generate_random_params(batch_size, device=DEVICE)  # shape: (BATCH_SIZE, 13)
        # Chuẩn hóa patient parameters
        norm_params = normalize_params(real_params)  # shape: (BATCH_SIZE, 13)

        # Zero your gradients for every batch!
        optimizerencoder.zero_grad()
        optimizerAll.zero_grad()

        # Get sighted Features
        featMap = sightedUnit(target_mi)
        
        # Truyền norn_params vào encoder
        repr_org= encoder(inputs, patient_params=norm_params)

        
        # Cần truyền cả patient_params vào simulator
        # if SIMULATOR == "interpol":
        #     spv = T.Resize(size=IMG_SIZE)(repr_org)
        # else:
        #     spv = simulator(repr_org)

        if SIMULATOR == "interpol":
            # Chế độ cũ (không dùng params)
            spv = T.Resize(size=IMG_SIZE)(repr_org)
        else:
            # Dynaphos Simulator: Cần nhận kích thích + tham số vật lý
            # Lưu ý: Bạn cần đảm bảo hàm forward của simulator chấp nhận tham số này
            spv = simulator(repr_org, phi=real_params)

        blindVision = blindUnit_feat(spv)
    
        spatialLoss =  LAMBDA * RECON_loss(blindVision, featMap)

        #get gradient of encoder
        spatialLoss.backward(retain_graph=True)

        # Get class prediction of representation from limited-class VGGnet
        blindPred = blindUnit_classifier(blindVision)

        # Compute the loss and its gradients
        CEloss = (1-LAMBDA) * CE_loss(blindPred, target_lbl)

        # gradient of all network
        CEloss.backward()

        # Adjust learning weights
        optimizerencoder.step()
        optimizerAll.step()
        
        loss = CEloss + spatialLoss

        running_loss += loss.item()
        if i % LOG_ON_BATCH == (LOG_ON_BATCH-1):
            last_loss = running_loss / LOG_ON_BATCH

            # print('  batch {} loss: {}'.format(i + 1, last_loss))
            loop.set_postfix(loss=last_loss)

            tb_x = epoch_index * len(train_dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
            

    return last_loss
    


epoch_number = 0
best_vloss = 1_000_000.
visSetFlag = False
train_losses = []
valid_losses = []

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    encoder.train(True)
    sightedUnit.train(True)
    blindUnit_feat.train(True)
    blindUnit_classifier.train(True)

    avg_loss = train_one_epoch(epoch_number, writer)
    
    train_losses.append(avg_loss)

    # We don't need gradients on to do reporting
    encoder.train(False)
    sightedUnit.train(False)
    blindUnit_feat.train(False)
    blindUnit_classifier.train(False)

    running_vloss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for i, vdata in enumerate(validation_dataloader):
        
            vinputs, vtarget = vdata
            if EXEC_UNIT == "GPU":
                vinputs = vinputs.cuda()
                # target = target.cuda()
                vtarget_lbl = vtarget["label"].cuda()
                vtarget_mi = vtarget["masked_img"].cuda()
                vtarget_gaze = target["gaze"].cuda()
            else: 
                vtarget_lbl = vtarget["label"]
                vtarget_mi = vtarget["masked_img"]
                vtarget_gaze = target["gaze"]

            if not visSetFlag:
                vis_imgs = vinputs
                visSetFlag = True

            # Sinh patient parameters ngẫu nhiên cho batch
            vbatch_size = vinputs.shape[0]
            real_params = generate_random_params(vbatch_size, device=DEVICE)  # shape: (BATCH_SIZE, 13)
            # Chuẩn hóa patient parameters
            norm_params = normalize_params(real_params)  # shape: (BATCH_SIZE, 13)

            featMap = sightedUnit(vtarget_mi)
            repr_org = encoder(vinputs, patient_params=norm_params)
            # spv = simulator(repr_org)

            # if not SIMULATOR:
            #     spv = T.Resize(size=IMG_SIZE)(repr_org)
            # else:
            #     spv = simulator(repr_org)

            # --- 3. SIMULATOR FORWARD (CÓ PARAM) ---
            if SIMULATOR == "interpol":
                # Chế độ cũ (không dùng params)
                spv = T.Resize(size=IMG_SIZE)(repr_org)
            else:
                # Dynaphos Simulator: Cần nhận kích thích + tham số vật lý
                # Lưu ý: Bạn cần đảm bảo hàm forward của simulator chấp nhận tham số này
                spv = simulator(repr_org, phi=real_params)
                    
            blindVision = blindUnit_feat(spv)
            blindPred = blindUnit_classifier(blindVision)
            spatialLoss =  LAMBDA * RECON_loss(blindVision, featMap)
        
            # spatialLoss =  (LAMBDA * RECON_loss(torch.mean(repr, 1), torch.mean(vggFeatMap,1)))
            CEloss = (1-LAMBDA) * CE_loss(blindPred, vtarget_lbl)

            vloss = spatialLoss + CEloss
            
            vacc = torch.sum(torch.argmax(blindPred, dim=1) == vtarget_lbl)/BATCH_SIZE

            running_vloss += vloss
            running_acc += vacc
            
        avg_vloss = running_vloss / (i + 1)
        avg_vacc = running_acc / (i + 1)
        valid_losses.append(avg_vloss)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        with open(LOG_DIR, 'a') as file:
            file.write('LOSS train {} valid {}, valid-acc {} \n'.format(avg_loss, avg_vloss, avg_vacc))
        # saving the spv
        repr_org = encoder(vis_imgs)
        print(repr_org.shape)
        spv = simulator(repr_org)
            
        visualizeBatch(vis_imgs, repr_org, spv, plt, epoch_number)

        # loss curve visualization
        # plt.plot(list(range(0,epoch+1)), train_losses, label='training loss')
        # plt.plot(list(range(0,epoch+1)), valid_losses, label='validation loss')
        # plt.title('Training and Validation Loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.savefig('loss_curve.png')
        # plt.close()
        
        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            enc_path = MODELS_DIR + '/enc2_{}_{}'.format(timestamp, epoch_number)
            torch.save(encoder.state_dict(), enc_path)

            blindUnit_feat_path = MODELS_DIR + '/blindUnit_feat_{}_{}'.format(timestamp, epoch_number)
            torch.save(blindUnit_feat.state_dict(), blindUnit_feat_path)

            blindUnit_clsfr_path = MODELS_DIR + '/blindUnit_clsfr_{}_{}'.format(timestamp, epoch_number)
            torch.save(blindUnit_classifier.state_dict(), blindUnit_clsfr_path)


    epoch_number += 1