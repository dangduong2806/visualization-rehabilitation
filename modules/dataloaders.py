import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os
from skimage import io
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
from configs import IMG_SIZE, FILTERED_CATS, FOVEA_SIZE




###### CLASS DEFENTIONS
class ImageNetteDataset(Dataset):

    def __init__(self, root_dir, transform=None):


        self.root_dir = root_dir
        self.transform = transform
        self.map = os.listdir(os.path.join(root_dir))
        self.samplesTable = []
        for path, subdirs, files in os.walk(root_dir):
          for name in files:
            img_path = os.path.join(path, name)
            self.samplesTable.append(img_path)

        print(len(self.samplesTable))

    def __len__(self):
        return len(self.samplesTable)

    def __getitem__(self, idx):
      
        img_path = self.samplesTable[idx]
        
        image = io.imread(img_path)

        if len(image.shape) == 2:
          image = np.stack((image,)*3, axis=-1)

        image = self.transform(image)
        # Lấy tên thư mục cha của file ảnh
        class_name = os.path.basename(os.path.dirname(img_path))
        # Tìm index
        label = self.map.index(class_name)
        # label = self.map.index(img_path.split("/")[2])

        # Đóng gói target thành từ điển (Dictionary) để khớp với main.py
        target = {
            "label": torch.as_tensor(label),           # Nhãn phân loại
            "masked_img": image,                       # Dùng tạm ảnh gốc làm masked_img (vì không có ảnh mask)
            "gaze": torch.tensor([0.0, 0.0])           # Tạo gaze giả (tọa độ 0,0)
        }
      
        return image, target



class CocoDataset(Dataset):

  def __init__(self, root_dir, split, transform=None):

    from pycocotools.coco import COCO

    self.root_dir = root_dir
    self.transform = transform
    self.split = split
    self.coco = COCO(os.path.join(self.root_dir, "annotations", "instances_"+self.split+".json"))

    lum_img = Image.new('L',[FOVEA_SIZE,FOVEA_SIZE] ,0) 
    draw = ImageDraw.Draw(lum_img)
    draw.pieslice([(0,0),(FOVEA_SIZE,FOVEA_SIZE)],0,360,fill=1)
    focusCircle = np.array(lum_img)
    
    self.focusCircle = np.stack((focusCircle,)*3, axis=-1)

    self.ids = []
    self.targetCats = []
    self.labels = []
    for cat_i in range(len(FILTERED_CATS)):
      catIds = self.coco.getCatIds(catNms=FILTERED_CATS[cat_i])
      # self.ids = list(sorted(self.coco.imgs.keys()))
      ids_i = self.coco.getImgIds(catIds=catIds)
      self.ids.extend(ids_i)
      self.targetCats.extend(catIds*len(ids_i))
      self.labels.extend([cat_i]*len(ids_i))
    # print(self.labels)

  def _biggestObj (self, objects):
    max_area = 0
    max_o = 0
    for o in range(1,len(objects)):
      if objects[o]['area'] > max_area:
          max_o = o
          max_area = objects[o]['area'] 
    return max_o
  
  def _findObjSegIdx(self, objList, catId):
      for o in range(len(objList)):
        if objList[o]['category_id'] == catId:
            return o
      
  def __len__(self):
    return len(self.ids)

  def _imgfocus(self, img, center, crop=False):
    
    focRad = int(FOVEA_SIZE/2) # focus radius

    padded_img = np.pad(img, ((focRad, focRad), (focRad, focRad), (0,0)), 'constant', constant_values=0)
    
    center = (center[0] + focRad, center[1] + focRad)
    if crop:
       return padded_img[(center[0]-focRad): (center[0]+focRad), (center[1]-focRad): (center[1]+focRad), :] * self.focusCircle
    else:
      visionMask = np.zeros((padded_img.shape))
      visionMask[(center[0]-focRad): (center[0]+focRad), (center[1]-focRad): (center[1]+focRad), :] = self.focusCircle
      visionMask = visionMask.astype('uint8')
      focusedImg = padded_img * visionMask
      return focusedImg[focRad:-focRad, focRad:-focRad, : ]


  def _gazeSampler(self, mask, d):
    
    # find all x and y's that are inside the mask of the main object
    gazeCans = np.where(mask==1)
    
    # if mask is empty, then handle by setting the gaze point to center
    if len(gazeCans[0]) == 0:
      return (int(IMG_SIZE/2), int(IMG_SIZE/2))
    
    # remove the ones that are closer than the fovea radius to the bounds of image
    filtered_list = []
    for i in range(len(gazeCans[0])):
        if gazeCans[0][i] >= d and gazeCans[1][i] >= d and gazeCans[0][i] < IMG_SIZE - d and gazeCans[1][i] < IMG_SIZE - d:
            filtered_list.append((gazeCans[0][i], gazeCans[1][i]))
    
    if len(filtered_list) == 0: # if no gaze candidate is left
      selctedIndx = random.randint(0, len(gazeCans[0])-1)
      return (gazeCans[0][selctedIndx], gazeCans[1][selctedIndx])
    else:
       selctedIndx = random.randint(0, len(filtered_list)-1)
       return list(filtered_list[selctedIndx])

  def __getitem__(self, idx):
    
    # Load the sample image
    imgID = self.ids[idx]
    path = self.coco.loadImgs(imgID)[0]["file_name"]
    img = Image.open(os.path.join(self.root_dir, self.split, path))
    img = np.asarray(img.resize((IMG_SIZE,IMG_SIZE)))

    # load annotation which contains multiple objects with their segmentation and category.
    anns = self.coco.loadAnns(self.coco.getAnnIds(imgID, self.targetCats[idx]))

    
    objId = self._findObjSegIdx(anns, self.targetCats[idx])
    maskO = self.coco.annToMask(anns[objId])

    maskO = Image.fromarray(maskO)
    maskO = np.asarray(maskO.resize((IMG_SIZE,IMG_SIZE)))

    gaze = self._gazeSampler(maskO, int(FOVEA_SIZE/2))
    
    if(len(img.shape)<3): # if image is grayscale
      img = np.repeat(img[:,:,np.newaxis], 3, axis=2)
    

    fltImg =  img * np.repeat(maskO[:,:,np.newaxis], 3, axis=2)

    # apply to focus on the image
    img = self._imgfocus(img, gaze, crop=True)

    # apply focus on the masked image
    fltImg = self._imgfocus(fltImg, gaze, crop=True)
    # fltImg = np.resize(fltImg, (IMG_SIZE, IMG_SIZE)) # distorts the img

    img = self.transform(img)
    fltImg = self.transform(fltImg)
   

    target = {}
    target["label"] = self.labels[idx]
    target["masked_img"] = fltImg
    target["gaze"] = torch.as_tensor(gaze)

    return img, target
    

##### MEDIUM FUNCTIONS
def imageNettePrep():
   
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    
    transes = transforms.Compose([
                # transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(IMG_SIZE),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
                ])
    # D:\Tinh-toan-khtk\E2E-Point-SPV\imagenette2-320
    trainDataset = ImageNetteDataset("D:/Tinh-toan-khtk/E2E-Point-SPV/imagenette2-320/train", transform=transes)
    validDataset = ImageNetteDataset("D:/Tinh-toan-khtk/E2E-Point-SPV/imagenette2-320/val", transform=transes)
    return trainDataset, validDataset


def CocoPrep():
   
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                     std=[0.229, 0.224, 0.225])
    gray2color = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
    transes = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Resize(size=IMG_SIZE),
                gray2color
                ])

    trainDataset = CocoDataset("/scratch-shared/anejad/COCO", "train2017" ,transform=transes)
    validDataset = CocoDataset("/scratch-shared/anejad/COCO", "val2017", transform=transes)
    # return trainDataset, validDataset
    return trainDataset, validDataset