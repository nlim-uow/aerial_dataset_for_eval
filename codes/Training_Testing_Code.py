#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
from torchvision.transforms import ToTensor
from torchvision.datasets import VOCSegmentation, VOCDetection
from torcheval.metrics import MultilabelAccuracy
from torchmetrics.classification import MultilabelJaccardIndex, Accuracy
from torch.utils.data import Dataset

from sklearn.preprocessing import MultiLabelBinarizer
import copy
from tqdm import tqdm
import cv2
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import ToTensor
from torchvision.datasets.utils import download_url
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
   
import torch.nn.functional as F
class FocalLoss(nn.Module):
   def __init__(self, alpha=1, gamma=2):
       super(FocalLoss, self).__init__()
       self.alpha = alpha
       self.gamma = gamma

   def forward(self, inputs, targets):
       bce_loss = F.binary_cross_entropy_with_logits(inputs.squeeze(),  targets.float())
       loss = self.alpha * (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss
       return loss
   
import torchvision.transforms.functional as TF
import random
from PIL import Image
from torchvision.transforms import GaussianBlur

class MyRotationTransform:
   """Rotate by one of the given angles."""

   def __init__(self, angles):
       self.angles = angles

   def __call__(self, x):
       angle = random.choice(self.angles)
       return TF.rotate(x, angle)

import sys
sys.path.append('PATH_TO_CAPTUM')
import captum

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device being used is: ',device)

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet18"

# Select the dataset to be trained on (voc, opet, cub, aerial)
dataset_name = "cub"

# Batch size for training (change depending on how much memory you have)
batch_size = 32

# Image Size 
input_size = 224


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256,antialias=True),
        transforms.CenterCrop(input_size),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),       
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256,antialias=True),
        transforms.CenterCrop(input_size),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


class vocSegmentationClassificationDataset(Dataset):
    def __init__(self, voc_dataset,blur_size=11,blur_sigma=2, transform=None, target_transform=None, blur_segments=False, image_set='train'):
        self.images=[]
        self.masks=[]
        self.targets=[]
        self.blur_segments = blur_segments
        self.transform = transform
        self.target_transform = target_transform
        self.blur_size=blur_size
        self.blur_sigma=blur_sigma
        self.toTensor = transforms.Compose([transforms.ToTensor(),
        transforms.Resize(256,antialias=True),
        transforms.CenterCrop(input_size)])
        self.image_set = image_set

        for i in range(len(voc_dataset)):
            img,segMask=voc_dataset.__getitem__(i)
            segMask=np.array(segMask)
            classes=np.sort(np.unique(segMask)[1:-1])
            if len(classes) == 1: 
                for class_id in classes:
                    self.images.append(voc_dataset.images[i])
                    self.masks.append(voc_dataset.masks[i])
                    self.targets.append(class_id-1)

    def __getitem__(self,index:int):
        img=Image.open(self.images[index]).convert("RGB")
        img=np.array(img)/255
        gt_mask=Image.open(self.masks[index])
        gt_mask = np.array(gt_mask)/255.0
        target=self.targets[index]

        gt_mask = np.reshape(gt_mask, [gt_mask.shape[0], gt_mask.shape[1], 1])
        gt_mask = np.repeat(gt_mask, 3, axis=2)
        gt_mask_bool = gt_mask == (target+1)/255.0
        gt_mask = img * gt_mask_bool
        
        img = self.transform(img)
        gt_mask = self.toTensor(gt_mask)
        gt_mask_bool = self.toTensor(gt_mask_bool)

        if self.image_set == "train":
            if random.random() > 0.5:
                img = transforms.functional.vflip(img)
                gt_mask = transforms.functional.vflip(gt_mask)
                gt_mask_bool = transforms.functional.vflip(gt_mask_bool)
            if random.random() > 0.5:
                img = transforms.functional.hflip(img)
                gt_mask = transforms.functional.hflip(gt_mask)
                gt_mask_bool = transforms.functional.hflip(gt_mask_bool)

        if self.blur_segments:
            gaussian_blur_ = GaussianBlur(kernel_size=23, sigma=11)
            img_blur_gt = gaussian_blur_(gt_mask_bool[:,:,:].to(torch.float))
        else: 
            img_blur_gt = gt_mask_bool[:,:,:].to(torch.float)
        
        return img.float(), target, gt_mask_bool, gt_mask, img_blur_gt
    
    def __len__(self):
        return len(self.targets)


class petSegmentationClassificationDataset(Dataset):
    def __init__(self, pet_dataset,blur_size=11,blur_sigma=2, transform=None, target_transform=None, blur_segments=False, image_set='train'):
        self.images=[]
        self.masks=[]
        self.targets=[]
        self.blur_segments = blur_segments
        self.transform = transform
        self.target_transform = target_transform
        self.blur_size=blur_size
        self.blur_sigma=blur_sigma
        self.toTensor = transforms.Compose([transforms.ToTensor(),
        # transforms.Resize(256,antialias=True),
        transforms.CenterCrop(input_size)])
        self.image_set = image_set

        for i in range(len(pet_dataset)):
            images, (segMask, class_id) = pet_dataset.__getitem__(i)
            segMask=np.array(segMask)
            self.images.append(images)
            self.masks.append(segMask)
            self.targets.append(class_id)

    def __getitem__(self,index:int):
        img=self.images[index]
        img=np.array(img)/255
        gt_mask=self.masks[index]
        gt_mask = np.array(gt_mask)
        target=self.targets[index]

        gt_mask_bool = gt_mask != (2.0)
        
        img = self.transform(img)
        gt_mask = self.toTensor(gt_mask)
        gt_mask_bool = self.toTensor(gt_mask_bool)

        if self.image_set == "train":
            if random.random() > 0.5:
                img = transforms.functional.vflip(img)
                gt_mask = transforms.functional.vflip(gt_mask)
                gt_mask_bool = transforms.functional.vflip(gt_mask_bool)
            if random.random() > 0.5:
                img = transforms.functional.hflip(img)
                gt_mask = transforms.functional.hflip(gt_mask)
                gt_mask_bool = transforms.functional.hflip(gt_mask_bool)
        
        if self.blur_segments:
            gaussian_blur_ = GaussianBlur(kernel_size=23, sigma=11)
            img_blur_gt = gaussian_blur_(gt_mask_bool[:,:,:].to(torch.float))
        else: 
            img_blur_gt = gt_mask_bool[:,:,:].to(torch.float)
        
        return img.float(), target, gt_mask_bool, gt_mask, img_blur_gt
    
    def __len__(self):
        return len(self.targets)

class CUB200Segmentation(datasets.VisionDataset):
    _SPLITS_DIR = "Segmentation"
    _TARGET_DIR = "SegmentationClass"
    _TARGET_FILE_EXT = ".png"

    def __init__(
        self,
        root: str,
        image_set: str = 'train',
        download: bool = False,
        transform = None,
        target_transform = None,
        blur_segments: bool = False,
        blur_size: int = 11,
        blur_sigma: int = 2,
        input_size: int = 224,
    ):
        super().__init__(root)
        self.url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
        self.segment_url = 'https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz?download=1'
        self.root = root
        self.filename = 'CUB_200_2011.tgz'
        self.segment_filename = 'CUB_200_2011_segmentation.tgz'
        self.md5 = '97eceeb196236b17998738112f37df78'
        self.segment_md5='4d47ba1228eae64f2fa547c47bc65255'
        self.base_dir = 'CUB_200_2011/images'
        self.img_root = os.path.join(self.root, self.base_dir)
        self.segment_dir = 'CUB_200_2011/segmentations'
        self.segment_root = os.path.join(self.root, self.segment_dir)
        directory = 'CUB_200_2011'
        self.root_dir = os.path.join(self.root, directory)
        self.toTensor = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize(256,antialias=True),
                                            transforms.CenterCrop(input_size)])
        self.input_size = input_size        
        self.transform=transform
        self.train = image_set=='train'
        self.blur_segments = blur_segments
        self.blur_size = blur_size
        self.blur_sigma= blur_sigma
        if download:
            self._download()
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')            
        #assert len(self.images) == len(self.targets)
    def _load_metadata(self):
        images= pd.read_csv(os.path.join(self.root,'CUB_200_2011','images.txt'), sep=' ', names=['img_id','filepath'])
        image_class_labels= pd.read_csv(os.path.join(self.root,'CUB_200_2011','image_class_labels.txt'), sep=' ', names=['img_id','target'])
        train_test_split= pd.read_csv(os.path.join(self.root,'CUB_200_2011','train_test_split.txt'),sep=' ', names=['img_id','is_training_img'])
        data=images.merge(image_class_labels, on='img_id')
        self.data=data.merge(train_test_split, on='img_id')
        if self.train:
            self.data = self.data[self.data.is_training_img==1]
        else:
            self.data = self.data[self.data.is_training_img==0]
    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False
        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.img_root, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True
    def _download(self):
        import tarfile
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_url(self.url, self.root, self.filename, self.md5)
        download_url(self.segment_url, self.root_dir, self.segment_filename, self.segment_md5)
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)
        with tarfile.open(os.path.join(self.root_dir, self.segment_filename), "r:gz") as tar:
            tar.extractall(path=self.root_dir)

    def __len__(self) -> int:
        return len(self.data)
    def __getitem__(self, idx: int):
        sample = self.data.iloc[idx]
        img_path = os.path.join(self.root, self.img_root, sample.filepath)
        target = self.data.iloc[idx].target
        mask_path = os.path.join(self.root, self.segment_dir, sample.filepath).replace('jpg','png')
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        if self.train:
            if random.random() > 0.5:
                img = transforms.functional.vflip(img)
                mask = transforms.functional.vflip(mask)
            if random.random() > 0.5:
                img = transforms.functional.hflip(img)
                mask = transforms.functional.hflip(mask)
        img = np.array(img)/255
        gt_mask = np.array(mask)/255

        gt_mask_bool = gt_mask > 0.5
        gt_mask = img * gt_mask_bool

        img = self.toTensor(img)
        gt_mask = self.toTensor(gt_mask)
        gt_mask_bool = self.toTensor(gt_mask_bool)

        mask = self.toTensor(mask)
        if self.train:
            if random.random() > 0.5:
                img = transforms.functional.vflip(img)
                gt_mask = transforms.functional.vflip(gt_mask)
                gt_mask_bool = transforms.functional.vflip(gt_mask_bool)
            if random.random() > 0.5:
                img = transforms.functional.hflip(img)
                gt_mask = transforms.functional.hflip(gt_mask)
                gt_mask_bool = transforms.functional.hflip(gt_mask_bool)
 
        if self.blur_segments:
            gaussian_blur_ = GaussianBlur(kernel_size=23, sigma=11)
            img_blur_gt = gaussian_blur_(gt_mask_bool[:,:,:].to(torch.float))
        else: 
            img_blur_gt = gt_mask_bool[:,:,:].to(torch.float)

        return img.float(), target-1, gt_mask_bool, gt_mask_bool.int(), img_blur_gt

class LoadAerialDataset(Dataset):
    """LCDB"""

    def __init__(self, root_dir, image_set, blur_segments=False, transform=None,blur_size=11,blur_sigma=2):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if image_set not in ['train', 'val']:
            raise ValueError('Invalid image set. Only train and val is allowed!')
        else:
            self.image_set = image_set
        
        self.root_dir = root_dir
        self.image_data_path = os.path.join(self.root_dir, 'cls', self.image_set)
        self.image_seg_data_path = os.path.join(self.root_dir, 'mask', self.image_set)
        self.transform = transform
        self.blur_size=blur_size
        self.blur_sigma=blur_sigma
        self.blur_segments = blur_segments
        self.toTensor = transforms.Compose([transforms.ToTensor()])

        self.images = []
        self.targets=[]

        self.get_filenames()
        self.load_data()
        
    def load_data(self):
        pass
        
    def get_filenames(self):
        image_list = []
        label_list = []
        for file_path in os.listdir(self.image_data_path):
            for image_path in os.listdir(os.path.join(self.image_data_path, file_path)):
                if os.path.isfile(os.path.join(self.image_data_path, file_path, image_path)):
#                     image_list.append(os.path.join(self.image_data_path, file_path, image_path))
                    image_list.append(os.path.join(file_path, image_path))
                    label_list.append(int(file_path))
        
        self.images = image_list
        self.targets = label_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index:int):
        img_name = self.images[index]
        target = self.targets[index]
        img = Image.open(os.path.join(self.image_data_path, img_name)).convert("RGB")
        img = np.array(img)/255.0

        gt_mask = Image.open(os.path.join(self.image_seg_data_path, img_name)).convert("RGB")
        gt_mask = np.array(gt_mask)/255.0
        
        img = self.transform(img)
        gt_mask = self.toTensor(gt_mask)
              
        if self.image_set == "train":
            if random.random() > 0.5:
                img = transforms.functional.vflip(img)
                gt_mask = transforms.functional.vflip(gt_mask)
            if random.random() > 0.5:
                img = transforms.functional.hflip(img)
                gt_mask = transforms.functional.hflip(gt_mask)
        
        gt_mask_bool = gt_mask > 0.1

        binary_mask = gt_mask[0,:,:].numpy()
#         print(np.shape(binary_mask))
        obj_pos = np.where(binary_mask > 0.2)
        
        if self.blur_segments:
            gaussian_blur_ = GaussianBlur(kernel_size=23, sigma=11)
            img_blur_gt = gaussian_blur_(gt_mask_bool[:,:,:].to(torch.float))
        else: 
            img_blur_gt = gt_mask_bool[:,:,:].to(torch.float)

                
        return img.float(), target, gt_mask_bool, gt_mask, img_blur_gt


print("Initializing Datasets and Dataloaders...")
# Number of classes in the dataset
if dataset_name == "voc":
    num_classes = 20
    voc_dataset=torchvision.datasets.VOCSegmentation('PATH_TO_VOC', year='2012', image_set='train', download=False)
    trainDataset=vocSegmentationClassificationDataset(voc_dataset,image_set='train',transform=data_transforms['train'],blur_segments=False)
    voc_dataset=torchvision.datasets.VOCSegmentation('PATH_TO_VOC', year='2012', image_set='val', download=False)
    valDataset=vocSegmentationClassificationDataset(voc_dataset,image_set='val',transform=data_transforms['val'],blur_segments=False)
    data_loader_train = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=4)
    data_loader_val = torch.utils.data.DataLoader(valDataset, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloaders_dict={'train': data_loader_train, 'val': data_loader_val }
    

elif dataset_name == "opet":
    num_classes = 37
    pet_dataset = OxfordIIITPet(root='PATH_TO_OPET', split='trainval', download=True, target_types=['segmentation', 'category'], transform=None, target_transform=None)
    trainDataset=petSegmentationClassificationDataset(pet_dataset,image_set='train',transform=data_transforms['train'],blur_segments=False)
    
    pet_dataset = OxfordIIITPet(root='PATH_TO_OPET', split='test', download=True, target_types=['segmentation', 'category'], transform=None, target_transform=None)
    valDataset=petSegmentationClassificationDataset(pet_dataset,image_set='val',transform=data_transforms['val'],blur_segments=False)
    data_loader_train = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=4)
    data_loader_val = torch.utils.data.DataLoader(valDataset, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloaders_dict={'train': data_loader_train, 'val': data_loader_val }

elif dataset_name == "cub":
    num_classes = 200
    image_datasets = {x: CUB200Segmentation('PATH_TO_CUB',image_set=x,download=True,transform=data_transforms[x],blur_segments=False) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

elif dataset_name == "aerial":
    num_classes = 12
    AerialDataTrain = LoadAerialDataset('PATH_TO_AERIAL_DATASET', image_set='train', blur_segments=False, transform=data_transforms['train'])
    AerialDataVal = LoadAerialDataset('PATH_TO_AERIAL_DATASET', image_set='val', blur_segments=False, transform=data_transforms['val'])
    

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for child in model.children():
            for layer in child.modules():
                if(isinstance(layer,torch.nn.BatchNorm2d)):
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    for param in layer.parameters():
                        param.requires_grad = True

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        model_ft.fc.activation = nn.Sigmoid()
        input_size = 224

    elif model_name == "resnet34":
        """ Resnet34
        """
        model_ft = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        model_ft.fc.activation = nn.Sigmoid()
        input_size = 224

    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        model_ft.fc.activation = nn.Sigmoid()
        input_size = 224

    elif model_name == "resnet101":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        model_ft.fc.activation = nn.Sigmoid()
        input_size = 224
        
    elif model_name == "resnet152":
        """ Resnet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        model_ft.fc.activation = nn.Sigmoid()
        input_size = 224    
    
    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
 
    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size



# In[36]:


class EarlyStopping:
    """Early stops the training if validation accuracy doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation accuracy improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation accuracy improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_acc = None
        self.best_acc_loss = None
        self.best_iou = None
        self.early_stop = False
        self.val_acc_max = -float('inf')
        self.val_acc_loss = -float('inf')
        self.val_iou = -float('inf')
        self.delta = delta

    def __call__(self, val_acc, val_loss, iou, model):
        if self.best_acc is None:
            self.best_acc = val_acc
            self.best_acc_loss = val_loss
            self.best_iou = iou
            self.save_checkpoint(val_acc, val_loss, iou, model)
        elif val_acc < self.best_acc + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_acc = val_acc
            self.best_acc_loss = val_loss
            self.best_iou = iou
            self.save_checkpoint(val_acc, val_loss, iou, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, val_loss, iou, model):
        """Saves model when validation accuracy increase."""
        if self.verbose:
            print(f'Validation accuracy increased ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), f'checkpoint_{model_name}_model_BlurGT_{mixing_ratio}.pth')
        self.val_acc_max = val_acc
        self.val_acc_loss = val_loss
        self.val_iou = iou        


def train_model(model, dataloaders, criterion1, criterion2, mixing_ratio, optimizer, num_epochs=25, is_inception=False, patience=15, min_delta=0.1):
    since = time.time()

    val_acc_history = []
    
    train_history = {}
    train_history["train accuracy"] = []
    train_history["train loss"] = []
    train_history["train iou"] = []
    train_history["val loss"] = []
    train_history["val accuracy"] = []
    train_history["val iou"] = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 99999
    iou = torch.tensor(0.0)
    early_stopping = EarlyStopping(patience=patience, verbose=True)


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_iou = 0.0
            running_corrects = 0

            # Iterate over data.
            count = 0
            metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
            for inputs, labels, _, pos_img, img_blur_gt in tqdm(dataloaders[phase], desc=f"Epoch {epoch + 1} in {phase}", leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss1 = criterion1(outputs, labels)
                        if mixing_ratio < 1.0:
                            loss2 = criterion2(model, inputs, img_blur_gt.to(device), class_name=labels)

                        if mixing_ratio < 1.0:
                            loss = (mixing_ratio*loss1) + ((1.0-mixing_ratio)*loss2)
                        else:
                            loss = loss1
                            
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
               
                running_corrects += metric(preds, labels)
                running_iou += iou
                count += 1
                      
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double()/count
            epoch_iou = running_iou/count

            print('{} Loss: {:.4f} Acc: {:.4f}, IoU: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_iou))
            train_history[phase + " loss"].append(epoch_loss)
            train_history[phase + " accuracy"].append(epoch_acc.to('cpu').numpy())
            train_history[phase + " iou"].append(epoch_iou.to('cpu').numpy())

        # Call the early stopping logic
    
        if phase == 'val':
            early_stopping(epoch_acc, epoch_loss, iou, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(early_stopping.val_acc_loss))
    print('Best val acc: {:4f}'.format(early_stopping.val_acc_max))
    print('Best val IoU: {:4f}'.format(early_stopping.val_iou))

    return model, train_history


# In[42]:


class ModelExplanations:
    def __init__(self, model, target_layer, mask_type, cam_type='LayerGradCam', use_cuda=False):
        self.mask_type = mask_type
        self.cam_type = cam_type
        
        if use_cuda:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        
        self.layer = target_layer
        if self.cam_type == 'LayerGradCam':
            self.cam = captum.attr.LayerGradCam(model, self.layer, device_ids=self.device)
        elif self.cam_type == 'GuidedGradCam':
            self.cam = captum.attr.GuidedGradCam(model, self.layer, device_ids=self.device)
        
    def make_cams(self, img, class_name, th1, th2):
        transform = transforms.Compose([
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        x = img
    #     x = x.unsqueeze(0)
        x_y=torch.flip(x,[3])
        x_x=torch.flip(x,[2])
        x_xy=torch.flip(x,[2,3])
        x_rot90=torch.rot90(x,1,[2,3])
        x_rot270=torch.rot90(x,3,[2,3])
        x_x_rot90=torch.rot90(x_x,1,[2,3])
        x_y_rot90=torch.rot90(x_y,1,[2,3])

        th=th1
        posClass=class_name

        grayscale_cam = self.cam.attribute(inputs=x.to(device),target=posClass.to(device))
        grayscale_cam_x = self.cam.attribute(inputs=x_x,target=posClass)
        grayscale_cam_y = self.cam.attribute(inputs=x_y,target=posClass)
        grayscale_cam_xy = self.cam.attribute(inputs=x_xy,target=posClass)
        grayscale_cam_x_rot90 = self.cam.attribute(inputs=x_rot90,target=posClass)
        grayscale_cam_x_rot270 = self.cam.attribute(inputs=x_rot270,target=posClass)
        grayscale_cam_x_x_rot90 = self.cam.attribute(inputs=x_x_rot90,target=posClass)
        grayscale_cam_x_y_rot90 = self.cam.attribute(inputs=x_y_rot90,target=posClass)

        grayscale_cam_ori = grayscale_cam
        grayscale_cam_y=torch.flip(grayscale_cam_y,[3])
        grayscale_cam_x=torch.flip(grayscale_cam_x,[2])
        grayscale_cam_xy=torch.flip(grayscale_cam_xy,[2,3])
        grayscale_cam_x_rot90=torch.rot90(grayscale_cam_x_rot90,3,[2,3])
        grayscale_cam_x_rot270=torch.rot90(grayscale_cam_x_rot270,1,[2,3])
        grayscale_cam_x_x_rot90=torch.flip(torch.rot90(grayscale_cam_x_x_rot90,1,[2,3]),[2])
        grayscale_cam_x_y_rot90=torch.flip(torch.rot90(grayscale_cam_x_y_rot90,1,[2,3]),[3])
        
        grayscale_cam_max=torch.max(grayscale_cam,grayscale_cam_x)
        grayscale_cam_max=torch.max(grayscale_cam_max,grayscale_cam_y)
        grayscale_cam_max=torch.max(grayscale_cam_max,grayscale_cam_xy)
        grayscale_cam_max=torch.max(grayscale_cam_max,grayscale_cam_x_rot90)
        grayscale_cam_max=torch.max(grayscale_cam_max,grayscale_cam_x_rot270)
        grayscale_cam_max=torch.max(grayscale_cam_max,grayscale_cam_x_x_rot90)
        grayscale_cam_max=torch.max(grayscale_cam_max,grayscale_cam_x_y_rot90)
        grayscale_cam_avg=torch.sum(torch.stack([grayscale_cam_ori,grayscale_cam_x,grayscale_cam_y,grayscale_cam_xy,
                                     grayscale_cam_x_rot90,grayscale_cam_x_rot270,grayscale_cam_x_x_rot90,
                                     grayscale_cam_x_y_rot90]), dim=0)/8

        if self.cam_type == 'LayerGradCam':
            grayscale_cam_max = captum.attr.LayerAttribution.interpolate(grayscale_cam_max, (224,224), interpolate_mode='bicubic')
            grayscale_cam_avg = captum.attr.LayerAttribution.interpolate(grayscale_cam_avg, (224,224), interpolate_mode='bicubic')
        elif self.cam_type == 'GuidedGradCam':
            grayscale_cam_max = grayscale_cam_max[:,1,:,:]
            grayscale_cam_avg = grayscale_cam_avg[:,1,:,:]

        return grayscale_cam_max.squeeze(0).squeeze(0).detach().cpu().numpy()
   


class SoLoss(nn.Module):
    """
    Segment Overlap Loss
    """
    def __init__(self, weight=None, size_average=True):
        super(SoLoss, self).__init__()
        self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        

    def forward(self, model, gt_img, gt_mask, class_name):
                
#         cam.cam.batch_size = batch_size
        # Make Explinations
        th1 = 0.75
        th2 = 0.99
        loss = torch.tensor(0.0)
        loss = loss.to(device)
        
        ep = cam.make_cams(gt_img, th1=th1, th2=th2, class_name=class_name)
        
        ep = torch.from_numpy(ep)
        ep = ep.to(device)
        gt_mask_layer = gt_mask[:,0,:,:]
        gt_mask_layer = gt_mask_layer.to(device)
        cs = (1 - self.cos(torch.flatten(ep), torch.flatten(gt_mask_layer)))
        loss += cs  

        loss = loss / gt_img.shape[0]
        loss = loss.requires_grad_(True)
        
        return loss


feature_extract = True

model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Number of training epochs
num_epochs = 75

model_ft = model_ft.to(device)
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

# Make the Cam Model
target_layer = model_ft.layer4[-1]
cam = ModelExplanations(model=model_ft, target_layer=target_layer, mask_type="None", cam_type='LayerGradCam', use_cuda=True)
mixing_ratio = 1.0           

optimizer_ft = optim.Adam(params_to_update, lr=0.001)
#     criterion = nn.BCEWithLogitsLoss()
criterion1 = nn.CrossEntropyLoss()
criterion2 = SoLoss()
# model_ft, train_hist = train_model(model_ft, dataloaders_dict, criterion1, criterion2, mixing_ratio, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"), patience=60, min_delta=0.1)


set_parameter_requires_grad(model_ft,False)
ft_epochs = 130
del cam
cam = ModelExplanations(model=model_ft, target_layer=target_layer, mask_type="None", cam_type='LayerGradCam', use_cuda=True)


params_to_update = model_ft.parameters()

model_ft.layer1.append(torch.nn.Dropout(0.25))
model_ft.layer2.append(torch.nn.Dropout(0.25))
model_ft.layer3.append(torch.nn.Dropout(0.25))
model_ft.layer4.append(torch.nn.Dropout(0.25))

optimizer_ft = optim.Adam(params_to_update, lr=0.0003)
mixing_ratio = 0.75
criterion1 = nn.CrossEntropyLoss()
criterion2 = SoLoss()
model_ft, train_hist = train_model(model_ft, dataloaders_dict, criterion1, criterion2, mixing_ratio, optimizer_ft, num_epochs=ft_epochs, is_inception=(model_name=="inception"), patience=60, min_delta=0.1)

os.rename(f'checkpoint_{model_name}_model_BlurGT_{mixing_ratio}.pth', f"{model_name}_VOC_SoLoss_BlurGT_{mixing_ratio}.pth")
