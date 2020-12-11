import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
import time
import random
"""
Dataset: Chest X-Ray8
https://www.kaggle.com/nih-chest-xrays/data
https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345
1) 112,120 X-ray images with disease labels from 30,805 unique patients
2）Label:['Atelectasis', 'Cardiomegaly', 'Effusion','Infiltration', 'Mass', 'Nodule', 'Pneumonia', \
        'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
"""
#generate dataset 
class DatasetGenerator(Dataset):
    def __init__(self, path_to_img_dir, path_to_dataset_file, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        for file_path in path_to_dataset_file:
            with open(file_path, "r") as f:
                for line in f:
                    items = line.split()
                    image_name= items[0].split('/')[1]
                    label = items[1:]
                    label = [int(i) for i in label]
                    image_name = os.path.join(path_to_img_dir, image_name)
                    image_names.append(image_name)
                    labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)

#generate box dataset
class BBoxGenerator(Dataset):
    def __init__(self, path_to_img_dir, path_to_dataset_file, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        boxes = []
        boxdata = pd.read_csv(path_to_dataset_file, sep=',')
        boxdata = boxdata[['Image Index','Finding Label','Bbox [x', 'y', 'w', 'h]']]
        for _, row in boxdata.iterrows():
            image_name = os.path.join(path_to_img_dir, row['Image Index'])
            image_names.append(image_name)
            label = np.zeros(len(CLASS_NAMES))
            label[CLASS_NAMES.index(row['Finding Label'])] = 1
            labels.append(label)
            boxes.append(np.array([row['Bbox [x'], row['y'], row['w'], row['h]']]))#xywh

        self.image_names = image_names
        self.labels = labels
        self.boxes = boxes
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        width, height = image.size 
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        #get gounded-truth boxes
        x_scale = 256/width
        y_scale = 256/height
        crop_del = (256-224)/2
        box = self.boxes[index]
        x, y, w, h = int(box[0])*x_scale-crop_del, int(box[1])*y_scale-crop_del, int(box[2])*x_scale, int(box[3])*y_scale
        gtbox = np.array([x,y,w,h])

        return image_name, gtbox, image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)

#config 
transform_seq = transforms.Compose([
   transforms.Resize((256,256)),
   #transforms.RandomCrop(224),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
])
transform_seq_amplify = transforms.Compose([
   transforms.Resize((256,256)),
   transforms.FiveCrop(224),
   transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
   transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(crop) for crop in crops])),
])

PATH_TO_IMAGES_DIR = '/data/fjsdata/NIH-CXR/images/images/'
PATH_TO_TRAIN_FILE = './Dataset/train.txt'
PATH_TO_VAL_FILE = './Dataset/val.txt'
PATH_TO_TEST_FILE = './Dataset/test.txt'
PATH_TO_BOX_FILE = './Dataset/fjs_BBox.csv'
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

def get_train_dataloader(batch_size, shuffle, num_workers):
    dataset_train = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                     path_to_dataset_file=[PATH_TO_TRAIN_FILE], transform=transform_seq)
    #sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train) #for multi cpu and multi gpu
    #data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, sampler = sampler_train, 
                                   #shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_train

def get_validation_dataloader(batch_size, shuffle, num_workers):
    dataset_validation = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                          path_to_dataset_file=[PATH_TO_VAL_FILE], transform=transform_seq)
    data_loader_validation = DataLoader(dataset=dataset_validation, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_validation


def get_test_dataloader(batch_size, shuffle, num_workers):
    dataset_test = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                    path_to_dataset_file=[PATH_TO_TEST_FILE], transform=transform_seq)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_test

def get_bbox_dataloader(batch_size, shuffle, num_workers):
    dataset_bbox = BBoxGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR, 
                                 path_to_dataset_file=PATH_TO_BOX_FILE, transform=transform_seq)
    data_loader_bbox = DataLoader(dataset=dataset_bbox, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_bbox

#for cross-validation
def get_train_dataloader_full(batch_size, shuffle, num_workers, split_ratio=0.1):
    dataset_train_full = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                         path_to_dataset_file=[PATH_TO_TRAIN_FILE, PATH_TO_VAL_FILE], transform=transform_seq)

    val_size = int(split_ratio * len(dataset_train_full))
    train_size = len(dataset_train_full) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset_train_full, [train_size, val_size])

    data_loader_train = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    data_loader_val = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_train, data_loader_val

if __name__ == "__main__":
    #for debug   
    data_loader_train = get_train_dataloader(batch_size=512, shuffle=True, num_workers=0)
    roi_idx = np.array([0,0,0, 1,1,1, 3,3,3, 511,510,511])
    for batch_idx, (image, label) in enumerate(data_loader_train):
         roi_label = label[roi_idx]
         print(roi_label)