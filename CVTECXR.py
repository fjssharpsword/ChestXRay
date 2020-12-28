import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import pandas as pd
import numpy as np
import time
import random
from sklearn.model_selection import train_test_split
import sys
import torch.nn.functional as F
import scipy
import SimpleITK as sitk
import pydicom
from scipy import ndimage as ndi
from PIL import Image
import PIL.ImageOps 

"""
Dataset: CVTE ChestXRay
"""

class DatasetGenerator_Test(Dataset):
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
                    items = line.strip().split(',') 
                    image_name = os.path.join(path_to_img_dir, items[0])
                    if os.path.isfile(image_name) == True:
                        label = int(eval(items[1])) #eval for 
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
        try:
            image_name = self.image_names[index]
            image = Image.open(image_name).convert('RGB')
            #image.save('/data/pycode/ChestXRay/Imgs/test.jpeg',"JPEG", quality=95, optimize=True, progressive=True)
            label = self.labels[index]
            if self.transform is not None:
                image = self.transform(image)
        except Exception as e:
            print("Unable to read file. %s" % e)
        
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)

class DatasetGenerator_Train(Dataset):
    def __init__(self, path_to_img_dir, path_to_dataset_file, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names_pos, image_names_neg = [], []
        for file_path in path_to_dataset_file:
            with open(file_path, "r") as f:
                for line in f: 
                    items = line.strip().split(',') 
                    image_name = os.path.join(path_to_img_dir, items[0])
                    if os.path.isfile(image_name) == True:
                        label = int(eval(items[1])) #eval for 
                        if label == 1: #pos_list
                            image_names_pos.append(image_name)    
                        else:#neg_list
                            image_names_neg.append(image_name)    
 
        image_pairs = []
        label_sim = []
        for i in range(len(image_names_pos)):
            neg = random.sample(image_names_neg, 1)[0]
            image_pairs.append([image_names_pos[i], neg])
            label_sim.append([0])

            pos = random.sample(image_names_pos, 1)[0]
            image_pairs.append([image_names_pos[i], pos])
            label_sim.append([1])
        """    
        for i in range(len(image_names_neg)):
            neg = random.sample(image_names_neg, 1)[0]
            image_pairs.append([image_names_neg[i], neg])
            label_sim.append([1])

            pos = random.sample(image_names_pos, 1)[0]
            image_pairs.append([image_names_neg[i], pos])
            label_sim.append([0])
        """

        self.image_pairs = image_pairs
        self.label_sim = label_sim
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        try:
            image_pair = self.image_pairs[index]
            image_a = Image.open(image_pair[0]).convert('RGB')
            image_b = Image.open(image_pair[1]).convert('RGB')
            label = self.label_sim[index]
            if self.transform is not None:
                image_a = self.transform(image_a)
                image_b = self.transform(image_b)
        except Exception as e:
            print("Unable to read file. %s" % e)
        
        return image_a, image_b, torch.FloatTensor(label)

    def __len__(self):
        return len(self.label_sim)

#config 
transform_seq_train = transforms.Compose([
   transforms.Resize((256,256)),#256
   transforms.RandomCrop(224),#224
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
])
transform_seq_test = transforms.Compose([
   transforms.Resize((256,256)),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
])

PATH_TO_IMAGES_DIR = '/data/fjsdata/CVTEDR/images'
PATH_TO_TRAIN_FILE = '/data/fjsdata/CVTEDR/cxr_train.txt'
PATH_TO_VAL_FILE = '/data/fjsdata/CVTEDR/cxr_val.txt'
PATH_TO_TEST_FILE = '/data/fjsdata/CVTEDR/cxr_test.txt'

def get_train_dataloader(batch_size, shuffle, num_workers):
    dataset_train = DatasetGenerator_Train(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                    path_to_dataset_file=[PATH_TO_TRAIN_FILE, PATH_TO_VAL_FILE], transform=transform_seq_train)
    #sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train) #for multi cpu and multi gpu
    #data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, sampler = sampler_train, 
                                   #shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)#drop_last=True
    return data_loader_train

def get_validation_dataloader(batch_size, shuffle, num_workers):
    dataset_validation = DatasetGenerator_Test(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                          path_to_dataset_file=[PATH_TO_TRAIN_FILE, PATH_TO_VAL_FILE], transform=transform_seq_test)
    data_loader_validation = DataLoader(dataset=dataset_validation, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_validation


def get_test_dataloader(batch_size, shuffle, num_workers):
    dataset_test = DatasetGenerator_Test(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                    path_to_dataset_file=[PATH_TO_TEST_FILE], transform=transform_seq_test)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_test



if __name__ == "__main__":
  
    #for debug   
    data_loader_train = get_train_dataloader(batch_size=10, shuffle=True, num_workers=0)
    for batch_idx, (image_a, image_b, label) in enumerate(data_loader_train):
        print(batch_idx)
        print(image_a.shape)
        print(image_b.shape)
        print(label.shape)
    
    
        
    
    
    
    