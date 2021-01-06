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
                    items = line.strip().split(',') 
                    if int(eval(items[1])) ==0: #cvte dataset
                        image_name = os.path.join(path_to_img_dir, items[0])
                    else: #chest x-ray8 dataset
                        image_name = os.path.join(PATH_TO_IMAGES_DIR_COM, items[0])
                    if os.path.isfile(image_name) == True:
                        label = int(eval(items[2])) #eval for 
                        if label ==0:  #negative
                            image_names.append(image_name)    
                            labels.append([1, 0])
                        elif label == 1: #positive
                            image_names.append(image_name)    
                            labels.append([0, 1])
                        else: continue

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
PATH_TO_IMAGES_DIR_COM = '/data/fjsdata/NIH-CXR/images/images/'
PATH_TO_TRAIN_FILE = '/data/fjsdata/CVTEDR/cxr_train.txt'
PATH_TO_VAL_FILE = '/data/fjsdata/CVTEDR/cxr_val.txt'
PATH_TO_TEST_FILE = '/data/fjsdata/CVTEDR/cxr_test.txt'

def get_train_dataloader(batch_size, shuffle, num_workers):
    dataset_train = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                    path_to_dataset_file=[PATH_TO_TRAIN_FILE], transform=transform_seq_train)
    #sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train) #for multi cpu and multi gpu
    #data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, sampler = sampler_train, 
                                   #shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)#drop_last=True
    return data_loader_train

def get_validation_dataloader(batch_size, shuffle, num_workers):
    dataset_validation = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                          path_to_dataset_file=[PATH_TO_VAL_FILE], transform=transform_seq_test)
    data_loader_validation = DataLoader(dataset=dataset_validation, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_validation


def get_test_dataloader(batch_size, shuffle, num_workers):
    dataset_test = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                    path_to_dataset_file=[PATH_TO_TEST_FILE], transform=transform_seq_test)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_test

def splitCVTEDR(dataset_path, pos_dataset_path): 
    """
    #deal with chest x-ray8 dataset (positive)
    com_data = pd.read_csv("/data/fjsdata/NIH-CXR/Data_Entry_2017_v2020.csv" , sep=',') #detailed information of images
    print (com_data.shape)
    com_data = com_data.drop(com_data[com_data['Finding Labels']=='No Finding'].index)
    com_data = com_data[['Image Index']]
    com_data.rename(columns={'Image Index':'name'}, inplace = True)
    com_data['label']=1
    com_data['flag'] = 1
    com_data = com_data.sample(n=20000, random_state=1) #random 20000
    """
    #deal with true positive samples
    pos_datas = pd.read_csv(pos_dataset_path, sep=',',encoding='gbk')
    print("\r CXR Columns: {}".format(pos_datas.columns))
    pos_images = pos_datas['图片路径'].tolist()
    pos_images = [x.split('\\')[-1].split('_')[0]+'.jpeg' for x in pos_images]
    
    #delete false positive samples
    datas = pd.read_csv(dataset_path, sep=',')
    datas_image = datas['name'].tolist()
    #assert set(datas_image) > set(pos_images) #true, contain
    pos_images_new = []
    for pname in pos_images:
        if pname in datas_image:
            pos_images_new.append(pname)
    
    datas = datas.drop(datas[datas['label']==3.0].index)
    datas = datas.drop(datas[datas['label']==1.0].index)
    datas['flag'] = 0

    #merge negative and positive sample
    pos_datas = pd.DataFrame(pos_images_new, columns=['name'])
    pos_datas['label'] = 1
    pos_datas['flag'] = 0

    datas = datas.sample(n=len(pos_datas), random_state=1) #random sampling 1 times for negative
    datas = pd.concat([datas, pos_datas], axis=0)
    #datas = pd.concat([datas, pos_datas, com_data], axis=0) 
    #datas = pd.concat([datas, com_data], axis=0)
    print("\r dataset shape: {}".format(datas.shape)) 
    print("\r dataset distribution: {}".format(datas['label'].value_counts()))

    #split train, validation, test
    images = datas[['name','flag']]
    labels = datas[['label']]
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.20, random_state=11)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=22)
    print("\r trainset shape: {}".format(X_train.shape)) 
    print("\r trainset distribution: {}".format(y_train['label'].value_counts()))
    print("\r valset shape: {}".format(X_val.shape)) 
    print("\r valset distribution: {}".format(y_val['label'].value_counts()))
    print("\r testset shape: {}".format(X_test.shape)) 
    print("\r testset distribution: {}".format(y_test['label'].value_counts()))
    trainset = pd.concat([X_train, y_train], axis=1).to_csv('/data/fjsdata/CVTEDR/cxr_train.txt', index=False, header=False, sep=',')
    valset = pd.concat([X_val, y_val], axis=1).to_csv('/data/fjsdata/CVTEDR/cxr_val.txt', index=False, header=False, sep=',')
    testset = pd.concat([X_test, y_test], axis=1).to_csv('/data/fjsdata/CVTEDR/cxr_test.txt', index=False, header=False, sep=',')
    
if __name__ == "__main__":

    #generate split lists
    #splitCVTEDR('/data/fjsdata/CVTEDR/CXR20201210.csv', '/data/fjsdata/CVTEDR/CVTE-DR-Pos-954.csv')
    
    #for debug   
    data_loader_train = get_train_dataloader(batch_size=10, shuffle=True, num_workers=0)
    for batch_idx, (image, label) in enumerate(data_loader_train):
        print(batch_idx)
        print(image.shape)
        print(label.shape)
        break
    
   
    
    
        
    
    
    
    