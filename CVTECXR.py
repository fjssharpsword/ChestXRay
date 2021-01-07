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
from sklearn.utils import shuffle
import shutil

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
                    image_name = os.path.join(path_to_img_dir, items[0])
                    if os.path.isfile(image_name) == True:
                        label = int(eval(items[1])) #eval for 
                        if label ==0:  #negative
                            image_names.append(image_name)    
                            labels.append([0])
                        elif label == 1: #positive
                            image_names.append(image_name)    
                            labels.append([1])
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
        
        #return image, torch.FloatTensor(label)
        return image_name, image, torch.LongTensor(label)

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
PATH_TO_TRAIN_FILE = '/data/fjsdata/CVTEDR/cxr_train_time.txt'
PATH_TO_VAL_FILE = '/data/fjsdata/CVTEDR/cxr_val_time.txt'
PATH_TO_TEST_FILE = '/data/fjsdata/CVTEDR/cxr_test_time.txt'

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
    #read all samples and drop positive sample(remain negative sample)
    datas = pd.read_csv(dataset_path, sep=',')
    neg_datas = datas.drop(datas[datas['label']==3.0].index)
    neg_datas = neg_datas.drop(neg_datas[neg_datas['label']==1.0].index)
    neg_images = neg_datas['name'].tolist()

    #read positive samples and validation
    pos_datas = pd.read_csv(pos_dataset_path, sep=',',encoding='gbk')
    print("\r CXR Columns: {}".format(pos_datas.columns))
    pos_images = pos_datas['图片路径'].tolist()
    pos_images = [x.split('\\')[-1].split('_')[0]+'.jpeg' for x in pos_images]

    #assert
    assert len(set(neg_images) & set(pos_images)) == 0
    """
    #-----------statistics-------------------------
    #According to time
    year_dict = {}
    for pname in pos_images:
        year = pname[2:4]
        if year in year_dict.keys():
            num = year_dict[year]
            year_dict[year] = num + 1
        else:
            year_dict[year] = 1
    print(year_dict) #{'18': 289, '17': 242, '19': 168, '16': 197, '20': 43}
    CXR Columns: Index(['图片路径', '0-肺扩张不全', '1-心脏扩大', '2-肺积液', '3-肺部浸润', '4-肺部块', '5-肺结节',
       '6-肺炎', '7-气胸', '8-肺实变', '9-肺水肿', '10-肺气肿', '11-肺纤维化', '12-肺膜增厚',
       '13-纤维灶', '14-钙化灶', '15-支扩', '16-肺大泡', '17-肺门增大'],
      dtype='object')
    #20-43: 1-3, 2-2, 3-2, 5-6, 6-4, 7-1, 12-6, 13-16, 14-9
    #-----------statistics-------------------------
    """
    #split trainset and testset
    pos_test, pos_train = [], []
    for x in pos_images:
        if x[2:4]=='20':
            pos_test.append(x)
        else:
            pos_train.append(x)
    neg_test, neg_train = [], []
    for x in neg_images:
        if x[2:4]=='20':
            neg_test.append(x)
        else:
            neg_train.append(x)
    neg_test = random.sample(neg_test, len(pos_test))
    neg_train = random.sample(neg_train, 2*len(pos_train))
    #merge positive and negative
    pos_datas_train = pd.DataFrame(pos_train, columns=['name'])
    pos_datas_train['label'] = 1
    neg_data_train = pd.DataFrame(neg_train, columns=['name'])
    neg_data_train['label'] = 0
    trainset = pd.concat([pos_datas_train, neg_data_train], axis=0)
    trainset = shuffle(trainset)
    print("\r trainset shape: {}".format(trainset.shape)) 
    print("\r trainset distribution: {}".format(trainset['label'].value_counts()))

    pos_datas_test = pd.DataFrame(pos_test, columns=['name'])
    pos_datas_test['label'] = 1
    neg_data_test = pd.DataFrame(neg_test, columns=['name'])
    neg_data_test['label'] = 0
    testset = pd.concat([pos_datas_test, neg_data_test], axis=0)
    testset = shuffle(testset)
    print("\r testset shape: {}".format(testset.shape)) 
    print("\r testset distribution: {}".format(testset['label'].value_counts()))

    #save 
    images = trainset[['name']]
    labels = trainset[['label']]
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.10, random_state=11)
    print("\r trainset shape: {}".format(X_train.shape)) 
    print("\r trainset distribution: {}".format(y_train['label'].value_counts()))
    print("\r valset shape: {}".format(X_val.shape)) 
    print("\r valset distribution: {}".format(y_val['label'].value_counts()))
    trainset = pd.concat([X_train, y_train], axis=1).to_csv('/data/fjsdata/CVTEDR/cxr_train_time.txt', index=False, header=False, sep=',')
    valset = pd.concat([X_val, y_val], axis=1).to_csv('/data/fjsdata/CVTEDR/cxr_val_time.txt', index=False, header=False, sep=',')
    testset = testset.to_csv('/data/fjsdata/CVTEDR/cxr_test_time.txt', index=False, header=False, sep=',')
 
def copyimage(dataset_path):
    with open(dataset_path, "r") as f:
        for line in f: 
            items = line.strip().split(',') 
            image_name = os.path.join(PATH_TO_IMAGES_DIR, items[0])
            if os.path.isfile(image_name) == True:
                label = int(eval(items[1])) #eval for
                shutil.copyfile(image_name, '/data/fjsdata/CVTEDR/test_images/'+str(label)+ '_' + items[0])

if __name__ == "__main__":

    #generate split lists
    #splitCVTEDR('/data/fjsdata/CVTEDR/CXR20201210.csv', '/data/fjsdata/CVTEDR/CVTE-DR-Pos-939.csv')
    #copy image to observe
    #copyimage('/data/fjsdata/CVTEDR/cxr_test_time.txt')
    
    #for debug   
    data_loader_train = get_train_dataloader(batch_size=10, shuffle=True, num_workers=0)
    for batch_idx, (_, image, label) in enumerate(data_loader_train):
        print(batch_idx)
        print(image.shape)
        print(label.shape)
        break
 
    
   
    
    
        
    
    
    
    