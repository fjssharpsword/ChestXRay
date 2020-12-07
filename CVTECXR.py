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
        with open(path_to_dataset_file, "r") as f:
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


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

transform_seq = transforms.Compose([
   transforms.Resize((256,256)),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   normalize,
])

PATH_TO_IMAGES_DIR = '/data/fjsdata/NIH-CXR/images/images/'
PATH_TO_TRAIN_FILE = './Dataset/train.txt'
PATH_TO_VAL_FILE = './Dataset/val.txt'
PATH_TO_TEST_FILE = './Dataset/test.txt'

def get_train_dataloader(batch_size, shuffle, num_workers):
    dataset_train = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                     path_to_dataset_file=PATH_TO_TRAIN_FILE, transform=transform_seq)
    #sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train) #for multi cpu and multi gpu
    #data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, sampler = sampler_train, 
                                   #shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_train

def get_validation_dataloader(batch_size, shuffle, num_workers):
    dataset_validation = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                          path_to_dataset_file=PATH_TO_VAL_FILE, transform=transform_seq)
    data_loader_validation = DataLoader(dataset=dataset_validation, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_validation


def get_test_dataloader(batch_size, shuffle, num_workers):
    dataset_test = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                    path_to_dataset_file=PATH_TO_TEST_FILE, transform=transform_seq)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_test

def CVTEDR_Filter():
    cxr = pd.read_csv('/data/fjsdata/CVTEDR/CXR.csv', sep=',', encoding='gbk')
    print("\r CXR Columns: {}".format(cxr.columns))
    cxr =  cxr[cxr['项目名称'].str.contains('胸部')]
    print("\r CXR shape: {}".format(cxr.shape))
    cxr.drop(columns=['项目名称', '检查子类'], inplace=True)
    cxr.dropna(subset=['阳性标识', '诊断结果', '描述'], axis=0, how='any', inplace=True)
    #cxr['阳性标识'].fillna(0.0, inplace=True) #nan replace with 0
    #cxr['阳性标识'] = cxr['阳性标识'].apply(lambda x: 1.0 if x==3.0 else x) 
    print("\r CXR shape: {}".format(cxr.shape))
    print("\r Num of classes: {}".format(cxr['阳性标识'].value_counts()) )


    classes_en = {0: 'Atelectasis', 1: 'Cardiomegaly', 2: 'Effusion', 3: 'Infiltration', 4:'Mass', 5:'Nodule', 6:'Pneumonia',\
                  7:'Pneumothorax', 8:'Consolidation', 9:'Edema', 10:'Emphysema',11:'Fibrosis',12:'Pleural_Thickening',13:'Hernia'}
    classes_zn = {0: '肺扩张不全', 1: '心脏扩大', 2: '肺积液', 3: '肺部浸润', 4:'肺部块', 5:'肺结节', 6:'肺炎',\
                  7:'气胸', 8:'肺实变', 9:'肺水肿', 10:'肺气肿',11:'肺纤维化',12:'肺膜增厚',13:'肺氙'}
    def f(x):
        for i in range(len(classes_zn)):
            if classes_zn[i] in x['诊断结果'] or classes_zn[i] in x['描述']: 
                return i
            elif x['阳性标识'] == 0.0: return 0
            else: return -1 
    cxr['疾病标识'] = cxr.apply(lambda x: f(x), axis=1) 
    print("\r Num of disease: {}".format(cxr['疾病标识'].value_counts()) )
    #save 
    cxr.to_csv('/data/fjsdata/CVTEDR/CXR20201204.csv', index=False, header=True, sep=',')


if __name__ == "__main__":

    #CVTEDR_Filter()
    
    #for debug   
    data_loader_train = get_train_dataloader(batch_size=512, shuffle=True, num_workers=0)
    roi_idx = np.array([0,0,0, 1,1,1, 3,3,3, 511,510,511])
    for batch_idx, (image, label) in enumerate(data_loader_train):
         roi_label = label[roi_idx]
         print(roi_label)
    
    