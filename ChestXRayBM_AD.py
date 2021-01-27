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
import re
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
    def __init__(self, path_to_img_dir, path_to_dataset_file, is_train = True):
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
                    image_name= items[0]#.split('/')[1]
                    label = list(items[1].replace(' ', ''))[1:15]
                    label = [int(eval(i)) for i in label]
                    if is_train == False: #for test set
                        if np.sum(label)==0: #normal
                            labels.append([0])
                        else:
                            labels.append([1])
                        image_name = os.path.join(path_to_img_dir, image_name)
                        image_names.append(image_name)
                    else: #for train set
                        if np.sum(label)==0: #normal
                            labels.append([0])
                            image_name = os.path.join(path_to_img_dir, image_name)
                            image_names.append(image_name)
                        
        self.image_names = image_names
        self.labels = labels

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
        mask = transform_seq_test(image)
        image = transform_seq_train(image)
        return image, mask, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)

#config 
transform_seq_test = transforms.Compose([
   transforms.Resize((256,256)),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
])
transform_seq_train = transforms.Compose([
   transforms.Resize((256,256)),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
])

PATH_TO_IMAGES_DIR = '/data/fjsdata/NIH-CXR/images/images/'
PATH_TO_TRAIN_VAL_FILE = './Dataset/bm_train_val.csv'
PATH_TO_TEST_FILE = './Dataset/bm_test.csv'

def get_test_dataloader(batch_size, shuffle, num_workers):
    dataset_test = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                    path_to_dataset_file=[PATH_TO_TEST_FILE], is_train=False)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_test

#for cross-validation
def get_train_dataloader(batch_size, shuffle, num_workers):
    dataset_test = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                    path_to_dataset_file=[PATH_TO_TRAIN_VAL_FILE], is_train=True)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_test


if __name__ == "__main__":

    #for debug   
    data_loader = get_train_dataloader(batch_size=10, shuffle=False, num_workers=0)
    for batch_idx, (image, label) in enumerate(data_loader):
        print(label.shape)
        break