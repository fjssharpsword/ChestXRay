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
                    image_name = os.path.join(path_to_img_dir, items[0])
                    if os.path.isfile(image_name) == True:
                        label = int(eval(items[1])) #eval for 
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
   transforms.Resize((256,256)),
   transforms.RandomCrop(224),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
])
transform_seq_test = transforms.Compose([
   transforms.Resize((256,256)),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
])
transform_seq_sample = transforms.Compose([
   transforms.Resize((256,256)),
   transforms.FiveCrop(224),
   transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
   transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(crop) for crop in crops])),
])

PATH_TO_IMAGES_DIR = '/data/fjsdata/CVTEDR/images'
PATH_TO_TRAIN_FILE = '/data/fjsdata/CVTEDR/cxr_train.txt'
PATH_TO_VAL_FILE = '/data/fjsdata/CVTEDR/cxr_val.txt'
PATH_TO_TEST_FILE = '/data/fjsdata/CVTEDR/cxr_test.txt'

def get_train_dataloader(batch_size, shuffle, num_workers):
    dataset_train = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                    path_to_dataset_file=[PATH_TO_TRAIN_FILE, PATH_TO_VAL_FILE], transform=transform_seq_train)
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

def getDicomImage(dicom_path, dataset_path):

    image_path = '/data/fjsdata/CVTEDR/images'
    datas = pd.read_csv(dataset_path, sep=',')
    dicoms = np.array(datas['图片路径']).tolist()
    labels =np.array(datas['阳性标识']).tolist()
    print('image number:{}'.format(len(dicoms)))

    images = []
    labels_new = []
    for idx in range(len(dicoms)):
        #sample: image\DX\20190124\DR190124024_1.2.156.600734.2466462228.11372.1548290939.55
        path = dicoms[idx].split('\\')[-1]
        dir = path.split('_')[0]
        file = path.split('_')[1]
        series_path = os.path.join(dicom_path, dir, file)
        if os.path.isdir(series_path) == False: continue 
        image_name = dir+'.jpeg'
        images.append(image_name)
        labels_new.append(labels[idx])
        if os.path.isfile(os.path.join(image_path, image_name)) == True: continue
        
        try:
            lstFilesDCM = []
            for root, dirs, files in os.walk(series_path):
                for file in files:
                    lstFilesDCM.append(os.path.join(root, file))
            slices = [pydicom.read_file(s) for s in lstFilesDCM]
            # filter PA/AP, SeriesDescription
            # the front view and lateral view can be chosen by model with MIMIC-CXRv2.0 dataset 
            si, ss = 0, slices[0]
            for i, s in enumerate(slices):
                if 'PA' in s.SeriesDescription or 'AP' in s.SeriesDescription:
                    si, ss = i, slices[i]

            sitk_image = sitk.ReadImage(lstFilesDCM[si])
            img = sitk.GetArrayFromImage(sitk_image)
            img = np.squeeze(img, axis=0)

            img = (img-np.min(img))/(np.max(img)-np.min(img)) * 255
            img = Image.fromarray(img.astype('uint8')).convert('RGB')#numpy to PIL

            #ss.PhotometricInterpretation: 'MONOCHROME1'=flip and 'MONOCHROME2'=normal
            if 'MONOCHROME1' in ss.PhotometricInterpretation:
                img = PIL.ImageOps.invert(img) #flip the white and black, RGB

            #store
            img.save(os.path.join(image_path, image_name),"JPEG", quality=95, optimize=True, progressive=True)
        except Exception as e:
                print("Unable to read file. %s" % e)
                continue

        sys.stdout.write('\r Image ID {} and path {}'.format((idx+1), series_path))
        sys.stdout.flush()

    image_name = pd.DataFrame(images, columns=['name'])
    image_label = pd.DataFrame(labels_new, columns=['label'])
    cxr = pd.concat([image_name, image_label],axis=1)
    print("\r dataset shape: {}".format(cxr.shape)) 
    print("\r Num of disease: {}".format(cxr['label'].value_counts()) )
    cxr.to_csv('/data/fjsdata/CVTEDR/CXR20201210.csv', index=False, header=True, sep=',')

def verfiyImage(dataset_path, image_path):
    datas = pd.read_csv(dataset_path, sep=',')
    names = np.array(datas['name']).tolist()
    labels =np.array(datas['label']).tolist()
    print('image number:{}'.format(len(names)))
    for name in names:
        if os.path.isfile(os.path.join(image_path, name)) == False: 
            print(name) #DR170210009.jpeg  deleted


def splitCVTEDR(dataset_path): 
    
    datas = pd.read_csv(dataset_path, sep=',')
    #datas = datas.drop(datas[datas['label']==3.0].index)
    datas['label'] = datas['label'].apply(lambda x: 1.0 if x==3.0 else x)
    print("\r dataset shape: {}".format(datas.shape)) 
    print("\r dataset distribution: {}".format(datas['label'].value_counts()))
    images = datas[['name']]
    labels = datas[['label']]
    #print("\r CXR Columns: {}".format(datas.columns))
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.20, random_state=11)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=22)
    print("\r trainset shape: {}".format(y_train.shape)) 
    print("\r trainset distribution: {}".format(y_train['label'].value_counts()))
    print("\r valset shape: {}".format(y_val.shape)) 
    print("\r trainset distribution: {}".format(y_val['label'].value_counts()))
    print("\r testset shape: {}".format(y_test.shape)) 
    print("\r trainset distribution: {}".format(y_test['label'].value_counts()))
    trainset = pd.concat([X_train, y_train], axis=1).to_csv('/data/fjsdata/CVTEDR/cxr_train.txt', index=False, header=False, sep=',')
    valset = pd.concat([X_val, y_val], axis=1).to_csv('/data/fjsdata/CVTEDR/cxr_val.txt', index=False, header=False, sep=',')
    testset = pd.concat([X_test, y_test], axis=1).to_csv('/data/fjsdata/CVTEDR/cxr_test.txt', index=False, header=False, sep=',')

def splitCVTEDR2(dataset_path, pos_dataset_path): 
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

    #merge negative and positive sample
    pos_datas = pd.DataFrame(pos_images_new, columns=['name'])
    pos_datas['label'] = 1

    datas = datas.sample(n=10*len(pos_datas), random_state=1) #random sampling 10 times for negative
    datas = pd.concat([datas, pos_datas], axis=0)
    print("\r dataset shape: {}".format(datas.shape)) 
    print("\r dataset distribution: {}".format(datas['label'].value_counts()))

    #split train, validation, test
    images = datas[['name']]
    labels = datas[['label']]
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.20, random_state=11)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=22)
    print("\r trainset shape: {}".format(y_train.shape)) 
    print("\r trainset distribution: {}".format(y_train['label'].value_counts()))
    print("\r valset shape: {}".format(y_val.shape)) 
    print("\r valset distribution: {}".format(y_val['label'].value_counts()))
    print("\r testset shape: {}".format(y_test.shape)) 
    print("\r testset distribution: {}".format(y_test['label'].value_counts()))
    trainset = pd.concat([X_train, y_train], axis=1).to_csv('/data/fjsdata/CVTEDR/cxr_train.txt', index=False, header=False, sep=',')
    valset = pd.concat([X_val, y_val], axis=1).to_csv('/data/fjsdata/CVTEDR/cxr_val.txt', index=False, header=False, sep=',')
    testset = pd.concat([X_test, y_test], axis=1).to_csv('/data/fjsdata/CVTEDR/cxr_test.txt', index=False, header=False, sep=',')

if __name__ == "__main__":

    #CVTEDR_Filter()
    #getDicomImage('/data/fjsdata/CVTEDR/dicoms', '/data/fjsdata/CVTEDR/CXR20201204.csv')
    #verfiyImage('/data/fjsdata/CVTEDR/CXR20201210.csv', '/data/fjsdata/CVTEDR/images')
    #splitCVTEDR2('/data/fjsdata/CVTEDR/CXR20201210.csv')
    #splitCVTEDR2('/data/fjsdata/CVTEDR/CXR20201210.csv', '/data/fjsdata/CVTEDR/CVTE-DR-Pos-954.csv')
    
    
    #for debug   
    data_loader_train = get_train_dataloader(batch_size=10, shuffle=True, num_workers=0)
    for batch_idx, (image, label) in enumerate(data_loader_train):
        print(batch_idx)
        print(image.shape)
        print(label.shape)
    
    
        
    
    
    
    