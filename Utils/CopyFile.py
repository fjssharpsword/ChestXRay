import pandas as pd
import numpy as np
import os
import shutil
import sys

def main(dataset_path):
    #locate the directory of DICOM file
    roots =['/data_share/hospital_data/cvte_health_center/CR/CVTE_CR_Deal2016/',
           '/data_share/hospital_data/cvte_health_center/CR/CVTE_CR_Deal2017/',
           '/data_share/hospital_data/cvte_health_center/CR/CVTE_CR_Deal2018/',
           '/data_share/hospital_data/cvte_health_center/CR/CVTE_CR_Deal2019/',
           '/data_share/hospital_data/cvte_health_center/DR/CVTE_DX_Deal/',
           '/data_share/hospital_data/cvte_health_center/DR/CVTE_DX_Deal2020/'
          ]
    paths = []
    dirs = []
    for root in roots:
        for dir in os.listdir(root):
            if 'DR' in dir:
                dirs.append(dir)
                paths.append(os.path.join(root, dir))
                sys.stdout.write('\r length of directorys: = {}'.format(len(dirs)))
                sys.stdout.flush()
    #copy file
    des_path ='/data/fjsdata/CVTEDR/dicoms/'
    datas = pd.read_csv(dataset_path, sep=',')
    images = np.array(datas['图片路径']).tolist()
    for image in images:
        #sample: image\DX\20190124\DR190124024_1.2.156.600734.2466462228.11372.1548290939.55
        path = image.split('\\')[-1]
        dir = path.split('_')[0]
        if 'DR' not in dir: continue
        ori_path = paths[dirs.index(dir)]
        try:
            shutil.copytree(ori_path, os.path.join(des_path, dir)) 
        except IOError as e:
            print("Unable to copy file. %s" % e)
            continue
        except:
            print("Unexpected error:", sys.exc_info())
            continue
        sys.stdout.write('\r Directory {} have been copied.'.format(ori_path))
        sys.stdout.flush()

if __name__ == "__main__":
    main('/data/fjsdata/CVTEDR/CXR20201204.csv')