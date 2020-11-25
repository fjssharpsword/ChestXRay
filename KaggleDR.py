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
Dataset: Diabetic Retinopathy Detection
https://www.kaggle.com/c/diabetic-retinopathy-detection/data
1) ??
2）Label:0 - No DR, 1 - Mild, 2 - Moderate, 3 - Severe, 4 - Proliferative DR
"""