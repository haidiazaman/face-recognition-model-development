import torch
import os
import cv2
# cv2.setNumThreads(0)
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torchvision.transforms import Compose, ToTensor
import random
from glob import glob
from alive_progress import alive_it
from alive_progress import config_handler
config_handler.set_global(length = 20, force_tty = True)
import imgaug as ia
import imgaug.augmenters as iaa
from tqdm.contrib.concurrent import process_map
import albumentations as A
import pickle


class LFW_Dataset(Dataset):
    def __init__(self, csv_path, label_int_mapping_path, split_type = '', input_dim = 512 ,seed = None, augmentation = False, preload = False):        
        self.split_type = split_type
        self.input_dim = input_dim
        self.augmentation = augmentation
        self.preload = preload
        
        self.label_mapping = self.load_dict_from_pickle(label_int_mapping_path)
        self.df=pd.read_csv(csv_path)
        if split_type:
            self.df = self.df[self.df['split_type'] == split_type].reset_index()      
        if seed: self.setup_seed(seed)
        
        # setup input transform applicable to all images
        self.input_transform = Compose([
            ToTensor(),
        ])
        
        # uncomment for sanity check
        # self.df = self.df.drop_duplicates(subset='person_name', keep='first')

        # self.df=self.df.iloc[:100]
        # self.df=self.df.iloc[:1000]
        
        self.image_list = np.array(self.df['image_path'])
        self.bbox_list = np.round(np.array([list(map(float, b.strip('[ ]').split(', '))) if len(b) > 2 else eval(b)
                                            for b in list(self.df['bbox'])])).astype(int)
        self.label_list = np.array(self.df['label'].tolist())
        self.class_weights = (1 - self.df.label.value_counts()/len(self.df)).tolist()
        
        if self.preload:
            self.image_preload = []
            print('Preload images...')
            input_params = []
            for index in range(len(self.image_list)):
                input_params.append({
                    'path': self.image_list[index],
                    'bbox': self.bbox_list[index],
                    'input_dim': self.input_dim
                })
            self.image_preload = process_map(self.load_image, input_params, max_workers=30, chunksize=1)
        self.length = len(self.image_list)
        
            
    def __len__(self):
        return len(self.df)

    def __getitem__(self,ind):
        image_path=self.image_list[ind]
        bbox=self.bbox_list[ind]     
        label=self.label_list[ind]
        if self.preload:
            image=self.image_preload[ind]
        else:
            input_params={
                'image_path': image_path,
                'bbox': bbox,
                'input_dim':self.input_dim
            }
            image=self.load_image(input_params)
        if self.augmentation:
            image=self.apply_augmentations(image)
        image=self.apply_input_transforms(image) #applies to all split type, train val test
        return image,label
            
    def apply_input_transforms(self,image): #apply /255 and totensor for val and test, can apply for train after augmentations
        transforms=Compose([
            ToTensor()
        ])
        image=transforms(image)
        return image 

    def apply_augmentations(self,image):
        # Define a pipeline of augmentations
        transform=A.Compose([
                A.SomeOf([
                    A.HorizontalFlip(p=0.5),
                    A.OneOf([
                        A.GaussNoise(var_limit=100.0, p=1.0),
                        A.MultiplicativeNoise(multiplier=(0.95, 1.05), p=1.0),
                        A.GaussianBlur(p=1.0),
                        A.MotionBlur(p=1.0),
                    ], p=0.1),
                    A.OneOf([
                        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.33),
                        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=0.33),
                        A.RGBShift(p=0.33)
                    ], p=0.2),
                    A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.2),
                    A.OneOf([
                        A.CoarseDropout (max_holes=16, max_height=8, max_width=8, p=0.5), # adjust this, reduce max height and max width to 8
                        A.PixelDropout(dropout_prob=0.02, p=0.5)
                    ], p=0.2),
                    A.OneOf([
                        A.CLAHE(p=1.0),
                        A.RandomToneCurve(p=1.0),
                    ], p=0.2),
                    A.OneOf([
                        A.Downscale(scale_min=0.8, scale_max=0.95, interpolation=cv2.INTER_AREA, p=1.0),
                        A.ImageCompression(quality_lower=75, quality_upper=95, p=1.0)
                    ], p=0.1),
                    # A.ToGray(p=0.1)
                ], n=4, replace=False),
                # should change to your dataset values
                # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # ToTensorV2()
            ])
        augmented = transform(image=image)
        augmented_image = augmented['image']
        return augmented_image
    
    
    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


    def load_image(self,input_param):
        expand_ratio = 0
        image = cv2.cvtColor(cv2.imread(input_param['image_path']), cv2.COLOR_BGR2RGB)
        l, t, r, b = input_param['bbox']
        w, h = r - l, b - t
        l, r = max(0, l - int(expand_ratio * w / 2)), min(image.shape[1] - 1, r + int(expand_ratio * w / 2))
        t, b = max(0, t - int(expand_ratio * h / 2)), min(image.shape[0] - 1, b + int(expand_ratio * h / 2))
        input_dim = input_param['input_dim']
        image = cv2.resize(image[t:b, l:r], (input_dim, input_dim))
        return image

    def load_dict_from_pickle(self,file_path):
        with open(file_path, 'rb') as file:
            loaded_dict = pickle.load(file)
        return loaded_dict 