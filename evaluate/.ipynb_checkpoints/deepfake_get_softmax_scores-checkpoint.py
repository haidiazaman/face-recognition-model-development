# imports
import sys
import torch
# from iso_eye_net import *
from cz_eye_net import *
from eye_dataset import Eye_Dataset
import argparse,torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
from glob import glob
from alive_progress import alive_it
from alive_progress import config_handler
config_handler.set_global(length = 20, force_tty = True)
from pathlib import Path
from scipy.special import softmax
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import cv2

csvPath='/home/jovyan/data/aurora/eye-blink-production/data/v2.20/v2.20_data_test_results.csv'
# csvPath='/home/jovyan/data/aurora/eye-blink-production/data/v2.20/v2-Copy1.20_data_test_results.csv'
# csvPath='/home/jovyan/data/aurora/eye-blink-production/data/v2.20/v2.20_data_full_results.csv'
# csvPath='/home/jovyan/data/aurora/eye-blink-production/data/new_prod_data/new_prod_for_training.csv'
df = pd.read_csv(csvPath)
print(df[["split_type","label"]].value_counts())
print()

print("benchmark data csvPath: ",csvPath)
print()
test_dataset = Eye_Dataset(csvPath, data_type = '',
                           balance = False, augmentation = False, preload = True)
print()
print("len of test dataset: ",test_dataset.__len__())
    
all_pairs=[
    
    # ['v2.9.2','800'],
    # ['v2.9.2','521'],
    # ['v2.9.2','994'],
    # ['v2.9.2','236'],
    # ['v2.9.2','877'],
    
    # ['v2.9.3','877'],
    # ['v2.9.3','400'],
    # ['v2.9.3','616'],
    # ['v2.9.3','985'],
    # ['v2.9.3','125'],
    ['v2.9.3','699'],
    ['v2.9.3','985'],
    ['v2.9.4','656'],
    
#     ['v2.9.4','656'],
#     ['v2.9.4','516'],
#     ['v2.9.4','364'],
#     ['v2.9.4','784'],
#     ['v2.9.4','446'],
    
#     ['v2.9.5','980'],
#     ['v2.9.5','786'],
#     ['v2.9.5','400'],
#     ['v2.9.5','557'],
#     ['v2.9.5','617'],
    # ['v2.9.2','762'],
    # ['v2.9.2','652'],
    # ['v2.9.3','699'],
    # ['v2.9.3','989'],
    # ['v2.9.4','919'],
    # ['v2.9.4','588'],
    # ['v2.7.7.1','017'], 
    # ['v2.7.7.1','097'], 
    # ['v2.7.10.1','900'],
    # ['v2.7.10.1','942'],
    # ['v2.7.12.1','807'],
    # ['v2.7.12.1','835'],
]
for pair in all_pairs:
    folder=pair[0]
    epoch=pair[1]
# for i in range(1):
    # if i==0:
    #     folder='v2.9.2'
    #     epoch="652"
    # if i==0:
    #     folder='v2.8.1'
    #     epoch="906"
    # elif i==1:
    #     folder='v2.8.2'
    #     epoch="948"
    # elif i==2:
    #     folder='v2.8.3'
    #     epoch="953"
    # elif i==3:
    #     folder='v2.8.4'
    #     epoch="991"
    # elif i==4:
    #     folder='v2.8.5'
    #     epoch="939"
    # elif i==5:
    #     folder='v2.8.6'
    #     epoch="911"
    # elif i==6:
    #     folder='v2.8.7'
    #     epoch="953"
    # elif i==7:
    #     folder='v2.8.8'
    #     epoch="991"
        
    print()
    model_path = f'/home/jovyan/data/aurora/eye-blink-production/train/result/{folder}/epoch{epoch}.pt'
    print("model path: ",model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    print("device: ",device)

    # model = Eye_Net(model_key='mobilenetv3_small_050',in_channel = 3)
    model = Eye_Net(in_channel = 3)
    
    _ = model.to(device)

    # model=nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path, map_location = device))
    _ = model.eval()

    # check model loaded to gpu
    next(model.parameters()).is_cuda # returns a boolean


    batch_size = 512
    batch_num = int(np.ceil(len(test_dataset) / batch_size))
    result = []
    for i in alive_it(range(batch_num)):
        image_batch = []
        for j in range(batch_size * i, min(batch_size * (i + 1), len(test_dataset))):
            image_batch.append(test_dataset[j][0].unsqueeze(0))
        image_batch = torch.concat(image_batch)
        image_batch = image_batch.to(device)
        tmp = model(image_batch).cpu().detach().numpy()
        result.append(tmp)
    result = np.concatenate(result)
    print(f"len result: {len(result)}")

    # run softmax on the entire result first
    softmax_result=[softmax(x) for x in result]

    # get the labels of the dataset
    label=test_dataset.label_list
    print(f"len label: {len(label)}")

    softmax_result = [x.tolist() for x in softmax_result]

    name=f"{folder}_epoch{epoch}_softmax"
    print(name)
    df[name]=softmax_result

    df.to_csv(csvPath,index=False)

    print(f"softmax scores for model {folder} finshed!")

