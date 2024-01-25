# imports for prediction script
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

def load_model(folder,epoch,model,device):
    print()
    model_path = f'/home/jovyan/data/aurora/eye-blink-production/train/result/{folder}/epoch{epoch}.pt'
    print("model path: ",model_path)

    _ = model.to(device)
    # model=nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path, map_location = device))
    _ = model.eval()

    # # check model loaded to gpu
    # next(model.parameters()).is_cuda # returns a boolean
    return model
    
def get_softmax_scores(device, folder, epoch, test_dataset, df, model, savePath, batch_size=512):
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
    # print(f"len result: {len(result)}")

    # run softmax on the entire result first
    softmax_result=[softmax(x) for x in result]

    # get the labels of the dataset
    label_list = test_dataset.label_list.tolist()
    # print(f"len label: {len(label_list)}")

    softmax_result = [x.tolist() for x in softmax_result]

    name=f"{folder}_epoch{epoch}_softmax"
    print(name)
    df[name]=softmax_result

    # saves results to same df
    df.to_csv(savePath,index=False)
    # print(f"softmax scores for model {folder} finshed!")
    
    return softmax_result,label_list

def get_softmax_scores_one_side(side,device, folder, epoch, test_dataset, df, model, savePath, batch_size=512):
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
    # print(f"len result: {len(result)}")

    # run softmax on the entire result first
    softmax_result=[softmax(x) for x in result]

    # get the labels of the dataset
    label_list = test_dataset.label_list.tolist()
    # print(f"len label: {len(label_list)}")

    softmax_result = [x.tolist() for x in softmax_result]

    name=f"{side}_{folder}_epoch{epoch}_softmax"
    print(name)
    df[name]=softmax_result

    # saves results to same df
    df.to_csv(savePath,index=False)
    # print(f"softmax scores for model {folder} finshed!")
    
    return softmax_result,label_list



def get_argmax_prediction(softmax_result,label_list):
    argmax_prediction=[]
    for i in range(len(label_list)):
        p=softmax_result[i]
        argmax_prediction.append(np.argmax(p))
    argmax_prediction=np.array(argmax_prediction)
                
    return argmax_prediction

def get_threshold_prediction(softmax_result,label_list,thre_close = 0.6,thre_block = 0.8):
    threshold_prediction=[]
    for i in range(len(label_list)):
        p=softmax_result[i]
        if p[2]>thre_block:
            threshold_prediction.append(2)
        elif p[1]>thre_close:
            threshold_prediction.append(1)
        else:
            threshold_prediction.append(0) 
    threshold_prediction=np.array(threshold_prediction)
    
    return threshold_prediction