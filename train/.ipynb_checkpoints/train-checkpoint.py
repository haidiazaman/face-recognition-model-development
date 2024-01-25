from dataloader import *
from model import *
from losses import FocalLoss

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
from torchsummary import summary
import yaml
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import time
# from torch.utils.tensorboard import Writer
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# def plot_result(result_list, path = './result/fig_train_2step.pdf'):
#     x = range(0, len(result_list[0]))
#     plt.switch_backend('agg')
#     for y, c, l in zip(result_list, ['blue', 'red', 'black'], ['open', 'close', 'block']):
#         plt.plot(x, y, color = c, marker = '.', label = l)
#     plt.ylim(0.8, 1)
#     plt.legend()
#     plt.grid()
#     plt.savefig(path)
#     plt.close()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    

# to calc time taken per epoch and entire training time
def convert_seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return int(hours), int(minutes), int(seconds)

# accuracy function
def get_accuracy(outputs,labels):
    predicted_classes = torch.argmax(nn.Softmax(dim=1)(outputs),dim=1).cpu().detach().numpy()
    correct_predictions=(predicted_classes==labels.cpu().detach().numpy()).sum().item()
    accuracy = correct_predictions/labels.size(0) * 100
    return accuracy

# def train_model():
def train_model(args):
    # load input params
    with Path(args.config).open() as f:
        args=yaml.safe_load(f)
    # input params   
    input_data_folder = args["input_data_folder"] # csvpath, train val shud be in same csv
    label_int_mapping_path = args["label_int_mapping_path"]
    output_folder = args["output_folder"] # (str): Folder name of output in ./result/
    input_dim = args["input_dim"]
    seed = args["seed"]
    pretrained = args["pretrained"]
    load_model = args["load_model"] # (str): Load model in ./result/ for finetuning, set to None to train from scratch
    if load_model == 'None':
        load_model = None
    num_workers = args["num_workers"]
    lr = args["lr"]
    batch_size = args["batch_size"]
    epochs = args["epochs"]
    weight_decay = args["weight_decay"]
    first_cycle_steps = args["first_cycle_steps"]
    cycle_mult = args["cycle_mult"]
    max_lr = args["max_lr"]
    min_lr = args["min_lr"]
    warmup_steps = args["warmup_steps"]
    gamma = args["gamma"]
    

    # initialise device
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # initialization
    if seed: setup_seed(seed)
    # initialise output path
    output_path=Path(output_folder)
    output_path.mkdir(exist_ok = True, parents = True)
    weights_folder_path=os.path.join(output_path,output_folder,'weights')
    weights_folder_path=Path(weights_folder_path)
    weights_folder_path.mkdir(exist_ok = True, parents = True)
    
    lrs_tracker=[np.nan]*epochs
    lrs_df = pd.DataFrame.from_dict({
        'lr':lrs_tracker
    })
    lrs_path = os.path.join(output_path, 'lrs.csv')
    lrs_df.to_csv(lrs_path,index=False)
    
    
    
    # setup dataset and dataloader
    train_dataset = LFW_Dataset(input_data_folder, label_int_mapping_path, split_type = 'train', input_dim = input_dim ,seed = seed, augmentation = True, preload = False)
    val_dataset = LFW_Dataset(input_data_folder, label_int_mapping_path, split_type = 'val', input_dim = input_dim ,seed = seed, augmentation = False, preload = False)
    train_data_loader = DataLoader(train_dataset, num_workers = num_workers, batch_size = batch_size, shuffle = True, pin_memory = True)
    val_data_loader = DataLoader(val_dataset, num_workers = num_workers, batch_size = batch_size, shuffle = False, pin_memory = True)

    
    # get num_classes - this has to be done dynamically since the num classes is num of unique persons in dataset
    class_mapping = train_dataset.label_mapping
    num_classes = len(class_mapping.values())
    print("num_classes: ",num_classes)
    class_weights = train_dataset.class_weights
    
    # setup model - from scratch or load model for finetune
    model = FaceNet(num_classes=num_classes,pretrained='vggface2')

    model=model.to(device)
    if load_model:
        model.load_state_dict(torch.load(os.path.join('./result', load_model), map_location = device))
        _ = model.eval()
    if device==torch.device('cuda'):
        model = nn.DataParallel(model)

    # setup optimizer and LR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10)
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                              first_cycle_steps=first_cycle_steps,
                                              cycle_mult=cycle_mult,
                                              max_lr=max_lr,
                                              min_lr=min_lr,
                                              warmup_steps=warmup_steps,
                                              gamma=gamma)
    
    # setup loss function - focal loss + no balance or crossentropyloss + balance
    class_weights = torch.tensor(class_weights, device=device) # change to 1-%of that class
    loss_func = FocalLoss(alpha=class_weights) 
    loss_func=loss_func.to(device)

    # setup output csv to store train and val ave loss and acc for each epoch
    output_dict = {
        'epoch': [epoch for epoch in range(epochs)],
        'train_loss': [0. for epoch in range(epochs)],
        'val_loss': [0. for epoch in range(epochs)],
        'train_acc': [0. for epoch in range(epochs)],
        'val_acc': [0. for epoch in range(epochs)],
        'train_time_taken': [0. for epoch in range(epochs)],
        'val_time_taken': [0. for epoch in range(epochs)],
    }
    output_df=pd.DataFrame.from_dict(output_dict)
    output_df_path = output_folder + "/results.csv"
    output_df.to_csv(output_df_path,index=False)

    torch.autograd.set_detect_anomaly(True)
    print('######################')
    print('######################')
    print('### START TRAINING ###')
    print()

    training_start_time=time.time()
    best_val_loss = float('inf')  # Initialize with positive infinity or any large value
    for epoch in range(epochs):
        #training loop
        model.train()
        train_running_loss,train_running_acc,train_start_time=0.,0.,time.time()
        for ind,(images,labels) in enumerate(alive_it(train_data_loader)):
            images=images.to(device)
            labels=labels.to(device)
            outputs=model(images)
            train_loss=loss_func(outputs,labels)
            train_acc=get_accuracy(outputs,labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_running_loss+=train_loss.item()
            train_running_acc+=train_acc
        train_time_taken=time.time()-train_start_time
        epoch_ave_train_loss=train_running_loss/(ind+1)
        epoch_ave_train_acc=train_running_acc/(ind+1)

        #validation loop
        model.eval()
        val_running_loss,val_running_acc,val_start_time=0.,0.,time.time()
        val_conf_matrix = np.zeros((num_classes,num_classes))
        for ind,(images,labels) in enumerate(alive_it(val_data_loader)):
            images=images.to(device)
            labels=labels.to(device)
            with torch.no_grad():
                outputs=model(images)
                val_loss=loss_func(outputs,labels)
                val_acc=get_accuracy(outputs,labels)
                val_running_loss+=val_loss.item()
                val_running_acc+=val_acc
                # y_pred = torch.argmax(outputs,dim=1).cpu().detach().numpy()
                y_pred = torch.argmax(nn.Softmax(dim=1)(outputs),dim=1).cpu().detach().numpy()
                y_label = labels.cpu().detach().numpy()
                for i,j in zip(y_label,y_pred):
                    val_conf_matrix[i][j]+=1
        val_time_taken=time.time()-val_start_time
        epoch_ave_val_loss=val_running_loss/(ind+1)
        epoch_ave_val_acc=val_running_acc/(ind+1)

        # val_conf_matrix=val_conf_matrix.astype(int)
        # val_acc_per_class_dict={}
        # for i in range(num_classes):
        #     class_name=class_mapping[i]
        #     class_acc=val_conf_matrix[i][i]/val_conf_matrix[i].sum()*100
        #     val_acc_per_class_dict[class_name]=class_acc    
        # output_per_class_acc_string = ', '.join([f'Acc-{class_name}: {round(acc,2)}%' for class_name, acc in val_acc_per_class_dict.items()])

        print()
        print(f'Epoch {epoch}, Training loss: {epoch_ave_train_loss:.4f}, Training acc: {epoch_ave_train_acc:.4f}, Validation loss: {epoch_ave_val_loss:.4f}, Validation acc: {epoch_ave_val_acc:.4f}')
        # print(f'Validation {output_per_class_acc_string}')
        # print(f'Validation confusion matrix:\n{val_conf_matrix}')
        # print()

        #save values to df
        output_df.loc[epoch,'epoch']=epoch
        output_df.loc[epoch,'train_loss']=round(epoch_ave_train_loss,4)
        output_df.loc[epoch,'val_loss']=round(epoch_ave_val_loss,4)
        output_df.loc[epoch,'train_acc']=round(epoch_ave_train_acc,2)
        output_df.loc[epoch,'val_acc']=round(epoch_ave_val_acc,2)
        output_df.loc[epoch,'train_time_taken']=round(train_time_taken,2)
        output_df.loc[epoch,'val_time_taken']=round(val_time_taken,2)

        # output_df.loc[epoch,'val_acc_real']=round(val_acc_per_class_dict['real'],2)
        # output_df.loc[epoch,'val_acc_deepfake']=round(val_acc_per_class_dict['deepfake'],2)
        # for x in val_conf_matrix.ravel().tolist():
        #     output_df.loc[epoch,'val_conf_mat'].append(x) 
        output_df.to_csv(output_df_path,index=False)


        # # TENSORBOARD - plot graphs
        # writer.add_scalar('train/training_loss',epoch_ave_train_loss,epoch)
        # writer.add_scalar('val/validation_loss',epoch_ave_val_loss,epoch)
        # writer.add_scalar('train/training_acc',epoch_ave_train_acc,epoch)
        # writer.add_scalar('val/validation_acc',epoch_ave_val_acc,epoch)
        # writer.add_scalar('learning_rate_scheduler',scheduler.get_lr()[0],epoch)

        # Save the model if the current validation loss is the best so far - this will only save the model not overwrite previous
        # print('ADD IN CODE SNIPPET TO SAVE ONLY 5 BEST MODELS VAL LOSS')
        if epoch_ave_val_loss < best_val_loss:
            best_val_loss = epoch_ave_val_loss
            torch.save(model.module.state_dict(), os.path.join(weights_folder_path,f'best_model.pt'))

        scheduler.step()
        
    print()
    print()
    print('### COMPLETED TRAINING ###')
    total_time_taken=time.time()-training_start_time
    h,m,s=convert_seconds_to_hms(total_time_taken)
    print(f'total time taken: {h}h, {m}m, {s}s')
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Pass input params to separate config file in .json format and then call it using parser")
    parser.add_argument("--config", type=str, help="Path to JSON config file.")
    args = parser.parse_args()
    train_model(args)