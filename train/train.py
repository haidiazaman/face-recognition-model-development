from cz_eye_net import *
# from eye_net import *
from eye_dataset_imgaug import Eye_Dataset
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
# from torch.utils.tensorboard import Writer
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts


def plot_result(result_list, path = './result/fig_train_2step.pdf'):
    x = range(0, len(result_list[0]))
    plt.switch_backend('agg')
    for y, c, l in zip(result_list, ['blue', 'red', 'black'], ['open', 'close', 'block']):
        plt.plot(x, y, color = c, marker = '.', label = l)
#     plt.ylim(np.min(result_list), 1)
    plt.ylim(0.8, 1)
    plt.legend()
    plt.grid()
    plt.savefig(path)
    plt.close()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_model(args):
    # load input params
    with Path(args.config).open() as f:
        args=yaml.safe_load(f)
    # input params
    input_data_folder = args['input_data_folder'] # csvpath, train val shud be in same csv
    output_folder = args['output_folder']  # (str): Folder name of output in ./result/
    balance = eval(args['balance']) # (bool): set to False if using Focal Loss
    load_model = args['load_model'] # (str): Load model in ./result/ for finetuning, set to None to train from scratch
    if load_model=='None':
        load_model=None
    num_workers = args['num_workers'] # (int): The number of workers
    seed = args['seed'] # (int): Fix random seed
    lr = args['lr'] # (float): Learning rate for training
    batch_size = args['batch_size']
    epochs = args['epochs']
    class_weights = eval(args['class_weights'])
    input_dim = args['input_dim']
    weight_decay = args['weight_decay']
    first_cycle_steps = args['first_cycle_steps']
    cycle_mult = args['cycle_mult']
    max_lr = args['max_lr']
    min_lr = args['min_lr']
    warmup_steps = args['warmup_steps']
    gamma = args['gamma']
    
    

    # initialise device
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # initialization
    if seed: setup_seed(seed)
    # initialise output path
    output_path=Path(output_folder)
    output_path.mkdir(exist_ok = True, parents = True)
    record_file = open(os.path.join(output_path, 'train2step_record.txt'), 'w')
    lrs_tracker=[np.nan]*epochs
    lrs_df = pd.DataFrame.from_dict({
        'lr':lrs_tracker
    })
    lrs_path = os.path.join(output_path, 'lrs.csv')
    lrs_df.to_csv(lrs_path,index=False)
    
    # setup model - from scratch or load model for finetune
    model = Eye_Net(in_channel = 3)
    # model_key="mobilenetv3_small_050"
    # model = Eye_Net(model_key,in_channel = 3)
    model=model.to(device)
    # print(summary(model,input_size=(3,80,80)))
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
    loss_func = FocalLoss(alpha=class_weights) # d
    # loss_func = nn.CrossEntropyLoss().cuda()
    # loss_func = nn.CrossEntropyLoss()
    loss_func=loss_func.to(device)

    # setup dataset and dataloader
    train_dataset = Eye_Dataset(input_data_folder, data_type = 'train', input_dim = input_dim,
                                seed = seed, balance = balance, augmentation = True, preload = True)
    val_dataset = Eye_Dataset(input_data_folder, data_type = 'val', input_dim = input_dim,
                              seed = seed, balance = balance, augmentation = False, preload = True)
    train_data_loader = DataLoader(train_dataset, num_workers = num_workers, batch_size = batch_size, shuffle = True, pin_memory = True)
    val_data_loader = DataLoader(val_dataset, num_workers = num_workers, batch_size = batch_size, shuffle = False, pin_memory = True)
    
    # START TRAINING LOOP
    torch.autograd.set_detect_anomaly(True)
    model.train()
    result_list = [[] for _ in range(3)]
    train_loss = np.inf
    for epoch in range(epochs):

        if epoch >= 0:
            model.eval()

            result_mat = np.ones((3, 3)) * 1e-10

            torch.save(model.module.state_dict(), os.path.join(output_path, 'epoch{0:03d}.pt'.format(epoch))) 
            val_loss = 0.0

            for index, (image, label) in enumerate(alive_it(val_data_loader)):
                # input_image = image.cuda(non_blocking = True)
                # y_label = label.cuda(non_blocking = True)
                input_image = image.to(device)
                y_label = label.to(device)


                with torch.no_grad():
                    score = model(input_image)
                    val_loss += float(loss_func(score, y_label)) * len(label)

                    y_prediction = torch.argmax(score, 1).cpu().detach().numpy()
                    y_label = y_label.cpu().detach().numpy()

                    for i, j in zip(y_label, y_prediction):
                        result_mat[i][j] += 1

            for i in range(3):
                result_list[i].append((result_mat[i][i] - 1e-10) / sum(result_mat[i]))
            result_log = ('\nepoch %3g : train loss = %.5e, val loss = %.5e,\n' + 
                            '            Acc-open: %3.2f%%, Acc-close: %3.2f%%, Acc-block: %3.2f%%\n') \
            % (epoch, train_loss / len(train_dataset), val_loss / len(val_dataset), result_list[0][-1] * 100, 
               result_list[1][-1] * 100, result_list[2][-1] * 100)
            result_log += 'confusion matrix:\n' + np.array2string(result_mat.astype(int)) + '\n'
            print(result_log)
            record_file.write(result_log)
            record_file.flush()

            plot_result(result_list, path = os.path.join(output_path, 'fig_train_2step.pdf'))

        model.train()
        train_loss = 0.0
        for index, (image, label) in enumerate(alive_it(train_data_loader)):
            # input_image = image.cuda(non_blocking = True)
            # y_label = label.cuda(non_blocking = True)
            input_image = image.to(device)
            y_label = label.to(device)

            score = model(input_image)
            loss = loss_func(score, y_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            train_loss += float(loss) * len(label)

        scheduler.step()
        lrs_df.loc[epoch+1,'lr']=scheduler.get_lr()[0]
        # lrs_df.loc[epoch+1,'lr']=scheduler.get_last_lr()[0]
        lrs_df.to_csv(lrs_path,index=False)

    print('Finished!!!')
    record_file.close()
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Pass input params to separate config file in .json format and then call it using parser")
    parser.add_argument("--config", type=str, help="Path to JSON config file.")
    args = parser.parse_args()
    train_model(args)
