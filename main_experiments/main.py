import argparse
import glob
import os
import time
from time import sleep

import numpy as np
import random
import torch
import torch.nn.functional as F
from models import Model
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import logging
from pathlib import Path
import json
import nni
import time
import datetime
from torch.utils.tensorboard import SummaryWriter
from thop import profile
import pynvml
import os
import networkx as nx

import socket
def client_send(device, state):
    hostname ='127.16.55.120' #'172.17.0.2'
    port = 4502
    addr = (hostname,port)

    clientsock = socket.socket()
    clientsock.connect(addr)

    if state == 0:
        say = 'GPU:{}; State:STOP'.format(device)
    if state == 1:
        say = 'GPU:{}; State:RUN'.format(device)
    clientsock.send(say.encode('utf-8'))
    clientsock.close()

pynvml.nvmlInit()

def get_memory(gpuid):
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpuid)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    memory_GB = meminfo.free/1024**2
    return memory_GB

def get_gpuid():
    # for i in range(8):
    #     client_send(i, 0)
    sleep(10)

    free_memory=[]
    for i in range(8):
        free_memory.append(get_memory(i))
    
    # for i in range(8):
    #         client_send(i, 1)
    
    gpuid=np.argmax(free_memory)
    return gpuid

def SetSeed(seed,det=True):
    """function used to set a random seed
    Arguments:
        seed {int} -- seed number, will set to torch, random and numpy
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    if det:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def count_parameters(model):
    return np.sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_args():
    parser = argparse.ArgumentParser()
    ############# system params ##############
    parser.add_argument('--seed', type=int, default=11, help='random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
    parser.add_argument('--epochs', type=int, default=500, help='maximum number of epochs')
    parser.add_argument('--patience', type=int, default=30, help='patience for early stopping')
    parser.add_argument('--ex_name',default='debug')

    ############## hyper-params ##############
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--nhid', type=int, default=128, help='hidden size')
    parser.add_argument('--pooling_nodes', type=float, default=0.8, help='pooling ratio')
    parser.add_argument('--pooling_edges', type=float, default=0.7, help='pooling ratio')
    parser.add_argument('--num_layers',default=5,type=int)
    parser.add_argument('--K',default=3,type=int)
    parser.add_argument('--decoupled',default=False)

    ############## experiments ###############
    parser.add_argument('--dataset', type=str, default='NCI1', help='DD/PROTEINS/NCI1/NCI109/Mutagenicity/ENZYMES/FRANKENSTEIN')
    parser.add_argument('--conv',default='LiCheb',help='GCN/ChebConv/GAT/LiCheb/LiMixhop/Mixhop/AttConv')
    parser.add_argument('--pool',default='LookHops',help='NoPool/TopkPool/SAGPool/EdgePool/ASAPPool/LookHops')
    args = parser.parse_args(args=[])
    return args


def train():
    min_loss = 1e10
    max_acc=0
    patience_cnt = 0
    val_loss_values = []
    val_acc_values = []
    best_epoch = 0

    t = time.time()
    model.train()
    step=0
    for epoch in range(args.epochs):
        torch.cuda.empty_cache()
        loss_train = 0.0
        correct = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(args.device)
            out = model(data)
            loss = F.nll_loss(out, data.y)
            if torch.isnan(loss):
                print('NO')
            loss.backward()
            step+=1
            optimizer.step()
            loss_train += loss.item()
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()

    # t2=time.time()
    # print(count_parameters(model))
    # print('{:.2f}'.format((t2-t)/10))
    # print()
        acc_train = correct / len(train_loader.dataset)
        acc_val, loss_val = compute_test(val_loader)

        # # client_send(gpuid, 1)
        # # if epoch>5:
        # #     client_send(gpuid, 1)

        outs='Epoch: {:04d}'.format(epoch + 1)+'\tloss_train: {:.6f}'.format(loss_train)+\
              '\tacc_train: {:.6f}'.format(acc_train)+ '\tloss_val: {:.6f}'.format(loss_val)+\
              '\tacc_val: {:.6f}'.format(acc_val)+'\ttime: {:.6f}s'.format(time.time() - t)
        nni.report_intermediate_result(-loss_val) 
        print(outs)
        logging.info(outs)

        val_loss_values.append(loss_val)
        val_acc_values.append(acc_val)
        torch.save(model.state_dict(), res/'{}.pth'.format(epoch))
        if val_loss_values[-1] < min_loss:
            min_loss = val_loss_values[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

        # if val_acc_values[-1] > max_acc:
        #     max_acc = val_acc_values[-1]
        #     best_epoch=epoch
        #     patience_cnt = 0
        # else:
        #     patience_cnt +=1

        if patience_cnt == args.patience:
            break

        files = glob.glob(res.as_posix()+'/*.pth')
        for f in files:
            epoch_nb = int(f.split('/')[-1].split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(f)

    files = glob.glob(res.as_posix()+'/*.pth')
    for f in files:
        epoch_nb = int(f.split('/')[-1].split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(f)
    outs='Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t)
    print(outs)

    return best_epoch


def compute_test(loader):
    model.eval()
    correct = 0.0
    loss_test = 0.0
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss_test += F.nll_loss(out, data.y).item()
    return correct / len(loader.dataset), loss_test


if __name__ == '__main__':
    # gpuid=get_gpuid()
    # client_send(gpuid, 0)
    # print('gpuid:',gpuid)
    # os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(gpuid)
    # print(os.environ["CUDA_VISIBLE_DEVICES"])
    # print(torch.cuda.device_count())
    
    args=get_args()
    tuner_params = nni.get_next_parameter()
    config=args.__dict__
    config.update(tuner_params)
    SetSeed(args.seed)
    # config['device']='cuda:{}'.format(gpuid)

    time_stamp = datetime.datetime.now()+datetime.timedelta(hours=8)
    config['ex_name'] = time_stamp.strftime('%Y.%m.%d-%H:%M:%S')+nni.get_trial_id()


    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    res= Path('/usr/data/gzy/GraphPool/ex_cmp/results')/args.dataset/'{}'.format(config['ex_name'])
    print(res)
    res.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(res/'log')

    sv_param = os.path.join(res, 'model_param.json')
    with open(sv_param, 'w') as file_obj:
        json.dump(args.__dict__, file_obj)

    logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
                    filename=res.as_posix()+'/log.log',#'log/{}_{}_{}.log'.format(args.gcn_type,args.graph_type,args.order_list)
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format='%(asctime)s - %(message)s'
                    #日志格式
                    )


    path = '/usr/data/gzy/GraphPool/ex_cmp/data/'+args.dataset
    dataset = TUDataset(path, name=args.dataset, use_node_attr=True)

    args.num_classes = dataset.num_classes
    args.num_features = dataset.num_features

    logging.info(args.__dict__)
    print(args)

    num_training = int(len(dataset) * 0.8)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - (num_training + num_val)
    training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = Model(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    # Model training
    best_model = train()
    # Restore best model for test set
    model.load_state_dict(torch.load(res/'{}.pth'.format(best_model)))
    test_acc, test_loss = compute_test(test_loader)
    outs='Test set results, loss = {:.6f}, accuracy = {:.6f}'.format(test_loss, test_acc)
    print(outs)
    logging.info(outs)
    nni.report_final_result(test_acc)