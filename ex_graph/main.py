import argparse
import glob
import os
import time

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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=777, help='random seed')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--nhid', type=int, default=128, help='hidden size')
    parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
    parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
    parser.add_argument('--structure_learning', type=bool, default=False, help='whether perform structure learning')
    parser.add_argument('--pooling_ratio', type=float, default=0.87, help='pooling ratio')
    parser.add_argument('--edge_ratio', type=float, default=0.4, help='pooling ratio')
    parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
    parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
    parser.add_argument('--dataset', type=str, default='ENZYMES', help='DD/PROTEINS/NCI1/NCI109/Mutagenicity/ENZYMES')
    parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
    parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs')
    parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
    parser.add_argument('--conv',default='LiCheb',help='GCN/ChebConv/LiCheb/LeCheb/Mix')  # ChebConv
    parser.add_argument('--pool',default='MAtt',help='HGPSL/MAtt')
    parser.add_argument('--K',default=2,type=int)
    args = parser.parse_args()
    return args




def train():
    min_loss = 1e10
    patience_cnt = 0
    val_loss_values = []
    best_epoch = 0

    t = time.time()
    model.train()
    step=0
    for epoch in range(args.epochs):
        loss_train = 0.0
        correct = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(args.device)
            out,x_score1,x_score2 = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            # writer.add_scalars('drop ratio',{'layer1':model.conv1.SA.threshold.data.detach().cpu().numpy(),'layer2':model.conv2.SA.threshold.data.detach().cpu().numpy()},global_step=step)
            writer.add_scalar('loss',loss.item(),global_step=step)
            writer.add_histogram('norm1',x_score1,global_step=step)
            writer.add_histogram('norm2',x_score2,global_step=step)
            step+=1
            optimizer.step()
            loss_train += loss.item()
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
        acc_train = correct / len(train_loader.dataset)
        acc_val, loss_val = compute_test(val_loader)

        outs='Epoch: {:04d}'.format(epoch + 1)+'\tloss_train: {:.6f}'.format(loss_train)+\
              '\tacc_train: {:.6f}'.format(acc_train)+ '\tloss_val: {:.6f}'.format(loss_val)+\
              '\tacc_val: {:.6f}'.format(acc_val)+'\ttime: {:.6f}s'.format(time.time() - t)
        nni.report_intermediate_result(-loss_val) 
        print(outs)
        logging.info(outs)

        val_loss_values.append(loss_val)
        torch.save(model.state_dict(), res/'{}.pth'.format(epoch))
        if val_loss_values[-1] < min_loss:
            min_loss = val_loss_values[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

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
        out,_,_ = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss_test += F.nll_loss(out, data.y).item()
    return correct / len(loader.dataset), loss_test


if __name__ == '__main__':
    args=get_args()
    tuner_params = nni.get_next_parameter()
    config=args.__dict__
    config.update(tuner_params)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    time_stamp = datetime.datetime.now()+datetime.timedelta(hours=8)
    config['ex_name'] = time_stamp.strftime('%Y.%m.%d-%H:%M:%S')+nni.get_trial_id()


    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    res= Path('/usr/commondata/weather/code/GraghPool/ex_graph/results')/args.dataset/'{}'.format(config['ex_name'])
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


    dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset, use_node_attr=True)

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