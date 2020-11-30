from sys import version
import numpy as np
import numpy
import os
from multiprocessing import Process, Manager
import signal
import time

  
cmd=[
    # ###########################  PROTEINS
    # 'CUDA_VISIBLE_DEVICES={}'+ ' python main.py   --dataset PROTEINS\
    #                                               --K 3\
    #                                               --num_layers 5\
    #                                               --pooling_ratio 0.271521829\
    #                                               --edge_ratio 0.439671177\
    #                                               --conv ChebConv\
    #                                               --decoupled False\
    #                                               --ex_name Pro_CC_PP',

    # 'CUDA_VISIBLE_DEVICES={}'+ ' python main.py   --dataset PROTEINS\
    #                                               --K 3\
    #                                               --num_layers 5\
    #                                               --pooling_ratio 0.271521829\
    #                                               --edge_ratio 0.439671177\
    #                                               --conv LiCheb\
    #                                               --decoupled False\
    #                                               --ex_name Pro_DLC_PP',

    # 'CUDA_VISIBLE_DEVICES={}'+ ' python main.py   --dataset PROTEINS\
    #                                               --K 3\
    #                                               --num_layers 5\
    #                                               --pooling_ratio 0.271521829\
    #                                               --edge_ratio 0.439671177\
    #                                               --conv LiCheb\
    #                                               --decoupled True\
    #                                               --ex_name Pro_DLC_DP',
    

    # ###########################  DD
    # 'CUDA_VISIBLE_DEVICES={}'+ ' python main.py   --dataset DD\
    #                                               --K 3\
    #                                               --num_layers 5\
    #                                               --pooling_ratio 0.543869989\
    #                                               --edge_ratio 0.330416052\
    #                                               --conv ChebConv\
    #                                               --decoupled False\
    #                                               --ex_name DD_CC_PP',

    # 'CUDA_VISIBLE_DEVICES={}'+ ' python main.py   --dataset DD\
    #                                               --K 3\
    #                                               --num_layers 5\
    #                                               --pooling_ratio 0.543869989\
    #                                               --edge_ratio 0.330416052\
    #                                               --conv LiCheb\
    #                                               --decoupled False\
    #                                               --ex_name DD_DLC_PP',

    # 'CUDA_VISIBLE_DEVICES={}'+ ' python main.py   --dataset DD\
    #                                               --K 3\
    #                                               --num_layers 5\
    #                                               --pooling_ratio 0.543869989\
    #                                               --edge_ratio 0.330416052\
    #                                               --conv LiCheb\
    #                                               --decoupled True\
    #                                               --ex_name DD_DLC_DP',

    
    # ###########################  NCI1
    # 'CUDA_VISIBLE_DEVICES={}'+ ' python main.py   --dataset NCI1\
    #                                               --K 3\
    #                                               --num_layers 5\
    #                                               --pooling_ratio 0.632026586\
    #                                               --edge_ratio 0.704032132\
    #                                               --conv ChebConv\
    #                                               --decoupled False\
    #                                               --ex_name NCI1_CC_PP',

    # 'CUDA_VISIBLE_DEVICES={}'+ ' python main.py   --dataset NCI1\
    #                                               --K 3\
    #                                               --num_layers 5\
    #                                               --pooling_ratio 0.632026586\
    #                                               --edge_ratio 0.704032132\
    #                                               --conv LiCheb\
    #                                               --decoupled False\
    #                                               --ex_name NCI1_DLC_PP',

    # 'CUDA_VISIBLE_DEVICES={}'+ ' python main.py   --dataset NCI1\
    #                                               --K 3\
    #                                               --num_layers 5\
    #                                               --pooling_ratio 0.632026586\
    #                                               --edge_ratio 0.704032132\
    #                                               --conv LiCheb\
    #                                               --decoupled True\
    #                                               --ex_name NCI1_DLC_DP',


    # ########################### NCI109
    # 'CUDA_VISIBLE_DEVICES={}'+ ' python main.py   --dataset NCI109\
    #                                               --K 3\
    #                                               --num_layers 4\
    #                                               --pooling_ratio 0.841586839\
    #                                               --edge_ratio 0.678934205\
    #                                               --conv ChebConv\
    #                                               --decoupled False\
    #                                               --ex_name NCI109_CC_PP',

    # 'CUDA_VISIBLE_DEVICES={}'+ ' python main.py   --dataset NCI109\
    #                                               --K 3\
    #                                               --num_layers 4\
    #                                               --pooling_ratio 0.841586839\
    #                                               --edge_ratio 0.678934205\
    #                                               --conv LiCheb\
    #                                               --decoupled False\
    #                                               --ex_name NCI109_DLC_PP',

    # 'CUDA_VISIBLE_DEVICES={}'+ ' python main.py   --dataset NCI109\
    #                                               --K 3\
    #                                               --num_layers 4\
    #                                               --pooling_ratio 0.841586839\
    #                                               --edge_ratio 0.678934205\
    #                                               --conv LiCheb\
    #                                               --decoupled True\
    #                                               --ex_name NCI109_DLC_DP',
    

    # ########################### Mutagenicity
    # 'CUDA_VISIBLE_DEVICES={}'+ ' python main.py   --dataset Mutagenicity\
    #                                               --K 2\
    #                                               --num_layers 2\
    #                                               --pooling_ratio 0.426631657\
    #                                               --edge_ratio 0.427344799\
    #                                               --conv ChebConv\
    #                                               --decoupled False\
    #                                               --ex_name Mut_CC_PP',

    # 'CUDA_VISIBLE_DEVICES={}'+ ' python main.py   --dataset Mutagenicity\
    #                                               --K 2\
    #                                               --num_layers 2\
    #                                               --pooling_ratio 0.426631657\
    #                                               --edge_ratio 0.427344799\
    #                                               --conv LiCheb\
    #                                               --decoupled False\
    #                                               --ex_name Mut_DLC_PP',

    # 'CUDA_VISIBLE_DEVICES={}'+ ' python main.py   --dataset Mutagenicity\
    #                                               --K 2\
    #                                               --num_layers 2\
    #                                               --pooling_ratio 0.426631657\
    #                                               --edge_ratio 0.427344799\
    #                                               --conv LiCheb\
    #                                               --decoupled True\
    #                                               --ex_name Mut_DLC_DP',
    

    # ########################### ENZYMES
    # 'CUDA_VISIBLE_DEVICES={}'+ ' python main.py   --dataset ENZYMES\
    #                                               --K 1\
    #                                               --num_layers 4\
    #                                               --pooling_ratio 0.799915408\
    #                                               --edge_ratio 0.247069653\
    #                                               --conv ChebConv\
    #                                               --decoupled False\
    #                                               --ex_name ENZ_CC_PP',

    # 'CUDA_VISIBLE_DEVICES={}'+ ' python main.py   --dataset ENZYMES\
    #                                               --K 1\
    #                                               --num_layers 4\
    #                                               --pooling_ratio 0.799915408\
    #                                               --edge_ratio 0.247069653\
    #                                               --conv LiCheb\
    #                                               --decoupled False\
    #                                               --ex_name ENZ_DLC_PP',

    # 'CUDA_VISIBLE_DEVICES={}'+ ' python main.py   --dataset ENZYMES\
    #                                               --K 1\
    #                                               --num_layers 4\
    #                                               --pooling_ratio 0.799915408\
    #                                               --edge_ratio 0.247069653\
    #                                               --conv LiCheb\
    #                                               --decoupled True\
    #                                               --ex_name ENZ_DLC_DP',

    
    ###########################  FRANKENSTEIN
    # 'CUDA_VISIBLE_DEVICES={}'+ ' python main.py   --dataset FRANKENSTEIN\
    #                                               --K 4\
    #                                               --num_layers 3\
    #                                               --pooling_ratio 0.881023822\
    #                                               --edge_ratio 0.365035598\
    #                                               --conv ChebConv\
    #                                               --decoupled False\
    #                                               --ex_name Far_CC_PP',

    # 'CUDA_VISIBLE_DEVICES={}'+ ' python main.py   --dataset FRANKENSTEIN\
    #                                               --K 4\
    #                                               --num_layers 3\
    #                                               --pooling_ratio 0.881023822\
    #                                               --edge_ratio 0.365035598\
    #                                               --conv LiCheb\
    #                                               --decoupled False\
    #                                               --ex_name Far_DLC_PP',

    # 'CUDA_VISIBLE_DEVICES={}'+ ' python main.py   --dataset FRANKENSTEIN\
    #                                               --K 4\
    #                                               --num_layers 3\
    #                                               --pooling_ratio 0.881023822\
    #                                               --edge_ratio 0.365035598\
    #                                               --conv LiCheb\
    #                                               --decoupled True\
    #                                               --ex_name Far_DLC_DP',
]



def run(command, gpuid, gpustate):
    os.system(command.format(','.join(list(map(str,gpuid)))))
    for i in range(len(gpuid)):
        gpustate[gpuid[i]] = True


def term(sig_num, addtion):
    print('terminate process {}'.format(os.getpid()))
    try:
        print('the processes is {}'.format(processes))
        for p in processes:
            print('process {} terminate'.format(p.pid))
            p.terminate()
            # os.kill(p.pid, signal.SIGKILL)
    except Exception as e:
        print(str(e))



if __name__ == '__main__':
    signal.signal(signal.SIGTERM, term)  # 注册信号量，使得在终端杀死主进程时，子进程也被杀死

    gpus = [(0,),(1,),(2,),(3,),(0,),(1,)]#(0,2,3,4),
    gpustate = Manager().dict({i: True for i in range(0,8)})
    processes = []
    idx = 0
    while idx < len(cmd):
        # 查询是否有可用gpu
        for gpuid in gpus:
            if gpustate[gpuid[0]]:
                p = Process(target=run, args=(
                    cmd[idx], gpuid, gpustate), name=str(gpuid))
                p.start()
                print('run {} with gpu {}'.format(cmd[idx], gpuid))
                processes.append(p)
                idx += 1

                if idx == len(cmd):
                    break

        time.sleep(600)

    for p in processes:
        p.join()

    while(1):
        pass
