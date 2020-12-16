import os
import signal
from multiprocessing import Process, Manager

cmd=[
    'CUDA_VISIBLE_DEVICES={} '+'python gpu.py --gpu 0',
    'CUDA_VISIBLE_DEVICES={} '+'python gpu.py --gpu 1',
    'CUDA_VISIBLE_DEVICES={} '+'python gpu.py --gpu 2',
    'CUDA_VISIBLE_DEVICES={} '+'python gpu.py --gpu 3',
    'CUDA_VISIBLE_DEVICES={} '+'python gpu.py --gpu 4',
    'CUDA_VISIBLE_DEVICES={} '+'python gpu.py --gpu 5',
    'CUDA_VISIBLE_DEVICES={} '+'python gpu.py --gpu 6',
    'CUDA_VISIBLE_DEVICES={} '+'python gpu.py --gpu 7',
]

# for i in range(200):
#     cmd.append('CUDA_VISIBLE_DEVICES={} '+'python train.py --seed 5')

def run(command,gpuid,gpustate):
    os.system(command.format(gpuid))
    gpustate[str(gpuid)] = True

def term(sig_num, addtion):
    print('terminate process {}'.format(os.getpid()))
    try:
        print('the processes is {}'.format(processes) )
        for p in processes:
            print('process {} terminate'.format(p.pid))
            p.terminate()
    except Exception as e:
        print(str(e))

if __name__  == '__main__':
    signal.signal(signal.SIGTERM, term)

    gpustate = Manager().dict({str(i):True for i in range(0, 8)})
    processes = []
    idx = 0

    while idx < len(cmd):
        for gpuid in range(0, 8):
            if gpustate[str(gpuid)] == True:
                print(idx)
                gpustate[str(gpuid)] = False
                p = Process(target = run, args = (cmd[idx], gpuid, gpustate), name = str(gpuid))
                p.start()

                print(gpustate)
                processes.append(p)
                idx += 1
                
                break

    for p in processes:
        p.join()
