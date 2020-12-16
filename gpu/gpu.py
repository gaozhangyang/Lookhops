import argparse
import GPUtil
import torch

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# parser = argparse.ArgumentParser(description='')
# parser.add_argument("--gpu", type=int, default=3)
# args = parser.parse_args()
# param = args.__dict__

if __name__ == '__main__':
    while True:
        for gpuid in [0,1]:
            device='cuda:{}'.format(gpuid)
            f = open('GPU_{}.txt'.format(gpuid),'r')
            lines = f.readlines()
            if len(lines) > 0:
                data = lines[-1][0:1]
                print(gpuid, data)

                if data == '0':
                    del a, b, c
                    a = 1
                    b = 1
                    c = 1
                    torch.cuda.empty_cache()

                if data == '1':
                    gpu = GPUtil.getGPUs()[gpuid]
                    try:
                      if gpu.memoryUsed < 31000:
                          memory = int(((34000 - gpu.memoryUsed) * 1024 * 1024 / 8 / 2) ** 0.5)
                          a = torch.ones((memory, memory)).to(device)
                          b = torch.ones((memory, memory)).to(device)
                          c = a * b
                    except:
                      pass
            else:
                print('None')