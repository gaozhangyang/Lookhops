import argparse
import GPUtil
import torch
import time

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# parser = argparse.ArgumentParser(description='')
# parser.add_argument("--gpu", type=int, default=3)
# args = parser.parse_args()
# param = args.__dict__

if __name__ == '__main__':
    # 1281 1233   3106
    # memory = 2000
    # device='cuda:{}'.format(4)
    # a = torch.ones((memory, memory)).to(device)
    # a.requires_grad=True
    # b = torch.ones((memory, memory)).to(device)
    # b.requires_grad=True
    # c = torch.mm(a,b)
    # c.sum().backward()
    # print()

    # while True:
    #     for gpuid in [4,5]:
    #         device='cuda:{}'.format(gpuid)
    #         f = open('GPU_{}.txt'.format(gpuid),'r')
    #         vars=[]
    #         lines = f.readlines()
    #         if len(lines) > 0:
    #             data = lines[-1][0:1]
    #             print(gpuid, data)

    #             if data == '0':
    #                 del vars
    #                 torch.cuda.empty_cache()

    #             if data == '1':
    #                 gpu = GPUtil.getGPUs()[gpuid]
    #                 try:
    #                 #   if gpu.memoryUsed < 33000:
    #                 #     #   memory = int(((36000 - gpu.memoryUsed) * 1024 * 1024 / 8 / 2) ** 0.5)
    #                 #       memory = 10000
    #                 #       vars.append( torch.ones((memory, memory)).to(device) )
    #                 #       vars.append( torch.ones((memory, memory)).to(device) ) 
    #                 #       vars[-1]*vars[-2]
    #                     memory = 10000
    #                     vars.append( torch.ones((memory, memory)).to(device) )
    #                     vars.append( torch.ones((memory, memory)).to(device) ) 
    #                     vars[-1]*vars[-2]
    #                     print('GPU{}:{}'.format(gpuid,len(vars)))
    #                 except:
    #                     print('GPU{}:error'.format(gpuid))
    #         else:
    #             print('None')


    vars=[]
    while True:
        time.sleep(5)
        for gpuid in [4]:
            device='cuda:{}'.format(gpuid)
            try:
                memory = 10000
                a=torch.ones((memory, memory)).to(device)
                b=torch.ones((memory, memory)).to(device)
                vars.append( a ) 
                vars.append( b ) 
                c=a*b
                print('GPU{}:{}'.format(gpuid,len(vars)))
            except:
                gpu = GPUtil.getGPUs()[gpuid]
                if gpu.memoryUsed < 32500:
                    print('GPU{}:out of memory, used {}'.format(gpuid,gpu.memoryUsed))