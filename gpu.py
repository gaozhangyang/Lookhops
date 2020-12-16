import pynvml
import torch
pynvml.nvmlInit()

def get_memory(gpuid):
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpuid)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    memory_GB = meminfo.free/1024**2
    return memory_GB

def occupy_gpu(gpuid):
  dummy=torch.zeros(10000,1,device='cuda:{}'.format(gpuid),dtype=torch.float64)
  print('OK')

if __name__ =='__main__':
  occupy_gpu(4)