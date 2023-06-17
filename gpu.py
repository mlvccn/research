import subprocess as sp
import os

def get_gpu_memory(index=0):
    command = "nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    current_gpu_info = memory_free_info[index].split(', ')
    print("GPU index: {}; Name: {}; Total memory: {}; Used memory: {}".format(current_gpu_info[0], \
        current_gpu_info[1], current_gpu_info[2], current_gpu_info[3]))

get_gpu_memory()