import os
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

def setup(rank, world_size, gpus_available, port='60000'):
    if world_size > 1:
        
        os.environ['MASTER_ADDR'] = 'localhost'  
        os.environ['MASTER_PORT'] = port 
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus_available))

        
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)  


def cleanup(world_size):
    if world_size > 1:
        
        dist.destroy_process_group()


def data_loader(dataset, is_shuffle, batch_size, rank, world_size):
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=is_shuffle)
        
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
    else:
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_shuffle, drop_last=True)

    return dataloader


def losses_reduce(epoch_loss, world_size, num_batches): 

    dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM) 
    avg_epoch_loss = epoch_loss / (world_size * num_batches) 
    
    return avg_epoch_loss