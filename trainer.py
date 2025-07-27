import argparse
import math
import os
import random
from signal import signal
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from ribogen.gpu import data_loader, losses_reduce, setup, cleanup
from ribogen.models import DitLM, DitLMSmall_TE_PE, ProteinRNADataset
from ribogen.utility import config_dict, display_epoch_loss, read_RNA_tokens, get_protein_features, position_encoding, load_model, save_checkpoint
from ribogen.utility import time_encoding, linear_noise_schedule, load_protein_features, pad_proteins, pad_rnas
from ribogen.utility import get_named_beta_schedule, mean_flat

from ribogen.step_sample import create_named_schedule_sampler
from ribogen.Diffusion import DiffusionLM


def train(rank, world_size, model, dataloader, protein_features_dict, rna_dict, optimizer, max_timesteps,
          conf, training=True):

    model.train() if training else model.eval()
    epoch_loss = torch.tensor(0.0).to(rank)

    accumulation_loss_backward_steps = conf["train"]["accumulation_loss_backward_steps"]
    max_timesteps = conf["dit"]["max_timesteps"]
    d_model = conf["dit"]["d_model"]
    time_enc = time_encoding(max_timesteps, d_model)
    time_enc = time_enc.cuda(rank)

    ignore_pad_id = conf["dit"]["ignore_pad_id"]

    d_model = conf["dit"]["d_model"]

    r_mask_null = conf["dit"]["rmask_null"]
    sampler_name = conf["train"]["t_sampler"]

    schedule_name = conf["dit"]["schedule_name"]
    diffusionLM = DiffusionLM(max_timesteps, schedule_name=schedule_name)

    t_sampler = create_named_schedule_sampler(sampler_name, diffusionLM)

    for i, (p_ids, _, r_ids) in enumerate(dataloader):
        batch_size = len(p_ids)

        protein_features = get_protein_features(p_ids, protein_features_dict)
        padded_protein_features, padding_mask = pad_proteins(protein_features)
        padded_protein_features = padded_protein_features.cuda(rank)
        p_mask = padding_mask.transpose(0, 1).cuda(rank)

        memory = padded_protein_features

        t, weights = t_sampler.sample(batch_size, device=p_mask.device)

        t_expanded = t.unsqueeze(1).cuda(rank)

        rna_sequences = np.array(
            list(map(lambda key: rna_dict.get(key, None), r_ids)))
        rna_sequences = np.asarray(rna_sequences, dtype=np.int32)
        rna_sequences = torch.tensor(
            rna_sequences, dtype=torch.int32).cuda(rank)
        rna_tokens = rna_sequences.long()
        pad_id = conf["pad_id"]
        r_mask = pad_rnas(rna_tokens, pad_id)

        rna_embeds = model.module.rna_embedding(rna_tokens)

        x_t, x_0, _ = diffusionLM.q_sample(rna_embeds, t, r_mask, pad_id)

        CFG = conf["train"]["CFG"]
        if CFG:
            conditional = random.choice([True, False])
        else:
            conditional = True

        p_mask = p_mask.transpose(0, 1)

        x_t = x_t.transpose(0, 1)

        tgt_mask = r_mask if not r_mask_null else None

        output_x0 = model(memory, x_t, p_mask, tgt_mask, t_expanded)

        output_x0 = output_x0.permute(1, 0, 2)

        losses = diffusionLM.loss_function(
            model, output_x0, x_0, rna_embeds, rna_tokens, t, r_mask)

        '''
        if ignore_pad_id:
            mask_expanded = r_mask.unsqueeze(-1).expand_as(loss)  
            masked_loss = loss * ~mask_expanded  
            num_non_padding = (~mask_expanded).sum()  
            loss = masked_loss.sum() / num_non_padding.float()
        else:
            loss = loss.mean()
        '''
        if sampler_name == "lossaware":
            t_sampler.update_with_local_losses(t, losses["loss"].detach())

        loss = (losses["loss"] * weights).mean()

        epoch_loss += loss.detach()

        if training:

            loss.backward()

            if (i + 1) % accumulation_loss_backward_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        stage = "Training" if training else "Test"
        if world_size == 1 or dist.get_rank() == 0:
            print(f"{stage}\t batch {i}\t{loss.item()}")

    if world_size > 1:
        avg_epoch_loss = losses_reduce(epoch_loss, world_size, len(dataloader))
    else:
        avg_epoch_loss = epoch_loss / len(dataloader)

    return avg_epoch_loss.item()


def get_dataloader(f_pairs, f_test_pairs, batch_size, rank, world_size, is_shuffle):
    df_pairs = pd.read_csv(f_pairs)

    df_pairs = df_pairs.sort_values(by='p_len', ascending=True)

    p_ids = df_pairs['p_id'].tolist()
    r_ids = df_pairs['r_id'].tolist()
    p_lens = df_pairs['p_len'].tolist()
    dataset = ProteinRNADataset(p_ids, p_lens, r_ids)

    df_test_pairs = pd.read_csv(f_test_pairs)
    df_test_pairs = df_test_pairs.sort_values(by='p_len', ascending=True)
    test_p_ids = df_test_pairs['p_id'].tolist()
    test_r_ids = df_test_pairs['r_id'].tolist()
    test_p_lens = df_test_pairs['p_len'].tolist()
    test_dataset = ProteinRNADataset(test_p_ids, test_p_lens, test_r_ids)

    dataloader = data_loader(dataset, is_shuffle, batch_size, rank, world_size)
    test_dataloader = data_loader(
        test_dataset, is_shuffle, batch_size, rank, world_size)

    return dataloader, test_dataloader


def test_train(rank, world_size, conf):
    gpus_available = conf["env"]["all_gpus"]
    nccl_port = conf["env"]["nccl_port"]
    setup(rank, world_size, gpus_available, port=nccl_port)

    num_epochs = conf["train"]["num_epochs"]
    batch_size = conf["train"]["batch_size"]
    learning_rate = conf["train"]["learning_rate"]

    hidden_dim = conf["dit"]["d_model"]
    n_heads = conf["dit"]["n_heads"]
    n_layers = conf["dit"]["n_layers"]
    vocab_size = conf["vocab_size"]
    max_timesteps = conf["dit"]["max_timesteps"]
    f_pairs = conf["train"]["file_p_r"]
    f_test_pairs = conf["test"]["file_p_r"]
    is_shuffle = conf["train"]["shuffle"]

    dataloader, test_dataloader = get_dataloader(
        f_pairs, f_test_pairs, batch_size, rank, world_size, is_shuffle)

    file_rnas = conf["train"]["file_rna_tokens"]
    pad_id = conf["pad_id"]
    rids_unique, rnas_unique, _ = read_RNA_tokens(file_rnas, pad_id)
    rna_dict = {rid: value for rid, value in zip(rids_unique, rnas_unique)}

    model_scale = conf["dit"]["scale"]
    d_input = conf["dit"]["d_input"]

    hidden_t_dim = conf["dit"]["hidden_t_dim"]
    max_position_emb_len = conf["dit"]["max_position_emb_len"]

    if model_scale == "small_PE_TE":
        model = DitLMSmall_TE_PE(d_input, hidden_dim, vocab_size, n_heads,
                                 n_layers, max_timesteps, hidden_t_dim, max_position_emb_len)
        print(f"model_scale: {model_scale}")
    else:
        model = DitLM(hidden_dim, vocab_size, n_heads, n_layers, max_timesteps)

    model_pth = conf["train"]["active_checkpoint"]

    if model_pth is not None:
        model, _ = load_model(model_pth, model, optimizer=None)

    model = model.cuda(rank)
    if world_size > 1:

        model = DDP(model, device_ids=[rank])

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    '''noise_schedule = linear_noise_schedule(max_timesteps, beta_min=0.0001, beta_max=0.02)
    noise_schedule = noise_schedule.cuda(rank)'''

    all_train_loss = []
    all_test_loss = []
    dir_proteins = conf["train"]["dir_protein_features"]
    protein_features_dict = load_protein_features(dir_proteins)

    start_time = time.time()
    if world_size == 1 or dist.get_rank() == 0:
        print(f"number of epoches: {num_epochs}, start time: {start_time}")

    for epoch in range(num_epochs):
        loss = train(rank, world_size, model, dataloader, protein_features_dict, rna_dict,
                     optimizer, max_timesteps, conf)

        display_epoch_loss(epoch, loss, world_size, training=True)
        all_train_loss.append(loss)

        if world_size > 1:
            dist.barrier()
            torch.cuda.empty_cache()

        '''
        
        with torch.no_grad():
            test_loss = train(rank, world_size, model, test_dataloader, protein_features_dict, rna_dict, 
                              optimizer, max_timesteps, conf, training=False)

        
        display_epoch_loss(epoch, test_loss, world_size, training=False)
        all_test_loss.append(test_loss)

        if world_size > 1: 
            dist.barrier()
            torch.cuda.empty_cache()
        '''
    time_cost = time.time() - start_time
    dir_out = conf["train"]["dir_out"]

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
        print(f"dir {dir_out} don't exist, created dir")

    all_test_loss = None
    save_checkpoint(model, time_cost, dir_out, world_size,
                    all_train_loss, all_test_loss)
    cleanup(world_size)


def main():

    parser = argparse.ArgumentParser(description="diffusion transformer")

    debug = True
    print(f"debug : {debug}")
    if not debug:
        parser.add_argument("-c", "--conf", type=str,
                            help="config file", required=True)

        group = parser.add_mutually_exclusive_group(required=True)

        group.add_argument(
            '-l', '--long', action='store_true', help="long tokens")
        group.add_argument(
            '-s', '--single', action='store_true', help="single token")

        args = parser.parse_args()
        f_conf = args.conf
    else:

        f_conf = "mymodel_small_TE_PE_v2.0_long/predictor_embeding_long_d128.json"

    conf = config_dict(f_conf)
    world_size = conf["env"]["n_gpus"]
    print(f"config:{f_conf}")
    print(f"epoch: {conf['train']['num_epochs']}")
    print(f"checkpoint: {conf['train']['active_checkpoint']}")
    print("training...")
    input("press enter to continue")

    if world_size > 1:
        mp.spawn(test_train, args=(world_size, conf), nprocs=world_size)
    else:
        test_train(0, 1, conf)


if __name__ == "__main__":
    main()
