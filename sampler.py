import argparse
from datetime import datetime
import json
import os
import torch
import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from tqdm import tqdm

from ribogen.gpu import data_loader, setup, cleanup
from ribogen.models import ProteinDataset

from ribogen.models import DitLMSmall_TE_PE
from ribogen.utility import config_dict, get_protein_features, position_encoding, time_encoding, linear_noise_schedule, pad_proteins, write_readme_conf
from ribogen.utility import load_protein_features, write_readme

from ribogen.models import SingleTokenizer
from ribogen.models import EmbeddingDecoderModel


from ribogen.Diffusion import DiffusionLM
from ribogen.rounding import denoised_fn_round

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def p_sample_loop(model, padded_protein_features, p_mask, num_steps, time_enc, batch_size, w, conf, denoised_fn, clip_denoised, clamp_step, clamp_first):
    device = padded_protein_features.device

    seq_len = conf["max_rna_len"]
    d_model = conf["dit"]["d_model"]
    schedule_name = conf["schedule_name"]

    x_t = torch.randn(seq_len, batch_size, d_model, device=device)

    save_xt_step = 100
    x_t_list = []

    p_mask = p_mask.transpose(0, 1)
    diffusion = DiffusionLM(num_steps, schedule_name=schedule_name)

    for t in reversed(range(num_steps)):

        t_tensor = torch.tensor([t] * batch_size, device=device)

        if not clamp_first:
            if t > clamp_step:
                denoised_fn_cur = None
            else:
                denoised_fn_cur = denoised_fn
        else:
            if t >= clamp_step:
                denoised_fn_cur = denoised_fn
            else:
                denoised_fn_cur = None

        with torch.no_grad():
            output, output_x_0, p_mean, p_mean_coef1, p_mean_coef2, sqrt_one_minus_alpha_bar = diffusion.p_sample_TEPE(
                model,
                padded_protein_features,
                x_t,
                t_tensor,
                time_enc,
                p_mask,
                w,
                denoised_fn_cur,
                clip_denoised,
                conf
            )

        x_t = output

        '''if dist.get_rank() == 0:
            
            
            
            pass'''

        if ((t + 1) % save_xt_step == 0):

            x_t_list += [x_t]

        if (t == 0):
            x_t_list += [x_t]

    return x_t, x_t_list


def format_token_lists(token_lists, end_id=0):
    result = []

    for token_list in token_lists:
        if end_id not in token_list:
            result.append(token_list)
        else:

            zero_index = token_list.index(end_id)

            result.append(token_list[:zero_index])
    return result


def tokens_to_rna(rna_tokenizer, token_lists):
    rnas = []
    for toke_list in token_lists:
        rna = rna_tokenizer.decode(toke_list)
        rnas.append(rna)

    return rnas


def mse_xt_embedding(emb_matrix, xt):
    def compute_mse_mean(emb_matrix, xt):
        """
            计算每个样本中每个位置的特征向量与其最接近的嵌入向量的 MSE 的均值。
            xt: 形状为 (batch, len, dim) 的张量，表示批次的嵌入特征。
            emb_matrix: 形状为 (vocab_size, dim) 的张量，表示嵌入矩阵。

            返回: mse_per_sample, 形状为 (batch,) 的张量，表示每个样本的 MSE 均值。
        """

        xt_expanded = xt.unsqueeze(2)
        emb_matrix_expanded = emb_matrix.unsqueeze(0).unsqueeze(0)

        distances = torch.sum((xt_expanded - emb_matrix_expanded) ** 2, dim=-1)

        closest_indices = torch.argmin(distances, dim=-1)

        closest_embeddings = emb_matrix[closest_indices]

        mse_per_position = torch.mean((xt - closest_embeddings) ** 2, dim=-1)

        mse_per_sample = torch.mean(mse_per_position, dim=-1)

        return mse_per_sample.detach().cpu().numpy()

    mse_per_sample = compute_mse_mean(emb_matrix, xt)
    return mse_per_sample


def mse_xt_embedding_before_eos(emb_matrix, xt):

    def compute_mse_mean(emb_matrix, xt):
        device = xt.device

        xt_expanded = xt.unsqueeze(2)
        emb_matrix_expanded = emb_matrix.unsqueeze(0).unsqueeze(0)

        distances = torch.sum((xt_expanded - emb_matrix_expanded) ** 2, dim=-1)

        closest_indices = torch.argmin(distances, dim=-1)

        closest_embeddings = emb_matrix[closest_indices]

        mse_per_position = torch.mean((xt - closest_embeddings) ** 2, dim=-1)

        mask = (closest_indices != 0).float()

        first_pad_index = torch.argmax((closest_indices == 0).int(), dim=1)

        first_pad_index[first_pad_index == 0] = xt.shape[1]

        batch_indices = torch.arange(xt.shape[0], device=device).unsqueeze(1)
        position_indices = torch.arange(
            xt.shape[1], device=device).unsqueeze(0)
        valid_mask = (position_indices < first_pad_index.unsqueeze(1)).float()

        masked_mse = mse_per_position * valid_mask
        valid_counts = torch.sum(valid_mask, dim=1)
        mse_per_sample = torch.sum(masked_mse, dim=1) / valid_counts

        return mse_per_sample.detach().cpu().numpy()

    return compute_mse_mean(emb_matrix, xt)


def generate_rna(model, f_pids, batch_size, num_steps, w, rank, world_size,
                 out_path, num_per_protein, conf):
    p_df = pd.read_csv(f_pids)
    p_ids = p_df['p_id'].tolist()

    p_ids = [id for id in p_ids for _ in range(num_per_protein)]

    dataset = ProteinDataset(p_ids)
    is_shuffle = False
    dataloader = data_loader(dataset, is_shuffle, batch_size, rank, world_size)

    path_p_feature = conf["dir_protein_features"]
    protein_features_dict = load_protein_features(path_p_feature)

    is_long_tokens = conf["long_token"]
    if is_long_tokens:
        tokenizer_pth = conf["pretrained_tokenizer"]
        rna_tokenizer = AutoTokenizer.from_pretrained(tokenizer_pth)
    else:

        rna_tokenizer = SingleTokenizer()

    now = datetime.now().strftime('%d_%H%M')
    full_path = os.path.join(out_path, now)
    os.makedirs(full_path, exist_ok=True)
    write_readme_conf(conf, f"{full_path}/readme_{now}.txt")

    d_model = conf["dit"]["d_model"]
    time_enc = time_encoding(num_steps, d_model)

    time_enc = time_enc.cuda(rank)

    protein_progress = tqdm(enumerate(dataloader),
                            total=len(dataloader),
                            desc=f"Rank {rank} Proteins",
                            position=rank+1,
                            leave=False)

    for batch, p_ids in protein_progress:
        protein_progress.set_description(f"Rank {rank} Protein batch {batch}")
        protein_features = get_protein_features(p_ids, protein_features_dict)
        padded_protein_features, p_mask = pad_proteins(protein_features)

        padded_protein_features = padded_protein_features.cuda(rank)

        p_mask = p_mask.transpose(0, 1).cuda(rank)

        x0, x_t_list = p_sample_loop(
            model,
            padded_protein_features,
            p_mask,
            num_steps,
            time_enc,
            batch_size,
            w,
            conf,
            denoised_fn=denoised_fn_round,
            clip_denoised=conf["clip_denoised"],
            clamp_step=conf["clamp_step"],
            clamp_first=conf["clamp_first"]
        )

        rnas = []
        mse_xt_list = []
        rna_tokens = []

        for i in range(len(x_t_list)):
            x_t = x_t_list[i]

            x_t = x_t.transpose(1, 0)
            logits = model.module.get_logits(x_t)

            probabilities = F.softmax(logits, dim=-1)
            token_ids = torch.argmax(probabilities, dim=-1)

            '''if world_size>1:
                embedding_matrix = predictor_model.module.rna_embedding.weight  
            else:
                embedding_matrix = predictor_model.rna_embedding.weight.data
            
            mse_xt = mse_xt_embedding_before_eos(embedding_matrix, logits)
            mse_xt_list.append(mse_xt)'''

            token_ids = token_ids.squeeze(0).tolist()
            rna_tokens.append(token_ids)

            token_lists = format_token_lists(token_ids)

            rna_t = tokens_to_rna(rna_tokenizer, token_lists)
            rnas.append(rna_t)

        lens = [len(s) for s in rnas[10]]
        df = pd.DataFrame({
            'p_id': p_ids,
            'RNA': rnas[10],
            'r_len': lens,
            'x_t10': rnas[0],
            'x_t9': rnas[1],
            'x_t8': rnas[2],
            'x_t7': rnas[3],
            'x_t6': rnas[4],
            'x_t5': rnas[5],
            'x_t4': rnas[6],
            'x_t3': rnas[7],
            'x_t2': rnas[8],
            'x_t1': rnas[9],
        })

        df_token = pd.DataFrame({
            'p_id': p_ids,
            'r_token': rna_tokens[10],
            'x_t10': rna_tokens[0],
            'x_t9': rna_tokens[1],
            'x_t8': rna_tokens[2],
            'x_t7': rna_tokens[3],
            'x_t6': rna_tokens[4],
            'x_t5': rna_tokens[5],
            'x_t4': rna_tokens[6],
            'x_t3': rna_tokens[7],
            'x_t2': rna_tokens[8],
            'x_t1': rna_tokens[9],

        })
        tokens_path = os.path.join(full_path, "tokens")

        os.makedirs(tokens_path, exist_ok=True)
        df_token.to_csv(f"{full_path}/tokens/r{rank}_b{batch}_w{w}.csv")

        df.to_csv(f"{full_path}/r{rank}_b{batch}_w{w}.csv")

    print("all done")


def test_generate_rna(f_pids, model_path, w, rank, world_size, out_path, num_steps, num_per_protein, conf):

    gpu_available = conf["env"]["all_gpus"]
    nccl_port = conf["env"]["nccl_port"]
    setup(rank, world_size, gpu_available, port=nccl_port)

    batch_size = conf["batch_size"]
    d_model = conf["dit"]["d_model"]
    output_dim = conf["vocab_size"]
    n_heads = conf["dit"]["n_heads"]
    n_layers = conf["dit"]["n_layers"]
    d_input = conf["dit"]["d_input"]
    hidden_dim = conf["dit"]["d_model"]
    vocab_size = conf["vocab_size"]
    hidden_t_dim = conf["dit"]["hidden_t_dim"]
    max_position_emb_len = conf["dit"]["max_position_emb_len"]

    def load_model(pth, model, optimizer=None):
        checkpoint = torch.load(pth)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model, optimizer

    model = DitLMSmall_TE_PE(d_input, hidden_dim, vocab_size, n_heads,
                             n_layers, num_steps, hidden_t_dim, max_position_emb_len)
    model, _ = load_model(model_path, model, optimizer=None)
    model = model.cuda(rank)
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model.eval()

    '''
    mlp_hidden_dim = conf["mlp"]["hidden_dim"] 
    decoder_model = EmbeddingDecoderModel(output_dim, d_model, d_model, mlp_hidden_dim)
    checkpoint = torch.load(model_decoder)
    decoder_model.load_state_dict(checkpoint['model_state_dict'])
    decoder_model.eval()
    decoder_model = decoder_model.cuda(rank)
    decoder_mlp = decoder_model.decoder_mlp
    decoder_mlp.eval()'''

    generate_rna(model, f_pids, batch_size, num_steps, w,
                 rank, world_size, out_path, num_per_protein, conf)

    cleanup(world_size)


def run_worker(rank, world_size, f_conf):
    conf = config_dict(f_conf)

    f_pids = conf["file_p"]
    out_path = conf["dir_out"]
    model_pth = conf["dit"]["dit_pth"]

    num_steps = conf["num_steps"]
    num_per_protein = conf["num_per_protein"]
    w = conf["cfg_w"]

    test_generate_rna(f_pids, model_pth, w, rank, world_size,
                      out_path, num_steps, num_per_protein, conf)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="rna generator")

    debug = False
    if not debug:
        parser.add_argument("--conf", type=str,
                            help="generator config file", required=False)

        group = parser.add_mutually_exclusive_group(required=False)

        group.add_argument(
            '-l', '--long', action='store_true', help="long tokens")
        group.add_argument(
            '-s', '--single', action='store_true', help="single token")

        args = parser.parse_args()
        print(f"long tokens: {args.long}")
        f_gener_conf = args.conf
    else:

        f_gener_conf = "results_small_v2.0_TEPE/generator_embeding_long_d128.json"

    conf = config_dict(f_gener_conf)
    print(conf["file_p"])
    print(conf["dit"]["dit_pth"])
    print(conf["env"]["all_gpus"])
    input("确认？")
    world_size = conf["env"]["n_gpus"]

    if world_size > 1:
        mp.spawn(run_worker, args=(world_size, f_gener_conf), nprocs=world_size)
    else:
        run_worker(conf["env"]["all_gpus"][0], 1, f_gener_conf)
