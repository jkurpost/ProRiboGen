from datetime import datetime
import os
import torch
import numpy as np
import math
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import json
from torch.nn.utils.rnn import pad_sequence


def config_dict(f_conf):
    with open(f_conf, 'r') as f:
        config = json.load(f)

    return config


def load_protein_features(folder_path):

    protein_features_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.npz'):
            file_path = os.path.join(folder_path, filename)

            npz_data = np.load(file_path)
            for protein_id, features in npz_data.items():
                protein_features_dict[protein_id] = features

    return protein_features_dict


def get_protein_features(p_ids, protein_features_dict):

    features = []
    for p_id in p_ids:
        feature = protein_features_dict[p_id]
        features.append(feature)

    return features


def get_protein_features_with_domain(p_ids, rrm_domains, protein_features_dict):

    batch_features = []

    for p_id, rrm_domain in zip(p_ids, rrm_domains):

        if p_id not in protein_features_dict:
            raise KeyError(
                f"Error: p_id {p_id} not found in protein_features_dict.")

        feature = protein_features_dict[p_id]

        try:
            start, end = map(int, rrm_domain.strip("[]").split(":"))
            if not (0 <= start < end <= feature.shape[0]):
                raise ValueError(
                    f"Error: RRM domain {rrm_domain} is out of range for {p_id}.")

            feature[start:end, :] = 0
        except Exception as e:
            raise ValueError(
                f"Error: Invalid RRM domain format for {p_id}: {rrm_domain}. Details: {e}")

        feature_tensor = torch.from_numpy(feature)
        batch_features.append(feature_tensor)

    return torch.stack(batch_features)


def get_protein_features_with_DomainMask(p_ids, rrm_domains, protein_features_dict):

    batch_features = []
    batch_masks = []

    for p_id, rrm_domain in zip(p_ids, rrm_domains):
        if p_id not in protein_features_dict:
            raise KeyError(
                f"Error: p_id {p_id} not found in protein_features_dict.")

        feature = protein_features_dict[p_id]
        L = feature.shape[0]
        domain_mask = torch.zeros(L, dtype=torch.bool)

        try:
            start, end = map(int, rrm_domain.strip("[]").split(":"))
            if not (0 <= start < end <= L):
                raise ValueError(
                    f"Error: RRM domain {rrm_domain} is out of range for {p_id}.")

            domain_mask[start:end] = True
        except Exception as e:
            raise ValueError(
                f"Error: Invalid RRM domain format for {p_id}: {rrm_domain}. Details: {e}")

        batch_features.append(torch.from_numpy(feature))
        batch_masks.append(domain_mask)

    return torch.stack(batch_features), torch.stack(batch_masks)


def pad_proteins(protein_features):

    protein_features = [torch.tensor(seq) for seq in protein_features]

    padded_protein_features = pad_sequence(protein_features, batch_first=True)

    padding_mask = torch.zeros(
        padded_protein_features.shape[:2], dtype=torch.bool)
    for i, seq in enumerate(protein_features):
        padding_mask[i, len(seq):] = True

    return padded_protein_features, padding_mask


def pad_rnas(rna_sequences, pad_token_id):

    padding_mask = rna_sequences == pad_token_id

    return padding_mask


def read_RNA_tokens(file_path, pad_id):

    data = []

    with open(file_path, 'r') as f:
        for line in f:

            values = line.strip().split('\t')

            ids = values[0]
            integers = list(map(int, values[1:]))
            data.append([ids] + integers)

    max_len = max(len(row) for row in data)

    np_array = np.array([
        [row[0]] + row[1:] + [pad_id] * (max_len - len(row))
        for row in data
    ], dtype=object)

    return np_array[:, 0], np_array[:, 1:], max_len - 1


def position_encoding(seq_len, d_model):

    position = torch.arange(seq_len).float().unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float()
                         * -(math.log(10000.0) / d_model))
    pos_enc = torch.zeros(seq_len, d_model)
    pos_enc[:, 0::2] = torch.sin(position * div_term)
    pos_enc[:, 1::2] = torch.cos(position * div_term)
    return pos_enc


def time_encoding(max_timesteps, d_model):

    t = torch.arange(max_timesteps).float().unsqueeze(1)

    div_term = torch.exp(torch.arange(0, d_model, 2).float()
                         * -(math.log(10000.0) / d_model))

    time_enc = torch.zeros(max_timesteps, d_model)

    time_enc[:, 0::2] = torch.sin(t * div_term)
    time_enc[:, 1::2] = torch.cos(t * div_term)

    return time_enc


def linear_noise_schedule(max_timesteps, beta_min=0.0001, beta_max=0.02):

    beta_schedule = np.linspace(beta_min, beta_max, max_timesteps)
    betas_tensor = torch.tensor(beta_schedule).float()

    return betas_tensor


def cosine_schedule(timesteps, beta_min=0.1, beta_max=0.2):
    steps = torch.linspace(0, 1, timesteps)
    betas = 0.5 * (1 - torch.cos(steps * np.pi))
    betas = beta_min + betas * (beta_max - beta_min)
    return betas


def noise_schedule_2_0(num_steps=500, max_beta=0.02):
    steps = range(0, num_steps)

    def alpha_bar(t):
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []

    for i in steps:
        t1 = i / num_steps
        t2 = (i + 1) / num_steps
        beta = min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta)
        betas.append(beta)

    betas_tensor = torch.tensor(betas)

    alphas = 1 - betas_tensor
    alpha_bar_tensor = torch.cumprod(alphas, dim=0)

    return betas_tensor, alphas, alpha_bar_tensor


def save_model(model, model_pth):

    model_to_save = model.module.cpu() if isinstance(model, DDP) else model.cpu()
    torch.save({
        'model_state_dict': model_to_save.state_dict(),
    }, model_pth)
    print(f"Model is saved in: {model_pth}")


def load_model(model_pth, model, optimizer=None):
    checkpoint = torch.load(model_pth)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer


def save_checkpoint(model, time_cost, dir_out, world_size, train_loss, test_loss):

    print(f"Total training time: {time_cost // 3600} hours "
          f"{(time_cost % 3600) // 60} minutes {(time_cost % 3600) % 60} seconds")

    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    pth = f"{dir_out}/dit_{now}.pth"
    if world_size > 1:
        if dist.get_rank() == 0:

            save_model(model, pth)
    else:
        save_model(model, pth)

    with open(f'{dir_out}/training_loss_{now}.txt', 'w') as f:
        for loss in train_loss:
            f.write(f"{loss}\n")

    if test_loss is not None:
        with open(f'{dir_out}/test_loss_{now}.txt', 'w') as f:
            for loss in test_loss:
                f.write(f"{loss}\n")


def display_epoch_loss(epoch, loss, world_size, training=True):
    stage = "training" if training else "test"
    if world_size > 1:
        if dist.get_rank() == 0:
            print(f'Epoch {epoch + 1} {stage}, Loss: {loss}')
    else:
        print(f'Epoch {epoch + 1} {stage}, Loss: {loss}')


def write_readme(f_conf, readme):
    with open(f_conf, 'r') as f:
        data = json.load(f)

    with open(readme, 'w') as f:
        json.dump(data, f, indent=4)


def write_readme_conf(conf, readme):

    with open(readme, 'w') as f:
        json.dump(conf, f, indent=4)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):

    if schedule_name == "linear":

        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == 'sqrt':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1-np.sqrt(t + 0.0001),
        )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar_left(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,
        )
    elif schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001
        beta_end = scale * 0.02
        first_part = np.linspace(
            beta_start, beta_mid, 10, dtype=np.float64
        )
        second_part = np.linspace(
            beta_mid, beta_end, num_diffusion_timesteps - 10, dtype=np.float64
        )
        return np.concatenate(
            [first_part, second_part]
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar_left(num_diffusion_timesteps, alpha_bar, max_beta=0.999):

    betas = []
    betas.append(min(1-alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps-1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def mean_flat(tensor):

    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def forward_diffusion(tgt_sequence, t_batch, noise_schedule):

    noisy_sequence = tgt_sequence.clone()
    batch_size, seq_len, dim = tgt_sequence.shape
    device = tgt_sequence.device

    alpha_t = 1 - noise_schedule
    alpha_bar_t = torch.cumprod(alpha_t, dim=0)

    epsilon_batch = torch.randn(batch_size, seq_len, dim, device=device)

    alpha_bar_t_val = alpha_bar_t[t_batch]
    sqrt_one_minus_alpha_bar_t_val = torch.sqrt(1 - alpha_bar_t_val)

    noisy_sequence = torch.sqrt(alpha_bar_t_val).unsqueeze(1).unsqueeze(
        2) * noisy_sequence + sqrt_one_minus_alpha_bar_t_val.unsqueeze(1).unsqueeze(2) * epsilon_batch

    return noisy_sequence, epsilon_batch


def generate_domain_zero_feature(p_ids, padded_protein_features, domain_file):

    for i, p_id in enumerate(p_ids):
        if p_id in domain_file:

            domains = domain_file[p_id]

            for domain in domains:
                start, end = domain

                padded_protein_features[i, start:end+1, :] = 0.0

    return padded_protein_features
