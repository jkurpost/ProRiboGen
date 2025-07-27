import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

from ribogen.utility import position_encoding
from ribogen.utility import get_named_beta_schedule, mean_flat
from .utils.nn import SiLU, linear, timestep_embedding
"""
DiffusionTransformer: 噪声预测模型, 采用标准的Transformer结构
"""


class DiffusionTransformer(nn.Module):
    def __init__(self, d_model, vocab_size, n_heads, n_layers, max_timesteps):
        super(DiffusionTransformer, self).__init__()

        self.hidden_dim = d_model

        self.rna_embedding = nn.Embedding(vocab_size, d_model)

        '''
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads),
            num_layers=n_layers
        )
        '''

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=n_heads, activation="gelu", dropout=0),
            num_layers=n_layers
        )

        self.max_timesteps = max_timesteps

    def forward(self, memory, tgt, p_mask, r_mask):

        output = self.decoder(
            tgt, memory, memory_key_padding_mask=p_mask, tgt_key_padding_mask=r_mask)

        return output


class DitLM(nn.Module):
    def __init__(self, d_model, vocab_size, n_heads, n_layers, max_timesteps):
        super(DitLM, self).__init__()

        self.hidden_dim = d_model

        self.rna_embedding = nn.Embedding(vocab_size, d_model)

        '''
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads),
            num_layers=n_layers
        )
        '''

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=n_heads, activation="gelu", dropout=0),
            num_layers=n_layers
        )

        self.lm_head = nn.Linear(self.hidden_dim, vocab_size)
        with torch.no_grad():
            self.lm_head.weight = self.rna_embedding.weight

        self.max_timesteps = max_timesteps

    def forward(self, memory, tgt, p_mask, r_mask):

        r_mask = None
        output = self.decoder(
            tgt, memory, memory_key_padding_mask=p_mask, tgt_key_padding_mask=r_mask)

        return output

    def get_logits(self, output):

        return self.lm_head(output)


class DitLMSmall(nn.Module):
    def __init__(self, d_input, d_model, vocab_size, n_heads, n_layers, max_timesteps):
        super(DitLMSmall, self).__init__()

        self.hidden_dim = d_model
        self.time_embeddings = None

        self.memory_linear = nn.Linear(d_input, d_model)

        self.rna_embedding = nn.Embedding(vocab_size, d_model)

        '''
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads),
            num_layers=n_layers
        )
        '''

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=n_heads, activation="gelu", dropout=0),
            num_layers=n_layers
        )

        self.lm_head = nn.Linear(self.hidden_dim, vocab_size)
        with torch.no_grad():
            self.lm_head.weight = self.rna_embedding.weight

        self.max_timesteps = max_timesteps

    def forward(self, memory, tgt, p_mask, r_mask):

        r_mask = None
        memory = self.memory_linear(memory)
        memory = memory + self.time_embeddings.unsqueeze(1)

        memory = memory.transpose(0, 1)

        output = self.decoder(
            tgt, memory, memory_key_padding_mask=p_mask, tgt_key_padding_mask=r_mask)

        return output

    def get_logits(self, output):

        return self.lm_head(output)

    def set_time_embeddings(self, time_embed):
        with torch.no_grad():
            self.time_embeddings = time_embed


class DitLMSmall_TE_PE(nn.Module):
    def __init__(self, d_input, d_model, vocab_size, n_heads, n_layers, max_timesteps, hidden_t_dim, max_position_emb_len):
        super(DitLMSmall_TE_PE, self).__init__()

        self.hidden_dim = d_model
        self.time_embeddings = None

        self.memory_linear = nn.Linear(d_input, d_model)
        self.hidden_t_dim = hidden_t_dim

        self.rna_embedding = nn.Embedding(vocab_size, d_model)

        time_embed_dim = hidden_t_dim * 4
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, self.hidden_dim),
        )

        '''
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads),
            num_layers=n_layers
        )
        '''

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=n_heads, activation="gelu", dropout=0),
            num_layers=n_layers
        )

        self.lm_head = nn.Linear(self.hidden_dim, vocab_size)
        with torch.no_grad():
            self.lm_head.weight = self.rna_embedding.weight

        self.max_timesteps = max_timesteps

        self.max_position_emb_len = max_position_emb_len
        self.position_embeddings = nn.Embedding(
            self.max_position_emb_len, self.hidden_dim)
        self.register_buffer("position_ids", torch.arange(
            self.max_position_emb_len).expand((1, -1)))

    def forward(self, memory, xt, p_mask, r_mask, t):

        r_mask = None
        memory = self.memory_linear(memory)

        t_encode = timestep_embedding(t, self.hidden_t_dim)
        t_emb = self.time_embed(t_encode)

        seq_length = xt.shape[0]
        position_ids = self.position_ids[:, :seq_length]
        position_embeddings = self.position_embeddings(position_ids)

        xt_input = xt + position_embeddings.transpose(0, 1)

        memory_time = memory.transpose(0, 1) + t_emb.transpose(0, 1)

        output = self.decoder(
            xt_input, memory_time, memory_key_padding_mask=p_mask, tgt_key_padding_mask=r_mask)

        return output

    def get_logits(self, output):

        return self.lm_head(output)

    '''def set_time_embeddings(self, time_embed):
        with torch.no_grad():
            self.time_embeddings = time_embed'''


"""
VAE: RNA tokens的编解码器, 用于将RNA tokens转换到潜在空间, 或者从潜在空间向量变换为tokens
"""


class VAE_2(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, latent_dim, pad_token_id):
        super(VAE, self).__init__()

        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=8, batch_first=True),
            num_layers=6
        )

        self.fc_mean = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )

    def encode(self, x):

        x_emb = self.embedding(x)

        position_enc = position_encoding(
            seq_len=x_emb.size(1), d_model=x_emb.size(2))

        position_enc = position_enc.unsqueeze(0).expand(x_emb.size(0), -1, -1)
        x_emb_pos = x_emb + position_enc

        x_emb_pos = x_emb_pos.permute(1, 0, 2)
        memory = self.encoder(x_emb_pos)

        mean = self.fc_mean(memory.permute(1, 0, 2))
        logvar = self.fc_logvar(memory.permute(1, 0, 2))

        return mean, logvar

    def reparameterize(self, mean, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def decode(self, z):

        logits = self.decoder(z)
        return logits

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        logits = self.decode(z)
        var = torch.exp(logvar)

        return logits, mean, logvar

    def loss_function(self, logits, x, mean, logvar):

        mask = (x != self.pad_token_id).float()

        recon_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), x.view(-1), reduction='none')

        num_non_pad_tokens = mask.sum()
        recon_loss = (recon_loss * mask.view(-1)).sum() / num_non_pad_tokens

        kl_loss = -0.5 * \
            torch.sum(1 + logvar - mean.pow(2) -
                      logvar.exp()) / num_non_pad_tokens

        total_loss = recon_loss + kl_loss
        return total_loss

    def encode_to_latent(self, x):

        with torch.no_grad():
            mean, logvar = self.encode(x)
            z = self.reparameterize(mean, logvar)

        return z


"""
VAE: RNA tokens的编解码器, 用于将RNA tokens转换到潜在空间, 或者从潜在空间向量变换为tokens
"""


class VAE_2(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, latent_dim, pad_token_id):
        super(VAE, self).__init__()

        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=8, batch_first=True),
            num_layers=6
        )

        self.fc_mean = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )

    def encode(self, x):

        x_emb = self.embedding(x)

        position_enc = position_encoding(
            seq_len=x_emb.size(1), d_model=x_emb.size(2))

        position_enc = position_enc.unsqueeze(0).expand(x_emb.size(0), -1, -1)
        x_emb_pos = x_emb + position_enc

        x_emb_pos = x_emb_pos.permute(1, 0, 2)
        memory = self.encoder(x_emb_pos)

        mean = self.fc_mean(memory.permute(1, 0, 2))
        logvar = self.fc_logvar(memory.permute(1, 0, 2))

        return mean, logvar

    def reparameterize(self, mean, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def decode(self, z):

        logits = self.decoder(z)
        return logits

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        logits = self.decode(z)
        var = torch.exp(logvar)

        return logits, mean, logvar

    def loss_function(self, logits, x, mean, logvar):

        mask = (x != self.pad_token_id).float()

        recon_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), x.view(-1), reduction='none')

        num_non_pad_tokens = mask.sum()
        recon_loss = (recon_loss * mask.view(-1)).sum() / num_non_pad_tokens

        kl_loss = -0.5 * \
            torch.sum(1 + logvar - mean.pow(2) -
                    logvar.exp()) / num_non_pad_tokens

        total_loss = recon_loss + kl_loss
        return total_loss

    def encode_to_latent(self, x):

        with torch.no_grad():
            mean, logvar = self.encode(x)
            z = self.reparameterize(mean, logvar)

        return z


class DecoderMLP(nn.Module):
    """
    从dit的embedding到rna token的MLP解码器
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DecoderMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class EmbeddingDecoderModel(nn.Module):
    """
        从dit的embedding到rna token的自编码器
    """

    def __init__(self, vocab_size, embedding_dim, input_dim, hidden_dim):
        super(EmbeddingDecoderModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.decoder_mlp = DecoderMLP(input_dim, hidden_dim, vocab_size)

    def forward(self, x):

        embedded_x = self.embedding(x)

        x = self.decoder_mlp(embedded_x)

        return x


class EmbToLogits(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, w):
        super(EmbToLogits, self).__init__()

        self.w = w
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        self.activation = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):

        x = self.fc1(x)

        x = self.activation(x)

        x = self.fc2(x)

        return x


"""
VAE: RNA tokens的编解码器, 用于将RNA tokens转换到潜在空间, 或者从潜在空间向量变换为tokens, 新版本
"""


class VAE(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, latent_dim, pad_token_id):
        super(VAE, self).__init__()

        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=8, batch_first=True),
            num_layers=6
        )

        self.fc_mean = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )

    def encode(self, x):

        x_emb = self.embedding(x)

        x_emb = x_emb.permute(1, 0, 2)
        memory = self.encoder(x_emb)

        mean = self.fc_mean(memory.permute(1, 0, 2))
        logvar = self.fc_logvar(memory.permute(1, 0, 2))

        return mean, logvar

    def reparameterize(self, mean, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def decode(self, z):

        logits = self.decoder(z)
        return logits

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        logits = self.decode(z)
        var = torch.exp(logvar)

        return logits, mean, logvar

    def loss_function(self, logits, x, mean, logvar):

        mask = (x != self.pad_token_id).float()

        recon_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), x.view(-1), reduction='none')

        num_non_pad_tokens = mask.sum()
        recon_loss = (recon_loss * mask.view(-1)).sum() / num_non_pad_tokens

        kl_loss = -0.5 * \
            torch.sum(1 + logvar - mean.pow(2) -
                      logvar.exp()) / num_non_pad_tokens

        total_loss = recon_loss + kl_loss
        return total_loss

    def encode_to_latent(self, x):

        with torch.no_grad():
            mean, logvar = self.encode(x)
            z = self.reparameterize(mean, logvar)

        return z


"""
训练样本的Dataset类
"""


class ProteinRNADataset(Dataset):
    def __init__(self, p_ids, p_lens, r_ids):
        self.p_ids = p_ids
        self.r_ids = r_ids
        self.p_lens = p_lens

    def __len__(self):
        return len(self.p_lens)

    def __getitem__(self, idx):
        return self.p_ids[idx], self.p_lens[idx], self.r_ids[idx]


class ProteinDataset(Dataset):
    def __init__(self, p_ids):
        self.p_ids = p_ids

    def __len__(self):

        return len(self.p_ids)

    def __getitem__(self, idx):
        return self.p_ids[idx]


class ProteinDatasetWithDomain(Dataset):
    def __init__(self, p_ids, rrm_domains):
        self.p_ids = p_ids
        self.rrm_domains = rrm_domains

    def __len__(self):
        return len(self.p_ids)

    def __getitem__(self, idx):
        return self.p_ids[idx], self.rrm_domains[idx]


class RNADataset(Dataset):
    def __init__(self, r_ids):
        self.r_ids = r_ids

    def __len__(self):
        return len(self.r_ids)

    def __getitem__(self, idx):
        return self.r_ids[idx]


class SingleTokenizer:
    """
        自定义的RNA tokenizer, A/C/G/U分别作为单独的token
    """
    token_to_id = {
        "A": 1,
        "C": 2,
        "G": 3,
        "U": 4,
        "<|end|>": 0,
        "<pad>": 5
    }

    eos_token = '0'

    id_to_token = {v: k for k, v in token_to_id.items()}

    def __init__(self):
        pass

    def encode(self, text, add_special_tokens=True):
        token_ids = [self.token_to_id.get(
            c, self.token_to_id["<pad>"]) for c in text]
        if add_special_tokens:
            token_ids.append(self.token_to_id["<|end|>"])
        return token_ids

    def decode(self, token_ids, skip_special_tokens=False):
        tokens = [self.id_to_token.get(t, "<pad>") for t in token_ids]
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in [
                "<|end|>", "<pad>"]]
        return "".join(tokens)
