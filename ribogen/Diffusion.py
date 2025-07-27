import torch

from ribogen.utility import get_named_beta_schedule, mean_flat
from ribogen.utility import position_encoding
from ribogen.rounding import denoised_fn_round


class DiffusionLM:
    def __init__(self, num_steps, schedule_name="sqrt"):
        self.num_steps = num_steps

        self.beta_t = torch.tensor(get_named_beta_schedule(
            schedule_name, num_steps), dtype=torch.float32)
        self.alpha_t = 1 - self.beta_t
        self.alpha_bar_t = torch.cumprod(self.alpha_t, dim=0)
        self.alpha_bar_t_pre = torch.cat(
            [torch.tensor([1.0]), self.alpha_bar_t[:-1]])

        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar_t)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1-self.alpha_bar_t)

        self.p_mean_coef1 = (
            self.beta_t * torch.sqrt(self.alpha_bar_t_pre) /
            (1.0 - self.alpha_bar_t)
        )
        self.p_mean_coef2 = (
            torch.sqrt(self.alpha_t) * (1.0 - self.alpha_bar_t_pre) /
            (1.0 - self.alpha_bar_t)
        )

        self.posterior_variance = (
            self.beta_t * (1.0 - self.alpha_bar_t_pre) /
            (1.0 - self.alpha_bar_t)
        )

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alpha_t)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alpha_bar_t)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(
            1.0 / self.alpha_bar_t - 1)

    def q_sample(self, emb, t, r_mask, pad_id):

        emb_fix = emb.clone()
        batch_size, seq_len, dim = emb.shape
        device = emb_fix.device

        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(
            device)

        std = self.sqrt_one_minus_alpha_bar[0].unsqueeze(
            0).unsqueeze(1).unsqueeze(2)
        noise_for_x0 = torch.randn(batch_size, seq_len, dim, device=device)
        x_0 = emb_fix + std * noise_for_x0

        t = t.to(device)

        sqrt_alpha_bar_t_val = self.sqrt_alpha_bar[t].unsqueeze(1).unsqueeze(2)
        sqrt_one_minus_alpha_bar_t_val = self.sqrt_one_minus_alpha_bar[t].unsqueeze(
            1).unsqueeze(2)

        x_0 = x_0.to(device)
        epsilon = torch.randn(batch_size, seq_len, dim, device=device)

        x_t = sqrt_alpha_bar_t_val * x_0 + sqrt_one_minus_alpha_bar_t_val * epsilon

        if r_mask is None:
            return x_t, x_0, epsilon
        else:

            r_mask = r_mask.unsqueeze(dim=-1)
            r_mask_expend = r_mask.expand(-1, -1, x_t.size(-1))

            x_t = torch.where(r_mask_expend == True, x_0, x_t)

        return x_t, x_0, epsilon

    def p_sample(self, model, padded_protein_features, x_t, t, time_enc, p_mask, w, denoised_fn, clip_denoised, conf):
        device = padded_protein_features.device

        t = t.to(device)
        self.p_mean_coef1 = self.p_mean_coef1.to(device)
        self.p_mean_coef2 = self.p_mean_coef2.to(device)
        self.beta_t = self.beta_t.to(device)
        self.posterior_variance = self.posterior_variance.to(device)

        t_expanded = t.unsqueeze(1).cuda(device)
        seq_len = x_t.size(0)
        dim = x_t.size(2)
        r_pos_enc = position_encoding(seq_len, dim)
        r_pos_enc = r_pos_enc.cuda(device)
        pos_unsqueezed = r_pos_enc.unsqueeze(1)

        time_embedings = time_enc[t_expanded.squeeze(1)]
        time_embedings = time_embedings.cuda(device)

        memory = padded_protein_features

        memory_un = torch.zeros_like(padded_protein_features)

        def process_xstart(x):
            if denoised_fn is not None:
                none_arg = None
                x = denoised_fn(none_arg, model, x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        x_t = x_t.to(device)
        x_t_pos = x_t + pos_unsqueezed

        with torch.no_grad():

            p_mean_coef1 = self.p_mean_coef1[t].view(1, -1, 1).to(device)
            p_mean_coef2 = self.p_mean_coef2[t].view(1, -1, 1).to(device)

            if w:
                model.module.set_time_embeddings(time_embedings)
                output_x_0 = model(memory, x_t_pos, p_mask, None).to(device)
                output_x_0_un = model(memory_un, x_t_pos,
                                    None, None).to(device)

                output_x_0 = process_xstart(output_x_0)
                output_x_0_un = process_xstart(output_x_0_un)

                x0_guided = (1 + w) * output_x_0 - w * output_x_0_un
                p_mean = p_mean_coef1 * x0_guided + \
                    p_mean_coef2 * x_t.to(device)
            else:

                model.module.set_time_embeddings(time_embedings)

                output_x_0 = model(memory, x_t_pos, p_mask, None).to(device)
                output_x_0 = process_xstart(output_x_0)

                p_mean = p_mean_coef1 * output_x_0 + \
                    p_mean_coef2 * x_t.to(device)

        p_var = self.posterior_variance[t].to(device)

        noise = torch.randn_like(p_mean).to(device)

        nonzero_mask = (t != 0).float().view(1, -1, 1)
        nonzero_mask = nonzero_mask.expand(p_mean.shape)

        p_var = p_var.view(1, -1, 1).expand(p_mean.shape)

        output = p_mean + nonzero_mask * torch.sqrt(p_var) * noise

        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.to(device)

        return output, output_x_0, p_mean, p_mean_coef1, p_mean_coef2, sqrt_one_minus_alpha_bar[t]

    def p_sample_TEPE(self, model, padded_protein_features, x_t, t, time_enc, p_mask, w, denoised_fn, clip_denoised, conf):
        device = padded_protein_features.device

        t = t.to(device)
        self.p_mean_coef1 = self.p_mean_coef1.to(device)
        self.p_mean_coef2 = self.p_mean_coef2.to(device)
        self.beta_t = self.beta_t.to(device)
        self.posterior_variance = self.posterior_variance.to(device)

        t_expanded = t.unsqueeze(1).cuda(device)

        memory = padded_protein_features
        memory_un = torch.zeros_like(padded_protein_features)

        def process_xstart(x):
            if denoised_fn is not None:
                none_arg = None
                x = denoised_fn(none_arg, model, x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        x_t = x_t.to(device)

        with torch.no_grad():

            p_mean_coef1 = self.p_mean_coef1[t].view(1, -1, 1).to(device)
            p_mean_coef2 = self.p_mean_coef2[t].view(1, -1, 1).to(device)

            if w:
                output_x_0 = model(memory, x_t, p_mask,
                                None, t_expanded).to(device)
                output_x_0_un = model(
                    memory_un, x_t, None, None, t_expanded).to(device)

                output_x_0 = process_xstart(output_x_0)
                output_x_0_un = process_xstart(output_x_0_un)

                x0_guided = (1 + w) * output_x_0 - w * output_x_0_un
                p_mean = p_mean_coef1 * x0_guided + \
                    p_mean_coef2 * x_t.to(device)
            else:
                output_x_0 = model(memory, x_t, p_mask,
                                None, t_expanded).to(device)
                output_x_0 = process_xstart(output_x_0)

                p_mean = p_mean_coef1 * output_x_0 + \
                    p_mean_coef2 * x_t.to(device)

        p_var = torch.concat((self.posterior_variance[1].unsqueeze(0), self.beta_t[1:]))[
            t].to(device)

        noise = torch.randn_like(p_mean).to(device)

        nonzero_mask = (t != 0).float().view(1, -1, 1)
        nonzero_mask = nonzero_mask.expand(p_mean.shape)

        p_var = p_var.view(1, -1, 1).expand(p_mean.shape)

        output = p_mean + nonzero_mask * torch.sqrt(p_var) * noise

        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.to(device)

        return output, output_x_0, p_mean, p_mean_coef1, p_mean_coef2, sqrt_one_minus_alpha_bar[t]

    def p_sample_eps(self, model, padded_protein_features, x_t, t, time_enc, p_mask, w, denoised_fn, clip_denoised, conf):
        device = padded_protein_features.device

        t = t.to(device)
        self.p_mean_coef1 = self.p_mean_coef1.to(device)
        self.p_mean_coef2 = self.p_mean_coef2.to(device)
        self.beta_t = self.beta_t.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(
            device)

        t_expanded = t.unsqueeze(1).cuda(device)

        memory = padded_protein_features
        memory_un = torch.zeros_like(padded_protein_features)

        def process_xstart(x):
            if denoised_fn is not None:
                none_arg = None
                x = denoised_fn(none_arg, model, x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        x_t = x_t.to(device)

        with torch.no_grad():

            p_mean_coef1 = self.p_mean_coef1[t].view(1, -1, 1).to(device)
            p_mean_coef2 = self.p_mean_coef2[t].view(1, -1, 1).to(device)

            if w:
                output = model(memory, x_t, p_mask, None,
                            t_expanded).to(device)
                output_un = model(memory_un, x_t, None, None,
                                t_expanded).to(device)

                output = process_xstart(output)
                output_un = process_xstart(output_un)

            else:
                output = model(memory, x_t, p_mask, None,
                            t_expanded).to(device)

        sqrt_recip_alphas = self.expand_tensor_sample(
            self.sqrt_recip_alphas[t], device)
        beta_t = self.expand_tensor_sample(self.beta_t[t], device)
        recip_sqrt_one_minus_alpha_bar = self.expand_tensor_sample(
            (1 / self.sqrt_one_minus_alpha_bar[t]), device)

        p_mean = sqrt_recip_alphas * \
            (x_t - beta_t * recip_sqrt_one_minus_alpha_bar * output)

        p_var = torch.concat((self.posterior_variance[1].unsqueeze(0), self.beta_t[1:]))[
            t].to(device)

        noise = torch.randn_like(p_mean).to(device)

        nonzero_mask = (t != 0).float().view(1, -1, 1)
        nonzero_mask = nonzero_mask.expand(p_mean.shape)

        p_var = p_var.view(1, -1, 1).expand(p_mean.shape)

        x_t_1 = p_mean + nonzero_mask * torch.sqrt(p_var) * noise

        model_output = output

        return x_t_1, model_output

    def loss_function(self, model, output_x0, x_0, emb_fix, tokens, t, r_mask):

        terms = {}
        assert output_x0.shape == x_0.shape
        terms["mse"] = mean_flat((x_0 - output_x0) ** 2)

        t0_mask = (t == 0)
        t0_loss = mean_flat((emb_fix - output_x0) ** 2)
        terms["mse"] = torch.where(t0_mask, t0_loss, terms["mse"])

        out_mean = self.sqrt_alpha_bar[self.num_steps - 1] * x_0
        tT_loss = mean_flat(out_mean ** 2)

        get_logits = model.module.get_logits
        logits = get_logits(x_0)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        decoder_nll = loss_fct(
            logits.view(-1, logits.size(-1)), tokens.view(-1)).view(tokens.shape)

        '''r_mask_r = ~r_mask
        if r_mask != None:
            decoder_nll *= r_mask_r
        
        if r_mask_r != None:
            decoder_nll = decoder_nll.sum(dim=-1)/r_mask_r.sum(dim=-1)
        else:
            decoder_nll = decoder_nll.mean(dim=-1)'''

        decoder_nll = decoder_nll.mean(dim=-1)
        terms["loss"] = terms["mse"] + decoder_nll + tT_loss

        return terms

    def DLM_predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        device = x_t.device
        t = t.to(self.sqrt_recip_alphas.device)

        sqrt_recip_alphas_cumprod = self.expand_tensor(
            self.sqrt_recip_alphas_cumprod[t], device)
        sqrt_recipm1_alphas_cumprod = self.expand_tensor(
            self.sqrt_recipm1_alphas_cumprod[t], device)

        predicted_x0 = sqrt_recip_alphas_cumprod * \
            x_t - sqrt_recipm1_alphas_cumprod * eps

        return predicted_x0

    def DLM_loss_function_eps(self, model, output, x_0, emb, tokens, t, r_mask, epsilon, x_t):

        terms = {}
        assert output.shape == x_0.shape
        terms["mse"] = mean_flat((epsilon - output) ** 2)

        t0_mask = (t == 0)

        x_t = x_t.transpose(0, 1)
        predicted_x0 = self.DLM_predict_xstart_from_eps(x_t, t, output)

        t0_loss = mean_flat((emb - predicted_x0) ** 2)
        terms["mse"] = torch.where(t0_mask, t0_loss, terms["mse"])

        out_mean = self.sqrt_alpha_bar[self.num_steps - 1] * x_0
        tT_loss = mean_flat(out_mean ** 2)

        get_logits = model.module.get_logits
        logits = get_logits(x_0)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        decoder_nll = loss_fct(
            logits.view(-1, logits.size(-1)), tokens.view(-1)).view(tokens.shape)

        '''r_mask_r = ~r_mask
        if r_mask != None:
            decoder_nll *= r_mask_r
        
        if r_mask_r != None:
            decoder_nll = decoder_nll.sum(dim=-1)/r_mask_r.sum(dim=-1)
        else:
            decoder_nll = decoder_nll.mean(dim=-1)'''

        decoder_nll = decoder_nll.mean(dim=-1)
        terms["loss"] = terms["mse"] + decoder_nll + tT_loss

        return terms

    def MY_predict_p_mean_from_eps(self, x_t, t, eps):

        assert x_t.shape == eps.shape

        device = x_t.device
        t = t.to(self.sqrt_recip_alphas.device)

        sqrt_recip_alphas = self.expand_tensor(
            self.sqrt_recip_alphas[t], device)
        beta_t = self.expand_tensor(self.beta_t[t], device)
        recip_sqrt_one_minus_alpha_bar = self.expand_tensor(
            (1 / self.sqrt_one_minus_alpha_bar[t]), device)

        predicted_x0 = sqrt_recip_alphas * \
            (x_t - beta_t * recip_sqrt_one_minus_alpha_bar * eps)

        return predicted_x0

    def MY_loss_function_eps(self, model, output, x_0, emb, tokens, t, r_mask, epsilon, x_t):

        terms = {}
        assert output.shape == x_0.shape
        terms["mse"] = mean_flat((epsilon - output) ** 2)

        t0_mask = (t == 0)
        x_t = x_t.transpose(0, 1)

        predicted_p_mean1 = self.MY_predict_p_mean_from_eps(x_t, t, output)

        t0_loss = mean_flat((emb - predicted_p_mean1) ** 2)
        terms["mse"] = torch.where(t0_mask, t0_loss, terms["mse"])

        out_mean = self.sqrt_alpha_bar[self.num_steps - 1] * x_0
        tT_loss = mean_flat(out_mean ** 2)

        get_logits = model.module.get_logits
        logits = get_logits(x_0)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        decoder_nll = loss_fct(
            logits.view(-1, logits.size(-1)), tokens.view(-1)).view(tokens.shape)

        '''r_mask_r = ~r_mask
        if r_mask != None:
            decoder_nll *= r_mask_r
        
        if r_mask_r != None:
            decoder_nll = decoder_nll.sum(dim=-1)/r_mask_r.sum(dim=-1)
        else:
            decoder_nll = decoder_nll.mean(dim=-1)'''

        decoder_nll = decoder_nll.mean(dim=-1)
        terms["loss"] = terms["mse"] + decoder_nll + tT_loss

        return terms

    def MY_loss_function_eps_p_round(self, model, output, x_0, emb, tokens, t, r_mask, epsilon, x_t):

        terms = {}
        assert output.shape == x_0.shape
        terms["mse"] = mean_flat((epsilon - output) ** 2)

        t0_mask = (t == 0)
        x_t = x_t.transpose(0, 1)

        predicted_p_mean1 = self.MY_predict_p_mean_from_eps(x_t, t, output)

        t0_loss = mean_flat((emb - predicted_p_mean1) ** 2)
        terms["mse"] = torch.where(t0_mask, t0_loss, terms["mse"])

        out_mean = self.sqrt_alpha_bar[self.num_steps - 1] * x_0
        tT_loss = mean_flat(out_mean ** 2)

        predicted_xstart = self.DLM_predict_xstart_from_eps(x_t, t, output)

        get_logits = model.module.get_logits
        logits = get_logits(predicted_xstart)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        decoder_nll = loss_fct(
            logits.view(-1, logits.size(-1)), tokens.view(-1)).view(tokens.shape)

        '''r_mask_r = ~r_mask
        if r_mask != None:
            decoder_nll *= r_mask_r
        
        if r_mask_r != None:
            decoder_nll = decoder_nll.sum(dim=-1)/r_mask_r.sum(dim=-1)
        else:
            decoder_nll = decoder_nll.mean(dim=-1)'''

        decoder_nll = decoder_nll.mean(dim=-1)
        terms["loss"] = terms["mse"] + decoder_nll + tT_loss

        return terms

    def expand_tensor(self, tensor, device):
        expanded_tensor = tensor.unsqueeze(1).unsqueeze(2)
        return expanded_tensor.to(device)

    def expand_tensor_sample(self, tensor, device):
        expanded_tensor = tensor.unsqueeze(0).unsqueeze(2)
        return expanded_tensor.to(device)
