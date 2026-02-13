import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from DRetHTR.model.FeatureExtractor import FeatureExtractor
class DRetHTR(torch.nn.Module):
    def __init__(self, decoder_mode="Transformer",
                 vocab_size = 83,
                 image_size = (28,420), patch_size = (4,4) , num_channels = 1, img_emb_dropout = 0.1, patch_order=0,
                 embed_dim = 128, d_model = 128, decoder_attention_heads = 4, decoder_ffn_dim = 512, decoder_depth = 3, decoder_dropout=0.3, decoder_emb_dropout=0.1,
                 cnn_dropout=0,
                 feature_extractor="Patch_embedding",
                 D_norm=True, ret_norm=True, gamma_subtracter=0, various_gamma_in_heads=False, increase_gamma_along_layers=False, text_length=94):

        super().__init__()

        self.feature_extractor = FeatureExtractor(feature_extractor, image_size=image_size, patch_size=patch_size, num_channels=num_channels,
                 dim=d_model, patch_order=patch_order, cnn_dropout=cnn_dropout, img_emb_dropout = img_emb_dropout)

        self.embed_tokens = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.embed_dropout = nn.Dropout(decoder_emb_dropout)
        self.txt_position_embedding = SinusoidalPositionEmbeddings(embed_dim)
        self.txt_position_embedding_txt_length = self.txt_position_embedding(text_length, "cuda")

        double_v_dim = False
        self.v_dim = d_model * 2 if double_v_dim else d_model

        self.dropout = nn.Dropout(decoder_dropout)

        self.retnet = Retnet(d_model, decoder_depth, decoder_attention_heads, d_model//decoder_attention_heads, decoder_ffn_dim, decoder_dropout, decoder_mode, D_norm=D_norm, ret_norm=ret_norm, gamma_subtracter=gamma_subtracter, various_gamma_in_heads=various_gamma_in_heads, increase_gamma_along_layers=increase_gamma_along_layers, text_length=text_length)
        self.generator = nn.Linear(in_features=d_model, out_features=vocab_size, bias=False)

        self.depth = decoder_depth
        self.heads = decoder_attention_heads
        self.d_model = d_model
        self.decoder_mode = decoder_mode

    def forward(self, img, decoder_input_ids, pad_positions=None, img_pad_ratios=None):
        x = self.feature_extractor(img)
        b, image_patches_length, _ = x.shape

        decoder_input_ids = self.embed_tokens(decoder_input_ids)
        b, text_length, _ = decoder_input_ids.shape
        decoder_input_ids = self.embed_dropout(decoder_input_ids)

        if "Transformer" in self.decoder_mode  or "Sinusoidal" in self.decoder_mode :
            pos_embeddings = self.txt_position_embedding(text_length, decoder_input_ids.device)
            decoder_input_ids = decoder_input_ids + pos_embeddings.unsqueeze(0).expand(b, -1, -1)

        X = torch.cat((x, decoder_input_ids), dim=1)

        X = self.retnet(X, image_patches_length, text_length, pad_positions)

        X = X[:,image_patches_length:]
        Y_parallel = self.generator(X)

        return Y_parallel

    def decode(self,decoder_input_ids, img_patches, img_pad_ratios=None):

        image_patches_length = img_patches.shape[1]
        decoder_input_ids = self.embed_tokens(decoder_input_ids)
        b, n, _ = decoder_input_ids.shape
        text_length = n
        decoder_input_ids = self.embed_dropout(decoder_input_ids)

        if "Transformer" in self.decoder_mode  or "Sinusoidal" in self.decoder_mode :
            # pos_embeddings = self.txt_position_embedding(n, decoder_input_ids.device)
            pos_embeddings = self.txt_position_embedding_txt_length[:n]
            decoder_input_ids = decoder_input_ids + pos_embeddings.unsqueeze(0).expand(b, -1, -1)

        X = torch.cat((img_patches, decoder_input_ids), dim=1)

        X = self.retnet(X, image_patches_length, text_length)

        X = X[:, image_patches_length:]

        return X

    def decode_effective(self, decoder_input_ids, img_patches, k_prev_list, v_prev_list, i, img_pad_ratios=None) :
        decoder_input_ids = self.embed_tokens(decoder_input_ids)
        b, n, _ = decoder_input_ids.shape
        image_patches_length = img_patches.shape[1]
        decoder_input_ids = self.embed_dropout(decoder_input_ids)
        # if self.decoder_mode == "Transformer"  :
        pos_embeddings = self.txt_position_embedding_txt_length
        decoder_input_ids = decoder_input_ids + pos_embeddings[i:i+1].unsqueeze(0).expand(b, -1, -1)

        # X = torch.cat((img_patches, decoder_input_ids), dim=1)
        X = decoder_input_ids
        X, k_list, v_list = self.retnet.forward_effective(X, k_prev_list, v_prev_list, image_patches_length, img_pad_ratios=img_pad_ratios)
        # X = X[:, image_patches_length:]
        return X, k_list, v_list

    def decode_recurrent(self, decoder_input_ids, s_n_1s, k_img_list, v_img_list, i, img_pad_ratios=None) :
        decoder_input_ids = self.embed_tokens(decoder_input_ids)
        b, n, _ = decoder_input_ids.shape
        decoder_input_ids = self.embed_dropout(decoder_input_ids)
        pos_embeddings = self.txt_position_embedding_txt_length
        decoder_input_ids = decoder_input_ids + pos_embeddings[i:i+1].unsqueeze(0).expand(b, -1, -1)

        # X = torch.cat((img_patches, decoder_input_ids), dim=1)
        X = decoder_input_ids
        X = self.retnet.forward_recurrent(X, s_n_1s, k_img_list, v_img_list, i, img_pad_ratios=img_pad_ratios)
        # X = X[:, image_patches_length:]
        return X


    def decode_recurrent_retnorm(self, decoder_input_ids, s_n_1s, k_img_list, v_img_list, k_prev_list, i, img_pad_ratios=None) :
        decoder_input_ids = self.embed_tokens(decoder_input_ids)
        b, n, _ = decoder_input_ids.shape
        decoder_input_ids = self.embed_dropout(decoder_input_ids)
        pos_embeddings = self.txt_position_embedding_txt_length
        decoder_input_ids = decoder_input_ids + pos_embeddings[i:i+1].unsqueeze(0).expand(b, -1, -1)

        # X = torch.cat((img_patches, decoder_input_ids), dim=1)
        X = decoder_input_ids
        X = self.retnet.forward_recurrent_retnorm(X, s_n_1s, k_img_list, v_img_list, k_prev_list, i, img_pad_ratios=img_pad_ratios)
        # X = X[:, image_patches_length:]
        return X


    def get_KV_img(self, img_patches, img_pad_ratios):
        k_prev_list, v_prev_list = self.retnet.get_KV_img(img_patches, img_pad_ratios)
        return k_prev_list, v_prev_list

class Retnet(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., mode="Transformer", D_norm=True, ret_norm=True, gamma_subtracter=0, various_gamma_in_heads=False, increase_gamma_along_layers=False, text_length=94):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MSARMF(dim, heads = heads, dim_head = dim_head, dropout = dropout, mode=mode, D_norm=D_norm, ret_norm=ret_norm, gamma_subtracter=gamma_subtracter, i=i, various_gamma_in_heads=various_gamma_in_heads, increase_gamma_along_layers=increase_gamma_along_layers, depth=depth, text_length=text_length)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x, image_patches_length, text_length, pad_positions=None):
        for msarmf, ff in self.layers:
            x = msarmf(x, image_patches_length, text_length, pad_positions) + x
            x = ff(x) + x
        return x

    def get_KV_img(self, img_patches, img_pad_ratios=None) :
        k_list = []; v_list = []
        for attn, ff in self.layers:
            out, k, v = attn.get_KV_img(img_patches, img_pad_ratios)
            img_patches = out + img_patches
            k_list.append(k); v_list.append(v)
            img_patches = ff(img_patches) + img_patches
        return k_list, v_list

    def forward_effective(self, x_n, k_prev_list, v_prev_list,image_patches_length, img_pad_ratios=None):
        k_list = [] ; v_list = []
        i_th_layer = 0
        for attn, ff in self.layers:
            o_n, k, v = attn.forward_effective(x_n, k_prev=k_prev_list[i_th_layer], v_prev=v_prev_list[i_th_layer],image_patches_length=image_patches_length, img_pad_ratios=img_pad_ratios)
            x_n = o_n + x_n
            k_list.append(k); v_list.append(v)
            x_n = ff(x_n) + x_n
            i_th_layer += 1
        return x_n, k_list, v_list

    def forward_recurrent(self, x_n, s_n_1s, k_img_list, v_img_list, i, img_pad_ratios=None):
        s_ns = []
        i_th_layer = 0
        for attn, ff in self.layers:
            # x = attn(x, image_patches_length, text_length) + x
            o_n, s_n = attn.forward_recurrent(x_n, s_n_1s[i_th_layer], k_img=k_img_list[i_th_layer], v_img=v_img_list[i_th_layer], i=i, img_pad_ratios=img_pad_ratios)
            x_n = o_n + x_n
            s_ns.append(s_n)

            x_n = ff(x_n) + x_n
            i_th_layer += 1
        return x_n, s_ns

    def forward_recurrent_retnorm(self, x_n, s_n_1s, k_img_list, v_img_list, k_prev_list, i, img_pad_ratios=None):
        s_ns = []; k_list = []
        i_th_layer = 0
        for attn, ff in self.layers:
            # x = attn(x, image_patches_length, text_length) + x
            o_n, s_n, k = attn.forward_recurrent_retnorm(x_n, s_n_1s[i_th_layer], k_img=k_img_list[i_th_layer], v_img=v_img_list[i_th_layer], k_prev=k_prev_list[i_th_layer], i=i, img_pad_ratios=img_pad_ratios)
            x_n = o_n + x_n
            s_ns.append(s_n)
            k_list.append(k)

            x_n = ff(x_n) + x_n
            i_th_layer += 1
        return x_n, s_ns, k_list

class MSARMF(nn.Module): # Multi-Scale-Attention-Retention-Modality-Fusion
    def __init__(self, dim, heads=4, dim_head=32, dropout=0., mode="Transformer", D_norm=True, ret_norm=True, gamma_subtracter=0, i=0, various_gamma_in_heads=False, increase_gamma_along_layers=False, depth=4, text_length=94):
        super().__init__()
        self.mode = mode
        self.i = i  # i-th layer
        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.gamma_subtracter = gamma_subtracter
        self.D_norm = D_norm
        self.ret_norm = ret_norm

        self.to_q = nn.Linear(dim, heads * dim_head, bias=False)
        self.to_kv = nn.Linear(dim, 2 * heads * dim_head, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(nn.Linear(heads * dim_head, dim), nn.Dropout(dropout))

        if "RetNet" in mode : # use Retention
            if "RetNet3" in mode : # use Data-Dependent Retention
                self.to_gt = nn.Linear(dim, heads, bias=False)
                gate_logit_scaler = 1
                self.gate_logit_normalizer = 16 * gate_logit_scaler
            else :
                if various_gamma_in_heads :
                    # self.gammas = torch.linspace(0.1088, 0.9688, heads).detach().cpu().tolist()
                    self.gammas = torch.linspace(1 - self.gamma_subtracter - torch.exp(torch.log(torch.tensor(1 / 32))),
                                             1 - torch.exp(torch.log(torch.tensor(1 / 32))), heads).tolist()
                else :
                    if increase_gamma_along_layers :
                        self.gammas = (
                            1 - self.gamma_subtracter*(1-(self.i/(depth-1))) - torch.exp(torch.linspace(math.log(1 / 32), math.log(1 / 512), heads))).detach().cpu().tolist()
                    else :
                        self.gammas = (
                            1 - self.gamma_subtracter - torch.exp(torch.linspace(math.log(1 / 32), math.log(1 / 512), heads))).detach().cpu().tolist()
                self.gammas_tensor = torch.tensor(self.gammas, device="cuda").view(1, heads, 1, 1)
                self.D_tensor = self._get_D(text_length)

                if self.D_norm:
                    self.D_tensor = self.D_tensor / self.D_tensor.sum(dim=-1, keepdim=True).sqrt()

            if "to_out" in self.mode: # No groupnorm, swish gate
                pass
            elif "group_norm" in self.mode or "group_split_norm" in self.mode :
                self.group_norm = nn.GroupNorm(heads, dim)
            else :
                self.group_norm = nn.GroupNorm(heads, dim)

                self.to_g = nn.Linear(dim, dim, bias=False)
                self.swish = lambda x: x * torch.sigmoid(x)

        elif "Transformer" in mode and "with_gamma" in mode:
            if various_gamma_in_heads:
                # self.gammas = torch.linspace(0.1088, 0.9688, heads).detach().cpu().tolist()
                self.gammas = torch.linspace(1 - self.gamma_subtracter - torch.exp(torch.log(torch.tensor(1 / 32))),
                                             1 - torch.exp(torch.log(torch.tensor(1 / 32))), heads).tolist()
            else:
                if increase_gamma_along_layers:
                    self.gammas = (
                            1 - self.gamma_subtracter * (1 - (self.i / (depth - 1))) - torch.exp(
                        torch.linspace(math.log(1 / 32), math.log(1 / 512), heads))).detach().cpu().tolist()
                else:
                    self.gammas = (
                            1 - self.gamma_subtracter - torch.exp(
                        torch.linspace(math.log(1 / 32), math.log(1 / 512), heads))).detach().cpu().tolist()
            self.gammas_tensor = torch.tensor(self.gammas, device="cuda").view(1, heads, 1, 1)
            self.D_tensor = self._get_D(text_length)

            if self.D_norm:
                self.D_tensor = self.D_tensor / self.D_tensor.sum(dim=-1, keepdim=True).sqrt()

        if "Bi_ret" in mode : # add bidirectional seuqential prior at image tokens
            n = torch.arange(140, device="cuda").unsqueeze(1) # number of image tokens(rows)
            m = torch.arange(140, device="cuda").unsqueeze(0) # number of image tokens(columns)
            n_m_diff = torch.abs(n - m)
            if "no_lwgs" in mode :
                gammas = (
                        1 - torch.exp(
                    torch.linspace(math.log(1 / 32), math.log(1 / 512), heads))).detach().cpu().tolist()
            else :
                if increase_gamma_along_layers:
                    gammas = (
                            1 - self.gamma_subtracter * (1 - (self.i / (depth - 1))) - torch.exp(
                        torch.linspace(math.log(1 / 32), math.log(1 / 512), heads))).detach().cpu().tolist()
                else:
                    gammas = (
                            1 - torch.exp(
                        torch.linspace(math.log(1 / 32), math.log(1 / 512), heads))).detach().cpu().tolist()

            gamma_tensor = torch.tensor(gammas, device="cuda").view(-1, 1, 1)
            D_tensor = gamma_tensor ** n_m_diff
            D_tensor[torch.isnan(D_tensor)] = 0
            D_tensor = D_tensor/ D_tensor.sum(dim=-1, keepdim=True).sqrt() # d_norm
            self.D_tensor_img = D_tensor

    def forward(self, x, image_patches_length, text_length, pad_positions=None):
        b, n, _, h = *x.shape, self.heads
        context = x

        q = self.to_q(x)
        kv = self.to_kv(context)
        k, v = kv.chunk(2, dim=-1)

        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        k = rearrange(k, "b n (h d) -> b h n d", h=h)
        v = rearrange(v, "b n (h d) -> b h n d", h=h)

        dots = einsum('bhid,bhjd->bhij', q, k) * self.scale

        if "RetNet" in self.mode :
            dots1, dots2 = torch.split(dots, [image_patches_length, text_length], dim=-1) # is this right? --> yes
            if "Bi_ret" in self.mode :
                attn1 = self.softmax(dots1)
                attn1 = attn1.clone()

                # dots1 = dots1.clone()
                D = self.D_tensor_img[None, :, :, :].to(device=attn1.device, dtype=attn1.dtype)
                attn1[:, :, :image_patches_length, :image_patches_length] = attn1[:, :, :image_patches_length, :image_patches_length] * D  # out-of-place multiply, assignment into cloned base
                # dots1[:, :, :image_patches_length, :image_patches_length] = dots1[:, :, :image_patches_length, :image_patches_length] * self.D_tensor_img[None, :, :, :]    # broadcast -> (b,h,I,I)

            else :
                attn1 = self.softmax(dots1)
            if "RetNet3" in self.mode :
                gt = self.to_gt(x[:,image_patches_length:])
                gt = rearrange(gt, "b n h -> b h n", h=h)
                causal_mask = torch.full([text_length, text_length], float("-inf"), device=q.device)
                causal_mask = torch.triu(causal_mask, 1).type_as(q)
                gt = F.logsigmoid(gt).cumsum(-1) / self.gate_logit_normalizer
                distance = gt[..., None] - gt[..., None, :]
                D_tensor = (distance + causal_mask).exp()
                if self.D_norm :
                    D_tensor = D_tensor / D_tensor.sum(dim=-1, keepdim=True).sqrt()
                mask = torch.zeros(b, h, n, text_length, device=attn1.device)
                mask[:,:, image_patches_length:, :] = D_tensor
            else :
                mask = torch.zeros(h, n, text_length, device=attn1.device)
                mask[:, image_patches_length:, :] = self.D_tensor[:, :text_length, :text_length]

            attn2 = dots2 * mask
            if self.ret_norm:
                norm = attn2.detach().abs().sum(dim=-1, keepdim=True).clamp_(min=1) # Use in-place operations to save memory and possibly improve speed
                attn2.div_(norm)
            attn = torch.cat([attn1, attn2], dim=-1)

            if pad_positions is not None:
                bsz, h, n, _ = attn.shape
                positions = torch.arange(n, device=attn.device).unsqueeze(0)  # Shape: (1, n)
                pad_positions_expanded = image_patches_length + pad_positions.unsqueeze(1)  # Shape: (bsz, 1)
                mask = positions < pad_positions_expanded  # Shape: (bsz, n)
                mask_2d = mask.unsqueeze(2) & mask.unsqueeze(1)  # Shape: (bsz, n, n)
                padding_mask = mask_2d.unsqueeze(1)  # Shape: (bsz, 1, n, n)
                attn = attn * padding_mask.float().clamp(min=1e-8)

            attn = self.dropout(attn)
            out = einsum('bhij,bhjd->bhid', attn, v)
            out = rearrange(out, "b h n d -> b n (h d)")

            if "to_out" in self.mode : # No group norm & swish
                if "sigmoid" in self.mode :
                    out = self.to_out(out.sigmoid())
                else :
                    out = self.to_out(out)
            elif "group_norm" in self.mode : #group norm for all tokens
                out_shape = out.shape
                out = self.group_norm(out.reshape(-1, self.dim)).reshape(out_shape)
                out = self.to_out(out)
            elif "group_split_norm" in self.mode :  # group norm for just text tokens
                out_i, out_t = torch.split(out, [image_patches_length, text_length], dim=-2) # here should be -2
                out_t_shape = out_t.shape
                out_t = self.group_norm(out_t.reshape(-1, self.dim)).reshape(out_t_shape)
                out = torch.cat([out_i, out_t], dim=-2)
                out = self.to_out(out)
            elif "split_out" in self.mode : # group norm and swish for just text token
                out_i, out_t = torch.split(out, [image_patches_length, text_length], dim=-2) # here should be -2
                out_t_shape = out_t.shape
                out_t = self.group_norm(out_t.reshape(-1, self.dim)).reshape(out_t_shape)
                out_t = self.swish(self.to_g(x[:, image_patches_length:, :])) * out_t
                out = torch.cat([out_i, out_t], dim=-2)
                out = self.to_out(out)
            else : # group norm and swish for all tokens
                # out1, out2 = torch.split(out, [image_patches_length, text_length], dim=-2) # here should be -2
                # out1 = self.to_out(out1)
                # out2_shape = out2.shape
                # out2 = self.group_norm(out2.reshape(-1, self.dim)).reshape(out2_shape)
                # out2 = self.to_o(self.swish(self.to_g(x[:,image_patches_length:,:])) * out2)
                # out = torch.cat([out1, out2], dim=-2)

                out_shape = out.shape
                out = self.group_norm(out.reshape(-1, self.dim)).reshape(out_shape)
                # out = self.to_o(self.swish(self.to_g(x[:,image_patches_length:,:])) * out)
                out = self.to_out(self.swish(self.to_g(x)) * out)

        elif "Transformer" in self.mode :

            def chk(name, t):
                ok = torch.isfinite(t).all().item()
                if not ok:
                    print("NON-FINITE:", name, "dtype", t.dtype,
                          "min/max", t.min().item(), t.max().item())
                    raise RuntimeError(name)

                chk("x", x)
                chk("q", q)
                chk("k", k)
                chk("v", v)
                chk("dots", dots)

            if "split_softmax" in self.mode :
                I = image_patches_length
                T = text_length

                dots1, dots2 = dots.split([I, T], dim=-1)  # dots1: (b,h,n,I), dots2: (b,h,n,T)
                b, h, n, _ = dots2.shape
                assert n == I + T, f"Expected n==I+T, got n={n}, I={I}, T={T}"

                # ---- attn1 over image keys (for ALL queries) ----
                attn1 = F.softmax(dots1, dim=-1, dtype=torch.float32).to(dots1.dtype)
                chk("attn1", attn1)
                if "Bi_ret" in self.mode:
                    attn1 = attn1.clone()
                    D_img = self.D_tensor_img[None, :, :I, :I].to(device=attn1.device, dtype=attn1.dtype)
                    D_img = torch.nan_to_num(D_img, nan=0.0, posinf=0.0, neginf=0.0)
                    attn1[:, :, :I, :I].mul_(D_img)
                    chk("Bi_ret_attn1", attn1)

                # ---- attn2 over text keys: ONLY for text queries ----
                attn2 = dots2.new_zeros((b, h, n, T))  # image-query rows stay 0 by design

                # text queries are rows [I : I+T]
                dots2_t = dots2[:, :, I:I + T, :]  # (b,h,T,T)

                # your mask already includes:
                # - image queries blocked (we don't softmax those rows anyway)
                # - causal for text-text
                mask_full = self.generate_text_part_mask(n, I, T, x.device)  # (n, T), bool
                mask_t = mask_full[I:I + T, :]  # (T, T), bool

                # Use finite NEG (not -inf) and fp32 softmax; then zero masked probs
                NEG = torch.finfo(dots2_t.dtype).min
                dots2_t = dots2_t.masked_fill(mask_t[None, None, :, :], NEG)
                chk("dots2_t", dots2_t)

                attn2_t = F.softmax(dots2_t, dim=-1, dtype=torch.float32).to(dots2_t.dtype)
                attn2_t = attn2_t.masked_fill(mask_t[None, None, :, :], 0.0)
                attn2_t = torch.nan_to_num(attn2_t, nan=0.0, posinf=0.0, neginf=0.0)
                chk("attn2_t", attn2_t)
                if "with_gamma" in self.mode:
                    D_txt = self.D_tensor[None, :, :T, :T].to(device=attn2_t.device, dtype=attn2_t.dtype)
                    D_txt = torch.nan_to_num(D_txt, nan=0.0, posinf=0.0, neginf=0.0)
                    attn2_t = attn2_t * D_txt  # old behavior: post-softmax scaling, no renorm
                    chk("with_gamma_attn2_t", attn2_t)

                attn2[:, :, I:I + T, :] = attn2_t
                chk("attn2", attn2)

                # concat image-key probs + text-key probs
                attn = torch.cat([attn1, attn2], dim=-1)  # (b,h,n,I+T)
                chk("attn", attn)
            else :
                if ("Bi_ret" in self.mode) or ("with_gamma" in self.mode):

                    I = image_patches_length
                    T = text_length

                    mask = self.generate_modality_fusion_mask(n, I, T, x.device)  # bool: True means "disallow"

                    # 1) Mask with FINITE value (avoids -inf/-inf -> NaN inside softmax)
                    NEG = torch.finfo(dots.dtype).min
                    dots_masked = dots.masked_fill(mask[None, None, :, :], NEG)
                    chk("dots_masked", dots_masked)

                    # 2) Softmax in fp32 for stability, then cast back
                    attn = F.softmax(dots_masked, dim=-1, dtype=torch.float32).to(dots.dtype)

                    # 3) Force masked positions to 0 (important when NEG is finite)
                    attn = attn.masked_fill(mask[None, None, :, :], 0.0)

                    # 4) (Optional but safe) kill any remaining non-finite values
                    attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

                    # 5) Apply your old post-softmax scaling (must clone before in-place edits)
                    attn = attn.clone()

                    if "Bi_ret" in self.mode:
                        D_img = self.D_tensor_img[None, :, :I, :I].to(device=attn.device, dtype=attn.dtype)
                        D_img = torch.nan_to_num(D_img, nan=0.0, posinf=0.0, neginf=0.0)
                        attn[:, :, :I, :I].mul_(D_img)
                        chk("Bi_ret_attn", attn)

                    if "with_gamma" in self.mode:
                        D_txt = self.D_tensor[None, :, :T, :T].to(device=attn.device, dtype=attn.dtype)
                        D_txt = torch.nan_to_num(D_txt, nan=0.0, posinf=0.0, neginf=0.0)
                        attn[:, :, I:I + T, I:I + T].mul_(D_txt)
                        chk("with_gamma_attn", attn)
                    # # dots is NOT a view, so slice in-place is fine
                    # if "Bi_ret" in self.mode:
                    #     # attn[:, :, :image_patches_length, :image_patches_length].mul_(self.D_tensor_img[None, :, :, :].to(attn.dtype))
                    #     attn[:, :, :image_patches_length, :image_patches_length] = attn[:, :, :image_patches_length, :image_patches_length] * self.D_tensor_img[None, :, :, :].to(device=attn.device, dtype=attn.dtype)
                    # if "with_gamma" in self.mode:
                    #     # attn[:, :, image_patches_length:, image_patches_length:].mul_(self.D_tensor[None, :, :text_length, :text_length].to(attn.dtype))
                    #     attn[:, :, image_patches_length:, image_patches_length:] = attn[:, :, image_patches_length:, image_patches_length:] * self.D_tensor[None, :, :text_length, :text_length].to(device=attn.device, dtype=attn.dtype)
                    #
                    # mask = torch.ones(h, n, n, device=dots.device)
                    # if "Bi_ret" in self.mode :
                    #     mask[:, :image_patches_length, :image_patches_length] = self.D_tensor_img[:, :, :]
                    # if "with_gamma" in self.mode :
                    #     mask[:, image_patches_length:, image_patches_length:] = self.D_tensor[:, :text_length, :text_length]
                    # dots = dots * mask
                # it worked in single gpu
                # if "Bi_ret" in self.mode :
                #     dots[:, :, :image_patches_length, :image_patches_length] = dots[:, :, :image_patches_length, :image_patches_length] * self.D_tensor_img[None, :, :, :]
                # if "with_gamma" in self.mode :
                #     print("dots.shape : ", dots.shape)
                #     print("self.D_tensor.shape : ", self.D_tensor.shape)
                #     print("text_length : ", text_length)
                #     print("image_patches_length : ", image_patches_length)
                #     dots[:, :, image_patches_length:, image_patches_length:] = dots[:, :, image_patches_length:, image_patches_length:] * self.D_tensor[None, :, :text_length, :text_length]

                #     # Masking: use finite min, not -inf (prevents -inf*0 NaNs)
                #     mask = self.generate_modality_fusion_mask(n, image_patches_length, text_length, x.device)
                #     NEG = torch.finfo(dots.dtype).min
                #     dots = dots.masked_fill(mask[None, None], NEG)
                #
                #     # Stable softmax in fp32
                #     attn = F.softmax(dots, dim=-1, dtype=torch.float32).to(dots.dtype)
                else :
                    mask = self.generate_modality_fusion_mask(n, image_patches_length, text_length, x.device)
                    dots = dots.masked_fill(mask[None, None, :, :], float('-inf'))
                    attn = self.softmax(dots)
            if pad_positions is not None: # padding position in text tokens
                bsz, h, n, _ = attn.shape
                positions = torch.arange(n, device=attn.device).unsqueeze(0)  # Shape: (1, n)
                pad_positions_expanded = image_patches_length + pad_positions.unsqueeze(1)  # Shape: (bsz, 1)
                mask = positions < pad_positions_expanded  # Shape: (bsz, n)
                mask_2d = mask.unsqueeze(2) & mask.unsqueeze(1)  # Shape: (bsz, n, n)
                padding_mask = mask_2d.unsqueeze(1)  # Shape: (bsz, 1, n, n)
                attn = attn * padding_mask.float().clamp(min=1e-8)

            attn = self.dropout(attn)
            out = einsum('bhij,bhjd->bhid', attn.float(), v.float()).to(v.dtype)
            out = rearrange(out, "b h n d -> b n (h d)")
            if "sigmoid" in self.mode:
                out = self.to_out(out.sigmoid())
            else :
                out = self.to_out(out)

            chk("out(before to_out)", out)

        return out

    def forward_recurrent(self, x_n, s_n_1, k_img, v_img, i, img_pad_ratios=None):
        b, n, _, h = *x_n.shape, self.heads
        q_n = self.to_q(x_n)
        kv = self.to_kv(x_n)
        k_n, v_n = kv.chunk(2, dim=-1)

        q_n = rearrange(q_n, "b n (h d) -> b h n d", h=h)
        k_n = rearrange(k_n, "b n (h d) -> b h n d", h=h)
        v_n = rearrange(v_n, "b n (h d) -> b h n d", h=h)

        dots = einsum('bhid,bhjd->bhij', q_n, k_img) * self.scale

        # recurrent decode is  always text query only (n_q=1), should skip Bi_ret
        # if "Bi_ret" in self.mode :
        #     dots = dots * self.D_tensor_img[:,i:i+1,:]
        attn = self.softmax(dots)
        out1 = einsum('bhij,bhjd->bhid', attn, v_img)

        # gammas_tensor = torch.tensor(self.gammas).view(1, h, 1, 1).to(q.device)
        KT_V = einsum('bhni,bhnj->bhij', k_n, v_n)


        if "RetNet3" in self.mode :
            gt = self.to_gt(x_n)
            gt = rearrange(gt, "b n h -> b h n", h=h)
            log_gamma = F.logsigmoid(gt) / self.gate_logit_normalizer
            self.gammas_tensor = log_gamma.exp()
        else :
            pass
        s_n = self.gammas_tensor * s_n_1 + KT_V
        # s_n shape : (16, 8, 128, 128)
        # self.gamma_tensor shape : (1, 8, 1, 1)
        if self.D_norm:
            # takes 61.743
            powers = torch.arange(1, i + 1, device=self.gammas_tensor.device).view(-1, 1, 1, 1)
            div = 1 + torch.sum(self.gammas_tensor ** powers, dim=0)
            s_n_scaled = s_n / div.sqrt()
            out2 = torch.einsum('bhij,bhjd->bhid', q_n, s_n_scaled) * self.scale
        else :
            out2 = einsum('bhij,bhjd->bhid', q_n, s_n) * self.scale

        if "to_out" in self.mode:
            out = out1 + out2
            out = rearrange(out, "b h n d -> b n (h d)")
            if "sigmoid" in self.mode:
                out = self.to_out(out.sigmoid())
            else :
                out = self.to_out(out)
        elif "group_norm" in self.mode:  # group norm for all tokens
            out = out1 + out2
            out = rearrange(out, "b h n d -> b n (h d)")
            out_shape = out.shape
            out = self.group_norm(out.reshape(-1, self.dim)).reshape(out_shape)
            out = self.to_out(out)
        elif "group_split_norm" in self.mode:
            out1 = rearrange(out1, "b h n d -> b n (h d)")
            out2 = rearrange(out2, "b h n d -> b n (h d)")
            out2_shape = out2.shape
            out2 = self.group_norm(out2.reshape(-1, self.dim)).reshape(out2_shape) # self.dim is h*d
            out = out1 + out2
            out = self.to_out(out)
        else :
            # out1 = rearrange(out1, "b h n d -> b n (h d)")
            # out1 = self.to_out(out1)
            #
            # out2 = rearrange(out2, "b h n d -> b n (h d)")
            # out_shape = out2.shape
            # out2 = self.group_norm(out2.reshape(-1, self.dim)).reshape(out_shape)
            # out2 = self.to_o(self.swish(self.to_g(x_n)) * out2)
            #
            # out = out1 + out2
            out = out1 + out2
            out = rearrange(out, "b h n d -> b n (h d)")
            out_shape = out.shape
            out = self.group_norm(out.reshape(-1, self.dim)).reshape(out_shape)
            out = self.to_out(self.swish(self.to_g(x_n)) * out)
        return out, s_n

    #retention score normalization is not nessecery in Recurrent representation because of the scale-invariant property. but to get exact same result with parallel representation, normalization can be applied by recieving the previous Keys.
    def forward_recurrent_retnorm(self, x_n, s_n_1, k_img, v_img, k_prev, i, img_pad_ratios=None):
        b, n, _, h = *x_n.shape, self.heads
        q_n = self.to_q(x_n)
        kv = self.to_kv(x_n)
        k_n, v_n = kv.chunk(2, dim=-1)

        q_n = rearrange(q_n, "b n (h d) -> b h n d", h=h)
        k_n = rearrange(k_n, "b n (h d) -> b h n d", h=h)
        v_n = rearrange(v_n, "b n (h d) -> b h n d", h=h)

        if k_prev is not None :
            k = torch.cat((k_prev, k_n), dim=2)
        else :
            k = k_n

        dots = einsum('bhid,bhjd->bhij', q_n, k_img) * self.scale
        # # recurrent decode is  always text query only (n_q=1), should skip Bi_ret
        # if "Bi_ret" in self.mode :
        #     dots = dots * self.D_tensor_img[:,i:i+1,:]
        attn = self.softmax(dots)
        out1 = einsum('bhij,bhjd->bhid', attn, v_img)
        KT_V = einsum('bhni,bhnj->bhij', k_n, v_n)

        if "RetNet3" in self.mode :
            gt = self.to_gt(x_n)
            gt = rearrange(gt, "b n h -> b h n", h=h)
            log_gamma = F.logsigmoid(gt) / self.gate_logit_normalizer
            self.gammas_tensor = log_gamma.exp().unsqueeze(-1)
        else :
            pass
        s_n = self.gammas_tensor * s_n_1  + KT_V # s_n shape : (16, 8, 128, 128) self.gamma_tensor shape : (1, 8, 1, 1)
        if self.D_norm:
            if "RetNet3" in self.mode : #data dependent retention
                if i > 0:
                    powers = torch.arange(1, i + 1, device=self.gammas_tensor.device).view(1, 1, -1, 1, 1)
                    gamma_powers = self.gammas_tensor.unsqueeze(2) ** powers
                    div = 1 + gamma_powers.sum(dim=2)
                else:
                    # Default divisor for `i = 0`
                    div = torch.ones_like(self.gammas_tensor)
                div = div.expand(-1, -1, s_n.shape[2], s_n.shape[3])  # Shape: [16, 8, dim, dim]
            else : # general retention
                powers = torch.arange(1, i + 1, device=self.gammas_tensor.device).view(-1, 1, 1, 1)
                div = 1 + torch.sum(self.gammas_tensor ** powers, dim=0)

            s_n_scaled = s_n / div.sqrt()
            out2 = torch.einsum('bhij,bhjd->bhid', q_n, s_n_scaled) * self.scale
        else :
            out2 = einsum('bhij,bhjd->bhid', q_n, s_n) * self.scale

        if "RetNet3" in self.mode:
            pass
        else :
            if self.ret_norm :
                dots = einsum('bhid,bhjd->bhij', q_n, k) * self.scale
                # print("dots shape:", dots.shape)
                # print("D_tensor slice shape:", self.D_tensor[:, i:i + 1, :i + 1].shape)
                ret = dots * self.D_tensor[:, i:i + 1, :i + 1]
                norm = ret.detach().abs().sum(dim=-1, keepdim=True).clamp_(min=1)
                out2.div_(norm)
                # applying ret_norm in recurrent form might be impossible since we can't access to qK^T and also K
                # but If we have K it is possible
                # pass
        out = out1 + out2
        out = rearrange(out, "b h n d -> b n (h d)")

        if "to_out" in self.mode:
            if "sigmoid" in self.mode:
                out = self.to_out(out.sigmoid())
            else :
                out = self.to_out(out)
        else :
            # out1 = rearrange(out1, "b h n d -> b n (h d)")
            # out1 = self.to_out(out1)
            # out2 = rearrange(out2, "b h n d -> b n (h d)")
            # out_shape = out2.shape
            # out2 = self.group_norm(out2.reshape(-1, self.dim)).reshape(out_shape)
            # out2 = self.to_o(self.swish(self.to_g(x_n)) * out2)
            # out = out1 + out2
            out_shape = out.shape
            out = self.group_norm(out.reshape(-1, self.dim)).reshape(out_shape)
            out = self.to_out(self.swish(self.to_g(x_n)) * out)

        return out, s_n, k

    #KV-cached inference
    def forward_effective(self, x_n, k_prev, v_prev, image_patches_length, img_pad_ratios=None):
        b, n, _, h = *x_n.shape, self.heads
        q_n = self.to_q(x_n)
        kv = self.to_kv(x_n)
        k_n, v_n = kv.chunk(2, dim=-1)

        q_n = rearrange(q_n, "b n (h d) -> b h n d", h=h)
        k_n = rearrange(k_n, "b n (h d) -> b h n d", h=h)
        v_n = rearrange(v_n, "b n (h d) -> b h n d", h=h)

        pre_img_text_length = k_prev.shape[2]

        if k_prev is not None and v_prev is not None:
            k = torch.cat((k_prev, k_n), dim=2)
            v = torch.cat((v_prev, v_n), dim=2)
        else:
            k = k_n;
            v = v_n

        dots = einsum('bhid,bhjd->bhij', q_n, k) * self.scale

        if "RetNet" in self.mode :
            text_length = pre_img_text_length - image_patches_length + 1
            dots1, dots2 = torch.split(dots, [image_patches_length, text_length], dim=-1)
            # # KV decode is always text query only (n_q=1), should skip Bi_ret
            # if "Bi_ret" in self.mode :
            #     dots1[:, :, :image_patches_length, :image_patches_length] = dots1[:, :, :image_patches_length, :image_patches_length] * self.D_tensor_img[None, :, :, :]    # broadcast -> (b,h,I,I)
            attn1 = self.softmax(dots1)

            mask = self.D_tensor[:, :text_length, :text_length]
            mask = mask[:,-1,:].unsqueeze(1)
            attn2 = dots2 * mask
            if self.ret_norm:
                norm = attn2.detach().abs().sum(dim=-1, keepdim=True).clamp_(min=1) # Use in-place operations to save memory and possibly improve speed
                attn2.div_(norm)
            attn = torch.cat([attn1, attn2], dim=-1)
            attn = self.dropout(attn)
            out = einsum('bhij,bhjd->bhid', attn, v)
            out = rearrange(out, "b h n d -> b n (h d)")
            out = self.to_out(out)
        else : # Transformer
            # # 	KV decode is  always text query only (n_q=1), should skip Bi_ret
            # if "Bi_ret" in self.mode:
            #     I = image_patches_length  # number of image tokens
            #
            #     # Case A: step-wise decoding (n == 1): apply ONLY ONE row of D_img
            #     if n == 1:
            #         # i = number of *previous text tokens* already in cache
            #         # k_prev includes image tokens + previous text tokens
            #         i = pre_img_text_length - I
            #         # pick the row corresponding to this step
            #         prior_row = self.D_tensor_img[:, i:i + 1, :I].unsqueeze(0)  # (1, h, 1, I)
            #         dots[:, :, :, :I] = dots[:, :, :, :I] * prior_row
            #     else:
            #         # Case B: parallel (not the case in "kv_cached forward"): full block
            #         dots[:, :, :I, :I] = dots[:, :, :I, :I] * self.D_tensor_img[None, :, :I, :I]
            if "split_softmax" in self.mode:
                text_length = pre_img_text_length - image_patches_length + 1
                dots1, dots2 = torch.split(dots, [image_patches_length, text_length], dim=-1)
                attn1 = self.softmax(dots1)
                if "with_gamma" in self.mode:
                    mask = self.D_tensor[:, :text_length, :text_length]
                    mask = mask[:, -1, :].unsqueeze(1)
                    attn2 = self.softmax(dots2)
                    attn2 = attn2 * mask
                else :
                    attn2 = self.softmax(dots2)
                attn = torch.cat([attn1, attn2], dim=-1)
            else :
                if "with_gamma" in self.mode:
                    attn = self.softmax(dots)  # because of this part?

                    I = image_patches_length
                    t_prev = 0 if k_prev is None else (k_prev.shape[2] - I)  # previous text length
                    t_new = t_prev + n  # total text keys after concat

                    # rows correspond to the new query token indices: [t_prev ... t_prev+n-1]
                    D_rows = self.D_tensor[:, t_prev:t_prev + n, :t_new]  # (h, n, t_new)

                    # apply to (query=text) x (keys=text) block only
                    attn[:, :, :, I:I + t_new] = attn[:, :, :, I:I + t_new] * D_rows.unsqueeze(0)
                else :
                    attn = self.softmax(dots)
            out = einsum('bhij,bhjd->bhid', attn.float(), v.float()).to(v.dtype)
            out = rearrange(out, "b h n d -> b n (h d)")
            if "sigmoid" in self.mode:
                out = self.to_out(out.sigmoid())
            else :
                out = self.to_out(out)
        return out, k, v

    def get_KV_img(self, x, img_pad_ratios=None):
        b, n, _, h = *x.shape, self.heads
        context = x

        q = self.to_q(x)
        kv = self.to_kv(context)
        k, v = kv.chunk(2, dim=-1)

        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        k = rearrange(k, "b n (h d) -> b h n d", h=h)
        v = rearrange(v, "b n (h d) -> b h n d", h=h)

        dots = einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = self.softmax(dots)

        if "Bi_ret" in self.mode :
            attn = attn * self.D_tensor_img

        attn = self.dropout(attn)

        out = einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        if "Transformer" in self.mode or "to_out" in self.mode or "split_out" in self.mode or "group_split_norm" in self.mode :
            if "sigmoid" in self.mode:
                out = self.to_out(out.sigmoid())
            else :
                out = self.to_out(out)
        elif "group_norm" in self.mode :
            out_shape = out.shape
            out = self.group_norm(out.reshape(-1, self.dim)).reshape(out_shape)
            out = self.to_out(out)
        else : # This part might be the error
            out_shape = out.shape
            out = self.group_norm(out.reshape(-1, self.dim)).reshape(out_shape)
            out = self.to_out(self.swish(self.to_g(x)) * out)
        # out = self.to_out(out)
        return out, k, v

    def _get_D(self, sequence_length):
        n = torch.arange(sequence_length, device="cuda").unsqueeze(1)
        m = torch.arange(sequence_length, device="cuda").unsqueeze(0)

        n_m_diff = n - m
        mask = (n_m_diff >= 0).float()

        gamma_tensor = torch.tensor(self.gammas, device="cuda").view(-1, 1, 1)
        D_tensor = gamma_tensor ** n_m_diff * mask
        D_tensor[torch.isnan(D_tensor)] = 0

        return D_tensor
    def generate_modality_fusion_mask(self, n, image_patches_length, text_length, device):
        # 0. Initialize mask
        mask = torch.zeros(n, n, device=device, dtype=torch.bool)

        # 1. no masking among the image tokens
        # 2. image tokens must not attend to text tokens
        # 3. text tokens must attend to image tokens
        mask[:image_patches_length, image_patches_length:] = 1 # 1 means blocked(masked)

        # 4. among the text tokens, a causal mask in a lower triangular structure is applied.
        causal_mask = torch.triu(torch.ones(text_length, text_length, device=device, dtype=torch.bool), diagonal=1)
        mask[image_patches_length:, image_patches_length:] = causal_mask

        return mask

    def generate_text_part_mask(self, n, image_patches_length, text_length, device):
        # 0. Initialize mask
        mask = torch.zeros(n, text_length, device=device, dtype=torch.bool)
        # 2. image tokens must not attend to text tokens
        mask[:image_patches_length, :] = 1 # 1 means blocked(masked)
        # 4. among the text tokens, a causal mask in a lower triangular structure is applied.
        causal_mask = torch.triu(torch.ones(text_length, text_length, device=device, dtype=torch.bool), diagonal=1)
        mask[image_patches_length:, :] = causal_mask

        return mask

    def generate_text_kv_mask(self,text_length, device):
        # 0. Initialize mask
        mask = torch.zeros(1, text_length, device=device, dtype=torch.bool)

        # 4. among the text tokens, a causal mask in a lower triangular structure is applied.
        causal_mask = torch.triu(torch.ones(text_length, text_length, device=device, dtype=torch.bool), diagonal=1)
        mask[image_patches_length:, :] = causal_mask

        return mask



class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, *args ,**kwargs):
        return self.fn(self.norm(x),*args, **kwargs)

    def get_KV_img(self, x, *args ,**kwargs):
        return self.fn.get_KV_img(self.norm(x), *args, **kwargs)
    def forward_effective(self, x, *args ,**kwargs):
        return self.fn.forward_effective(self.norm(x),*args, **kwargs)

    def forward_recurrent(self, x, *args ,**kwargs):
        return self.fn.forward_recurrent(self.norm(x),*args, **kwargs)

    def forward_recurrent_retnorm(self, x, *args ,**kwargs):
        return self.fn.forward_recurrent_retnorm(self.norm(x),*args, **kwargs)



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, sequence_length, device, padding_idx = None):
        # sequence_length: the length of the sequence for the embeddings
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        position_ids = torch.arange(sequence_length, device=device).unsqueeze(1)
        position_embeddings = position_ids * embeddings.unsqueeze(0)
        position_embeddings = torch.cat((position_embeddings.sin(), position_embeddings.cos()), dim=-1)
        if padding_idx is not None :
            position_embeddings[padding_idx, :] = 0
        return position_embeddings
