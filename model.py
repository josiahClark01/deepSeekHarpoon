import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.weight / (x.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=128000, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings

    def forward(self, x, seq_len=None):
        positions = torch.arange(0, seq_len if seq_len else x.shape[2], device=x.device).float()
        freqs = positions[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_head, d_c, d_prime_c, d_rh):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_rh = d_rh
        self.d_c = d_c
        self.d_prime_c = d_prime_c
        self.W_DQ = nn.Linear(d_model, d_prime_c, bias=False)  # Query compression (enhanced)
        self.W_UQ = nn.Linear(d_prime_c, n_heads * d_head, bias=False)
        self.W_DKV = nn.Linear(d_model, d_c, bias=False)  # KV compression
        self.W_UK = nn.Linear(d_c, n_heads * d_head, bias=False)
        self.W_UV = nn.Linear(d_c, n_heads * d_head, bias=False)
        self.W_QR = nn.Linear(d_prime_c, n_heads * d_rh, bias=False)  # Decoupled RoPE query
        self.W_KR = nn.Linear(d_model, d_rh, bias=False)  # Shared RoPE key
        self.norm_after_latent = RMSNorm(d_c)  # Additional norm after latents (V3 enhancement)
        self.W_O = nn.Linear(n_heads * d_head, d_model, bias=False)
        self.rope = RotaryEmbedding(d_rh)

    def forward(self, x, mask=None):
        bs, seq_len, _ = x.shape
        # Query latent (enhanced with d_prime_c)
        c_Q = self.W_DQ(x)
        q_C = self.W_UQ(c_Q).view(bs, seq_len, self.n_heads, self.d_head)
        q_R = self.rope(self.W_QR(c_Q).view(bs, seq_len, self.n_heads, self.d_rh), seq_len)
        q = torch.cat((q_C, q_R), dim=-1)

        # KV latent
        c_KV = self.norm_after_latent(self.W_DKV(x))  # V3: Norm after latent
        k_C = self.W_UK(c_KV).view(bs, seq_len, self.n_heads, self.d_head)
        v_C = self.W_UV(c_KV).view(bs, seq_len, self.n_heads, self.d_head)
        k_R = self.rope(self.W_KR(x)[:, :, None, :].repeat(1, 1, self.n_heads, 1), seq_len)
        k = torch.cat((k_C, k_R), dim=-1)

        # Attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head + self.d_rh)
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = attn @ v_C.transpose(1, 2)  # [bs, heads, seq, d_head]
        out = out.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.W_O(out)

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, ff_dim):
        super().__init__()
        self.w1 = nn.Linear(d_model, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, ff_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class DeepSeekMoE(nn.Module):
    def __init__(self, d_model, ff_dim, num_shared_experts, num_routed_experts, top_k):
        super().__init__()
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_routed_experts, bias=False)
        self.shared_experts = nn.ModuleList([SwiGLUFFN(d_model, ff_dim) for _ in range(num_shared_experts)])
        self.routed_experts = nn.ModuleList([SwiGLUFFN(d_model, ff_dim) for _ in range(num_routed_experts)])

    def forward(self, x):
        bs, seq_len, d = x.shape
        x_flat = x.view(-1, d)
        gates = torch.sigmoid(self.gate(x_flat))  # V3: Sigmoid affinity
        top_k_vals, top_k_indices = torch.topk(gates, self.top_k, dim=-1)
        top_k_vals = top_k_vals / top_k_vals.sum(dim=-1, keepdim=True)  # Normalize selected

        out = torch.zeros_like(x_flat)
        # Shared experts (always active)
        for expert in self.shared_experts:
            out += expert(x_flat)
        # Routed experts
        for k in range(self.top_k):
            mask = torch.zeros_like(gates).scatter_(1, top_k_indices[:, k:k+1], 1)
            weights = top_k_vals[:, k:k+1] * mask
            for i in range(self.num_routed_experts):
                expert_out = self.routed_experts[i](x_flat) * weights[:, i:i+1]
                out += expert_out

        return out.view(bs, seq_len, d) / (self.num_shared_experts + self.top_k)

class DeepSeekLayer(nn.Module):
    def __init__(self, layer_id, d_model, n_heads, d_head, d_c, d_prime_c, d_rh, ff_dim, num_shared, num_routed, top_k):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)
        self.attn = MultiHeadLatentAttention(d_model, n_heads, d_head, d_c, d_prime_c, d_rh)
        if layer_id < 3:  # First 3 layers: Dense FFN
            self.ffn = SwiGLUFFN(d_model, ff_dim * 4)  # Approximate intermediate size
        else:  # MoE for rest
            self.ffn = DeepSeekMoE(d_model, ff_dim, num_shared, num_routed, top_k)

    def forward(self, x, mask):
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class DeepSeekModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_head, d_c, d_prime_c, d_rh, ff_dim, num_shared, num_routed, top_k, mtp_depth=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([DeepSeekLayer(i, d_model, n_heads, d_head, d_c, d_prime_c, d_rh, ff_dim, num_shared, num_routed, top_k) for i in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.mtp_heads = nn.ModuleList([nn.Linear(d_model, vocab_size, bias=False) for _ in range(mtp_depth)])  # MTP: Additional heads

    def forward(self, input_ids, labels=None, use_mtp=False):
        x = self.embed(input_ids)
        seq_len = input_ids.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)[None, None, :, :]
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            if use_mtp:  # V3 MTP (simple D=1)
                mtp_logits = self.mtp_heads[0](x[..., :-2, :])  # Predict next-next
                mtp_labels = labels[..., 2:].contiguous()
                mtp_loss = F.cross_entropy(mtp_logits.view(-1, mtp_logits.size(-1)), mtp_labels.view(-1))
                loss = loss + 0.1 * mtp_loss  # Weight as in report
        return logits, loss