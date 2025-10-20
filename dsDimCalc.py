'''DeepSeek-V3 is a Mixture-of-Experts (MoE) model with a transformer-like architecture, featuring Multi-Latent Attention (MLA) with LoRA-optimized projections, rotary embeddings, RMSNorm, and MoE feed-forward blocks. 

This is a basic calculator to calc gem shapes for dsv3.

For simplicity, assume:

input_batch_size = B # (e.g., MICRO_BATCH_SIZE)
sequence length $ S $ (SEQ_LEN=4096), 
hidden size $ H = 7168 $, 
attention heads $ N_h = 128 $, 
Q LoRA rank $ R_q = 1536 $, 
KV LoRA rank $ R_{kv} = 512 $, 
QK no-PE head dim $ D_{qk-nope} = 128 $, 
QK RoPE head dim $ D_{qk-rope} = 64 $, 
V head dim $ D_v = 128 $, 
number of experts $ E = 256 $, 
top-k $ K = 8 $, 
MoE intermediate size $ I_e = 2048 $, 
and shared experts $ N_s = 1 $. 
Each layer processes input $ X \in \mathbb{R}^{B \times S \times H} $. 

Operations include matrix multiplications (GEMMs), softmax, all-to-all for MoE routing, and element-wise ops (e.g., RMSNorm, SwiGLU).


Layer 1 Forward

Pre-Attention RMSNorm:

Input: $ X \in \mathbb{R}^{B \times S \times H} $
Operation: Normalize $ X $ with learned weight $ W_{norm} \in \mathbb{R}^H $: $ X_{norm} = \frac{X}{\sqrt{\frac{1}{H} \sum X^2 + \epsilon}} \odot W_{norm} $
Output: $ X_{norm} \in \mathbb{R}^{B \times S \times H} $
Tensor ops: Element-wise division, mean reduction over H, sqrt/add/scale.


Multi-Latent Attention (MLA):

Q Projection: $ Q = X_{norm} W_q $, where $ W_q \in \mathbb{R}^{H \times R_q} $, $ Q \in \mathbb{R}^{B \times S \times R_q} $

Op: GEMM (BSH @ H*R_q)


K Projection: $ K = X_{norm} W_k $, where $ W_k \in \mathbb{R}^{H \times R_{kv}} $, $ K \in \mathbb{R}^{B \times S \times R_{kv}} $

Op: GEMM


V Projection: $ V = X_{norm} W_v $, where $ W_v \in \mathbb{R}^{H \times R_{kv}} $, $ V \in \mathbb{R}^{B \times S \times R_{kv}} $

Op: GEMM


Rotary Embeddings: Apply RoPE to QK RoPE parts (QK_ROPE_HEAD_DIM=64) for positional encoding.

Op: Element-wise cos/sin application on Q/K subsets.


Attention Computation: Split Q/K/V into heads (N_h=128). Compute scores $ S = Q K^T / \sqrt{D_{qk-nope} + D_{qk-rope}} \in \mathbb{R}^{B \times N_h \times S \times S} $, softmax, then $ A = S V \in \mathbb{R}^{B \times N_h \times S \times (D_{qk-nope} + D_{qk-rope} + D_v)} $

Ops: GEMM for QK^T, softmax, GEMM for SV, reshape/flatten for heads.


Output Projection: $ O = A W_o $, where $ W_o \in \mathbb{R}^{R_q \times H} $, $ O \in \mathbb{R}^{B \times S \times H} $

Op: GEMM


Add residual: $ X_{attn} = X + O $


Pre-MoE RMSNorm:

Input: $ X_{attn} \in \mathbb{R}^{B \times S \times H} $
Output: $ X_{moe-norm} \in \mathbb{R}^{B \times S \times H} $
Op: Same as pre-attention norm.


MoE Feed-Forward:

Router: Compute scores $ R = X_{moe-norm} W_r \in \mathbb{R}^{B \times S \times E} $, where $ W_r \in \mathbb{R}^{H \times E} $, softmax top-k=8 per token.

Op: GEMM, softmax, argtopk.


Dispatch: All-to-all to route tokens to top-k experts across EP=8 GPUs (32 experts/GPU). Token tensor: ~B * S * K * H (routed subset ~ B * S * 8 * 7168).
Expert Computation: For each expert, SwiGLU FFN: Gate/up/down projections $ Y = (SiLU(G) \odot U) W_d $, where G/U = X_routed * W_g/W_u (W_g/W_u/W_d ∈ \mathbb{R}^{H × I_e} for gate/up/down).

Op: GEMMs (3 per expert), SiLU/element-wise mul.


Shared Expert: Similar FFN on full input, added to routed output.
Aggregate: All-to-all to combine expert outputs, weighted by router scores: $ O_{moe} \in \mathbb{R}^{B \times S \times H} $.
Add residual: $ X_{out} = X_{attn} + O_{moe} $

Layer 2 Forward: Same as Layer 1, input $ X_{out} $ from Layer 1.




Backward Propagation for Two Layers
Backward computes gradients w.r.t. loss (e.g., LM cross-entropy on output). Uses chain rule: Gradients flow reverse, with ops like transposed GEMMs and derivatives.
Layer 2 Backward (Starts from Loss Grad $ \nabla L \in \mathbb{R}^{B \times S \times H} $)

Post-MoE Residual Grad: $ \nabla X_{attn} = \nabla L $, $ \nabla O_{moe} = \nabla L $
MoE Backward:

Aggregate grad: All-to-all to distribute $ \nabla O_{moe} $ to experts.
Expert grad: For each expert, SwiGLU backprop: $ \nabla W_d = Y^T \nabla O_e $, $ \nabla Y = \nabla O_e W_d^T $, then derivatives for SiLU/mul, $ \nabla W_g/W_u = X_routed^T \nabla G/U $.

Ops: Transposed GEMMs, element-wise derivatives.


Shared expert grad: Similar, full tensor.
Router grad: Softmax derivative on scores, $ \nabla W_r = X_{moe-norm}^T \nabla R $.
Dispatch grad: All-to-all reverse.


Pre-MoE Norm Grad: Backprop norm derivative to $ \nabla X_{attn} $.
Post-Attention Residual Grad: $ \nabla X = \nabla X_{attn} $, $ \nabla O = \nabla X_{attn} $
Attention Backward:

Output proj grad: $ \nabla W_o = A^T \nabla O $, $ \nabla A = \nabla O W_o^T $
Attention grad: $ \nabla V = S^T \nabla A $, $ \nabla S = \nabla A V^T $, softmax deriv, $ \nabla Q = \nabla S K / \sqrt{D} $, $ \nabla K = \nabla S^T Q / \sqrt{D} $

Ops: Transposed GEMMs, softmax deriv.


RoPE deriv: Backprop through cos/sin.
Proj grads: $ \nabla W_q = X_{norm}^T \nabla Q $, similar for K/V/output.


Pre-Attention Norm Grad: Backprop to input of Layer 2 (output of Layer 1).


Layer 1 Backward: Same as Layer 2, starting from $ \nabla X $ from Layer 2 backward.

Dimensions Summary

Input/Output per Layer: $ B \times S \times H $
Attention Projections: Q $ B \times S \times R_q $, K/V $ B \times S \times R_{kv} $
Attention Scores: $ B \times N_h \times S \times S $
MoE Routed: ~ $ B \times S \times K \times H $ (dispatched subset)
Expert FFN: Input $ T \times H $ (T=routed tokens/GPU), Output $ T \times I_e $, then back to $ T \times H $

Forward/backward ops are GEMMs (O(B S H^2)), softmax (O(B S^2)), all-to-all (O(B S H * K / EP)). For 2 layers, backward adds ~2x compute (grads + chain rule). In your setup, this enables high throughput but OOM at high MBS due to activations.



'''


#TODO, complete the calc, I think I have it floating around now in 3 different locations...
# input_batch_size = 2 # B

