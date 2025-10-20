'''Simple script for back of the napkin megatron pretraining parallelism calcs.

The accuracy of this script should be considered suspect!!! Use at your own risk!!!

Example usage:

Based on your model arch, add something like the below at the end of the file::

    model_params = {
        'num_layers': 1,
        'hidden_size': 7168,
        'num_heads': 128,
        'num_experts': 256,
        'moe_topk': 8,
        'q_lora_rank': 1536,
        'kv_lora_rank': 512,
        'expert_inter_size': 2048,
        'vocab_size': 128256,
    }

    num_nodes = 1
    gpus_per_node = 8

    parallelism_params = {
        'TP': 1,
        'PP': 1,
        'EP': 8,
        'SP': True,
        'micro_batch_size': 16,
        'global_batch_size': 4096,
        'seq_length': 4096
    }

'''

class LLMTrainingCalculator:
    def __init__(self, model_params, num_nodes, gpus_per_node, parallelism_params, gpu_memory_gb, gpu_flops_tflops, assumed_mfu=0.5):
        """
        Initializes the calculator.
        
        Args:
        - model_params (dict): Model dimensions, e.g., {'num_layers': 61, 'hidden_size': 7168, 'num_heads': 128, 
          'num_experts': 256, 'moe_topk': 8, 'q_lora_rank': 1536, 'kv_lora_rank': 512, 'expert_inter_size': 2048,
          'vocab_size': 128256}.
          Required: 'num_layers', 'num_heads', 'hidden_size'. For MoE: 'num_experts', 'moe_topk'. Optional: 'q_lora_rank', 'kv_lora_rank', 'expert_inter_size', 'vocab_size'.
        - num_nodes (int): Number of compute nodes.
        - gpus_per_node (int): GPUs per node.
        - parallelism_params (dict): Megatron params, e.g., {'TP': 1, 'PP': 1, 'EP': 8, 'ETP': 1, 'SP': True, 
          'micro_batch_size': 32, 'global_batch_size': 256, 'seq_length': 4096}.
          Required: 'TP', 'PP', 'EP', 'SP', 'micro_batch_size', 'global_batch_size', 'seq_length'. 'ETP' optional (default 1).
        - gpu_memory_gb (float): Available memory per GPU in GB (e.g., 256 for MI325X).
        - gpu_flops_tflops (float): Peak FLOPs per GPU in TFLOPS (e.g., 1307 for MI325X BF16).
        - assumed_mfu (float, optional): Assumed Model FLOPs Utilization (0-1). Default 0.5.
        """
        self.model_params = model_params
        self.num_nodes = num_nodes
        self.total_gpus = num_nodes * gpus_per_node
        self.parallelism = parallelism_params
        self.gpu_memory_gb = gpu_memory_gb
        self.gpu_flops_tflops = gpu_flops_tflops
        self.assumed_mfu = assumed_mfu

        # Validate parallelism dimensions
        tp = self.parallelism['TP']
        pp = self.parallelism['PP']
        ep = self.parallelism['EP']
        if 'ETP' in self.parallelism:
            etp = self.parallelism['ETP']
            if etp != 1 and tp != ep:
                raise ValueError("When ETP > 1, TP must equal EP.")
        else:
            etp = 1
        self.parallelism['ETP'] = etp

        self.dp = self.total_gpus // (tp * pp * ep)
        if self.total_gpus != (tp * pp * ep * self.dp):
            raise ValueError("Total GPUs must be divisible by TP * PP * EP")

        self.bytes_per_float = 2  # bf16
        self.activation_factor = 16  # Rough factor for activation tensors per layer
        self.headroom_factor = 0.8  # Assume 80% max utilization for safety

    def compute_params_per_layer(self):
        """
        Estimates parameters per layer based on MoE or dense structure.
        Returns params_per_layer (int), calculated_total_params (int)
        """
        hidden = self.model_params['hidden_size']
        num_layers = self.model_params['num_layers']
        
        q_lora_rank = self.model_params.get('q_lora_rank', hidden)
        kv_lora_rank = self.model_params.get('kv_lora_rank', hidden // self.model_params['num_heads'])
        
        # Attention params
        attn_params = hidden * q_lora_rank + 2 * (hidden * kv_lora_rank) + q_lora_rank * hidden
        
        if 'num_experts' in self.model_params:
            num_experts = self.model_params['num_experts']
            expert_inter_size = self.model_params.get('expert_inter_size', hidden * 2)
            num_shared = 1  # From script
            shared_inter_size = num_shared * expert_inter_size
            
            # SwiGLU FFN: 3 * hidden * inter (gate + up + down)
            expert_params_per = 3 * hidden * expert_inter_size
            moe_params = num_experts * expert_params_per
            
            shared_params = 3 * hidden * shared_inter_size
            
            router_params = hidden * num_experts
            
            # Norms: 2 RMSNorm (pre-attn, pre-ffn)
            norm_params = hidden * 2
            
            params_per_layer = attn_params + moe_params + shared_params + router_params + norm_params
        else:
            inter_size = self.model_params.get('intermediate_size', hidden * 4)
            ffn_params = 3 * hidden * inter_size  # SwiGLU
            norm_params = hidden * 2
            params_per_layer = attn_params + ffn_params + norm_params
        
        # Embeddings and LM head (untied)
        embed_params = 0
        if 'vocab_size' in self.model_params:
            vocab = self.model_params['vocab_size']
            embed_params = vocab * hidden * 2  # Input embed + output weights
        
        calculated_total_params = params_per_layer * num_layers + embed_params
        
        return params_per_layer, calculated_total_params

    def compute_distribution(self):
        """
        Computes and returns the model distribution as a dictionary.
        """
        output = {}
        
        # Model sharding
        num_layers = self.model_params['num_layers']
        layers_per_gpu = num_layers / self.parallelism['PP']  # Assuming even distribution
        output['layers_per_gpu'] = layers_per_gpu
        
        num_heads = self.model_params['num_heads']
        heads_per_gpu = num_heads / self.parallelism['TP']
        output['attn_heads_per_gpu'] = heads_per_gpu
        
        if 'num_experts' in self.model_params:
            num_experts = self.model_params['num_experts']
            experts_per_gpu = num_experts / self.parallelism['EP']
            output['experts_per_gpu'] = experts_per_gpu
            
            etp = self.parallelism['ETP']
            if etp > 1:
                output['expert_shard_size'] = f"Each expert sharded across {etp} GPUs"
            else:
                output['expert_shard_size'] = "Full expert on each assigned GPU"
        
        # Sequence parallelism
        sp = self.parallelism['SP']
        output['sequence_parallel'] = "Enabled" if sp else "Disabled"
        if sp:
            output['sequence_sharding_note'] = f"Activations sharded along sequence dimension across TP={self.parallelism['TP']} GPUs"
        
        # Training distribution
        mbs = self.parallelism['micro_batch_size']
        gbs = self.parallelism['global_batch_size']
        dp = self.dp
        accum_steps = gbs // (mbs * dp)
        output['data_parallel_size'] = dp
        output['micro_batch_size_per_gpu'] = mbs
        output['global_batch_size'] = gbs
        output['gradient_accumulation_steps_per_gpu'] = accum_steps
        output['effective_sequences_per_gpu_per_update'] = mbs * accum_steps
        output['tokens_per_gpu_per_update'] = output['effective_sequences_per_gpu_per_update'] * self.parallelism['seq_length']
        
        # Shard placement summary
        output['shard_placement_summary'] = (
            f"Each GPU holds {layers_per_gpu:.1f} layers (PP={self.parallelism['PP']}), "
            f"sharded tensors across TP={self.parallelism['TP']} (e.g., {heads_per_gpu:.1f} attn heads), "
        )
        if 'num_experts' in self.model_params:
            output['shard_placement_summary'] += (
                f"{experts_per_gpu:.1f} experts (EP={self.parallelism['EP']}), "
                f"with each expert {output['expert_shard_size'].lower()}, "
            )
        output['shard_placement_summary'] += f"SP={sp}. DP={dp} for data replication."

        # Params calculation
        params_per_layer, calculated_total_params = self.compute_params_per_layer()
        output['params_per_layer'] = params_per_layer
        output['calculated_total_params'] = calculated_total_params
        output['params_per_layer_billion'] = round(params_per_layer / 1e9, 2)
        output['calculated_total_params_billion'] = round(calculated_total_params / 1e9, 2)
        output['params_note'] = (
            f"Estimated params per layer: {output['params_per_layer_billion'] }B (attention + MoE + norm). "
            f"Calculated total: {output['calculated_total_params_billion'] }B (layers + embeddings if vocab provided). "
            "Estimates assume SwiGLU FFN (3 * hidden * inter) and may not include all components; actual may vary."
        )

        # Memory estimates using calculated_total_params
        hidden = self.model_params['hidden_size']
        seq_len = self.parallelism['seq_length']
        tp = self.parallelism['TP']
        mp_size = tp * self.parallelism['PP'] * self.parallelism['EP']
        per_gpu_params = calculated_total_params / mp_size

        model_gb = per_gpu_params * self.bytes_per_float / 1e9
        grad_gb = per_gpu_params * self.bytes_per_float / 1e9
        opt_gb = per_gpu_params * 4 / 1e9  # 4 bytes/param for Adam optimizer states
        activ_gb = mbs * seq_len * hidden * num_layers * self.activation_factor * self.bytes_per_float / 1e9
        if sp:
            activ_gb /= tp

        output['approx_model_memory_per_gpu_gb'] = round(model_gb, 2)
        output['approx_gradient_memory_per_gpu_gb'] = round(grad_gb, 2)
        output['approx_optimizer_memory_per_gpu_gb'] = round(opt_gb, 2)
        output['approx_activation_memory_per_mbs_gb'] = round(activ_gb, 2)
        output['approx_total_memory_per_gpu_gb'] = round(model_gb + grad_gb + opt_gb + activ_gb, 2)
        output['memory_note'] = "Rough estimates in GB for bf16 training without activation checkpointing, offloading, or other optimizations. Actual usage may vary; activations are per micro-batch and do not scale with accumulation steps."

        return output

    def analyze_fit_and_throughput(self):
        """
        Analyzes if the model fits in GPU memory and, if so, estimates throughput.
        Returns a dictionary with fit and throughput assessment.
        """
        dist = self.compute_distribution()
        total_est_gb = dist['approx_total_memory_per_gpu_gb']
        max_usable_gb = self.gpu_memory_gb * self.headroom_factor  # Apply headroom for safety

        fit_result = {}
        fit_result['will_fit'] = total_est_gb <= max_usable_gb
        fit_result['estimated_usage_gb'] = total_est_gb
        fit_result['available_after_headroom_gb'] = round(max_usable_gb, 2)
        fit_result['fit_note'] = (
            f"Model {'fits' if fit_result['will_fit'] else 'does not fit'} with {self.headroom_factor*100}% headroom. "
            f"Estimated: {total_est_gb} GB vs. Usable: {max_usable_gb} GB. "
            "Consider enabling activation checkpointing or increasing parallelism if it doesn't fit."
        )

        if fit_result['will_fit']:
            # Use calculated_total_params
            calculated_total_params = dist['calculated_total_params']
            effective_params = calculated_total_params
            if 'num_experts' in self.model_params and 'moe_topk' in self.model_params:
                num_experts = self.model_params['num_experts']
                topk = self.model_params['moe_topk']
                moe_fraction = 0.8  # Assumed fraction of params in MoE layers
                activation_ratio = topk / num_experts
                effective_ratio = (1 - moe_fraction) + (activation_ratio * moe_fraction)
                effective_params = calculated_total_params * effective_ratio

            flops_per_token = 6 * effective_params
            gpu_flops_per_sec = self.gpu_flops_tflops * 1e12 * self.assumed_mfu
            tokens_per_gpu_per_sec = gpu_flops_per_sec / flops_per_token
            fit_result['estimated_tokens_per_gpu_per_sec'] = round(tokens_per_gpu_per_sec, 2)
            fit_result['throughput_note'] = (
                f"Estimated training throughput in tokens/GPU/s, assuming 6 * effective_params FLOPs per token and {self.assumed_mfu} MFU. "
                "For MoE, effective_params adjusted by top-k activation (assumed MoE param fraction=0.8). Actual throughput depends on implementation efficiency."
            )

        return fit_result

if __name__ == "__main__":
    # Define inputs for DeepSeek-V3-671B test
    model_params = {
        'num_layers': 1,
        'hidden_size': 7168,
        'num_heads': 128,
        'num_experts': 256,
        'moe_topk': 8,
        'q_lora_rank': 1536,
        'kv_lora_rank': 512,
        'expert_inter_size': 2048,
        'vocab_size': 128256,
    }

    num_nodes = 1
    gpus_per_node = 8

    parallelism_params = {
        'TP': 1,
        'PP': 1,
        'EP': 8,
        'SP': True,
        'micro_batch_size': 16,
        'global_batch_size': 4096,
        'seq_length': 4096
    }

    gpu_memory_gb = 256.0  # MI325X
    gpu_flops_tflops = 1307.0  # MI325X BF16 peak

    # Instantiate and compute
    calculator = LLMTrainingCalculator(model_params, num_nodes, gpus_per_node, parallelism_params, gpu_memory_gb, gpu_flops_tflops)
    result = calculator.compute_distribution()
    analysis = calculator.analyze_fit_and_throughput()
    print("Distribution:", result)
    print("Analysis:", analysis)