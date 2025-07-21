import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.optim import AdamW
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
import model  # Import from model.py

# Environment setup
local_rank = int(os.environ['LOCAL_RANK'])
rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

# Hyperparameters (full size)
vocab_size = 128000
d_model = 7168
n_layers = 61  # Corrected: 61 total layers as per DeepSeek-V3 specs
n_heads = 56
d_head = 128
d_c = 512
d_prime_c = 1536
d_rh = 64
ff_dim = 2048
num_shared = 1
num_routed = 256
top_k = 8
mtp_depth = 1  # MTP depth

# Model init
model = model.DeepSeekModel(
    vocab_size=vocab_size,
    d_model=d_model,
    n_layers=n_layers,
    n_heads=n_heads,
    d_head=d_head,
    d_c=d_c,
    d_prime_c=d_prime_c,
    d_rh=d_rh,
    ff_dim=ff_dim,
    num_shared=num_shared,
    num_routed=num_routed,
    top_k=top_k,
    mtp_depth=mtp_depth
).to(torch.bfloat16)  # BF16 for memory

# Custom MoE sharding: Shard routed experts across world_size
# Simple: Assign experts to GPUs round-robin
experts_per_gpu = num_routed // world_size  # Approximate; adjust for 304 GPUs
for layer in model.layers:
    if hasattr(layer, 'ffn') and isinstance(layer.ffn, model.DeepSeekMoE):
        for i, expert in enumerate(layer.ffn.routed_experts):
            if i % world_size != rank:
                for param in expert.parameters():
                    param.data = torch.empty_like(param.data, device='meta')  # Meta device for non-owned
        # Shared experts remain on all GPUs

# Wrap with FSDP: Shard params, use activation checkpointing
model = FSDP(
    model,
    auto_wrap_policy=size_based_auto_wrap_policy(min_num_params=1e7),  # Shard large linears
    cpu_offload=CPUOffload(offload_params=True),  # Offload to CPU (use host RAM)
    use_orig_params=True,
    limit_all_gathers=True,
    sync_module_states=True,
    forward_prefetch=True
)
model.train()

# Optimizer
optimizer = AdamW(model.parameters(), lr=3e-5, betas=(0.9, 0.95), weight_decay=0.1)

# Data: Load FineWeb sharded across ranks
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2", trust_remote_code=True)  # Use actual tokenizer
dataset = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
dataset = dataset.shuffle(buffer_size=1000000, seed=42)

def preprocess(examples):
    tokenized = tokenizer(examples['text'], truncation=True, max_length=4096)  # Adjust seq len
    return tokenized

dataset = dataset.map(preprocess, batched=True, remove_columns=['text'])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Distributed sampler
sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,  # Micro-batch; global ~2*304=608, accum for larger
    sampler=sampler,
    collate_fn=data_collator,
    num_workers=8, pin_memory=True
)

# Training loop
num_epochs = 1  # Single pass over dataset; adjust for tokens
accumulation_steps = 16  # For effective batch ~10k
scaler = torch.amp.GradScaler('cuda')  # For mixed precision

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    total_loss = 0.0
    for step, inputs in enumerate(dataloader):
        input_ids = inputs['input_ids'].to(device=local_rank, dtype=torch.long)
        labels = inputs['labels'].to(device=local_rank)
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            # Activation checkpointing for memory
            def fwd_fn(x):
                return model(x, labels=labels, use_mtp=True)
            logits, loss = checkpoint(fwd_fn, input_ids)
        
        scaler.scale(loss / accumulation_steps).backward()
        total_loss += loss.item()
        
        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        if rank == 0 and step % 100 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {total_loss / (step+1)}")
        
        # Save checkpoint (use FSDP state_dict)
        if step % 10000 == 0:
            dist.barrier()
            if rank == 0:
                torch.save(model.state_dict(), f"checkpoint_{epoch}_{step}.pt")
    
    dist.barrier()  # Sync across nodes

# Cleanup
dist.destroy_process_group()