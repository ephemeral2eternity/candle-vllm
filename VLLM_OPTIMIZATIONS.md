# Candle-vLLM Optimizations Tutorial

This document provides a comprehensive overview of the vLLM-style optimizations implemented in candle-vllm, a high-performance LLM inference engine written in Rust.

## Table of Contents

1. [Continuous Batching](#continuous-batching)
2. [Paged Attention](#paged-attention)
3. [Flash Attention](#flash-attention)
4. [CUDA Graph Compilation](#cuda-graph-compilation)
5. [Chunked Prefilling](#chunked-prefilling)
6. [Quantization Support](#quantization-support)
7. [FP8 KV Cache](#fp8-kv-cache)
8. [Block-Based KV Cache Management](#block-based-kv-cache-management)
9. [Multi-GPU and Multi-Node Support](#multi-gpu-and-multi-node-support)

---

## Continuous Batching

**Status: ✅ Implemented**

Continuous batching (also known as batched decoding) is a key optimization that allows the system to efficiently handle multiple requests arriving at different times by dynamically batching them together.

### Implementation Details

- **Location**: `src/scheduler/mod.rs`, `src/openai/pipelines/llm_engine.rs`
- **Key Components**:
  - `Scheduler` maintains three queues: `waiting`, `running`, and `swapped_out`
  - Sequences are dynamically added to batches as they arrive
  - The scheduler automatically manages sequence groups and batches them for efficient GPU utilization

### How It Works

1. **Request Arrival**: New requests are added to the `waiting` queue
2. **Scheduling**: The scheduler periodically checks the waiting queue and moves sequences to `running` when resources are available
3. **Dynamic Batching**: Multiple sequences in the `running` queue are batched together for a single forward pass
4. **Completion Handling**: Finished sequences are removed, and new ones can be added without waiting for the entire batch to complete

### Code Reference

```rust
// src/scheduler/mod.rs
pub struct Scheduler {
    waiting: VecDeque<Arc<SequenceGroup>>,
    running: VecDeque<Arc<SequenceGroup>>,
    swapped_out: VecDeque<Arc<SequenceGroup>>,
    // ...
}

pub fn schedule(&mut self) -> SchedulerOutput {
    // Dynamically batches sequences from waiting queue
    // Returns batched sequences ready for inference
}
```

---

## Paged Attention

**Status: ✅ Implemented**

Paged Attention is a memory-efficient attention mechanism that stores KV cache in non-contiguous memory blocks, similar to virtual memory paging in operating systems.

### Implementation Details

- **Location**: `src/paged_attention/mod.rs`, `src/backend/paged_attention.rs`, `kernels/src/pagedattention.cu`
- **Key Features**:
  - Block-based KV cache storage (default block size: 16 tokens)
  - Efficient memory allocation and deallocation
  - Support for variable-length sequences without padding
  - CUDA and Metal kernel implementations

### How It Works

1. **Block Allocation**: KV cache is divided into fixed-size blocks (typically 16 tokens)
2. **Block Tables**: Each sequence maintains a block table mapping logical blocks to physical memory blocks
3. **Attention Computation**: The paged attention kernel efficiently accesses non-contiguous blocks during attention computation
4. **Memory Efficiency**: Only allocates memory for actual sequence length, eliminating padding waste

### Code Reference

```rust
// src/paged_attention/mod.rs
pub struct PagedAttention {
    num_attention_heads: usize,
    head_dim: usize,
    num_key_value_heads: usize,
    scale: f32,
    // ...
}

// CUDA kernel: kernels/src/pagedattention.cu
// Metal kernel: metal-kernels/src/pagedattention.metal
```

### Benefits

- **Memory Efficiency**: Reduces memory fragmentation and waste
- **Flexible Batching**: Handles sequences of different lengths efficiently
- **Better Throughput**: Enables larger batch sizes with the same GPU memory

---

## Flash Attention

**Status: ✅ Implemented (Optional Feature)**

Flash Attention is an optimized attention implementation that reduces memory usage and improves performance for long sequences by computing attention in chunks.

### Implementation Details

- **Location**: `src/paged_attention/mod.rs` (lines 96-135)
- **Feature Flag**: `flash-attn` (requires CUDA_ARCH >= 800)
- **Usage**: Automatically used for prefill phases when enabled

### How It Works

1. **Conditional Usage**: Flash Attention is used during prefill (prompt processing) when the feature is enabled
2. **Fallback**: Falls back to chunked attention or paged attention when Flash Attention is not available
3. **Sliding Window Support**: Supports sliding window attention for models like Mistral

### Code Reference

```rust
// src/paged_attention/mod.rs
#[cfg(feature = "flash-attn")]
let att = if input_metadata.is_prompt {
    // Uses flash_attn_softcap or flash_attn_windowed_softcap
    candle_flash_attn::flash_attn_softcap(
        &q, &k, &v,
        self.scale as f32,
        Some(softcapping.unwrap_or(0.0f64) as f32),
        true,
    )?
} else {
    None
};
```

### Build Configuration

```bash
# Build with Flash Attention support
cargo build --release --features cuda,nccl,graph,flash-attn
```

---

## CUDA Graph Compilation

**Status: ✅ Implemented (Optional Feature)**

CUDA Graph compilation captures and optimizes CUDA kernel execution patterns, reducing CPU overhead and improving inference latency.

### Implementation Details

- **Location**: `src/backend/graph.rs`
- **Feature Flag**: `graph`
- **Capture Strategy**: Captures graphs for multiple batch sizes (1-15, then 16, 32, 48, 64)

### How It Works

1. **Graph Capture**: During warmup, captures CUDA execution graphs for different batch sizes
2. **Graph Replay**: During inference, replays the pre-captured graph instead of launching individual kernels
3. **Batch Size Matching**: Selects the smallest captured graph that fits the current batch size

### Code Reference

```rust
// src/backend/graph.rs
pub struct GraphCapturer<M: CudaGraphModule> {
    pub model: M,
    pub graph_bs: Vec<usize>,  // Batch sizes to capture: [1,2,...,15,16,32,48,64]
    // ...
}

pub fn capture(&mut self, device: &Device, kv_caches: Option<&Vec<(Tensor, Tensor)>>) -> Result<()> {
    // Captures CUDA graphs for each batch size
}
```

### Benefits

- **Reduced CPU Overhead**: Eliminates repeated kernel launch overhead
- **Lower Latency**: Faster inference for decode phases
- **Better GPU Utilization**: More efficient GPU execution

### Build Configuration

```bash
# Build with CUDA Graph support
cargo build --release --features cuda,nccl,graph
```

---

## Chunked Prefilling

**Status: ✅ Implemented**

Chunked prefilling breaks long prompts into smaller chunks, allowing the system to interleave prefill and decode phases more efficiently.

### Implementation Details

- **Location**: `src/openai/pipelines/llm_engine.rs`, `src/scheduler/mod.rs`
- **Default Chunk Size**: 8192 tokens (8K)
- **Configurable**: Via `--prefill-chunk-size` flag

### How It Works

1. **Chunk Processing**: Long prompts are processed in chunks (default 8K tokens)
2. **Progressive Caching**: Each chunk's KV cache is computed and stored incrementally
3. **Interleaving**: Allows decode requests to be processed between prefill chunks
4. **Completion Detection**: System tracks which sequences have completed prefilling

### Code Reference

```rust
// src/openai/pipelines/llm_engine.rs
const PREFILL_CHUNK_SIZE: usize = 8192;

fn prepare_prompt(&self, groups: &VecDeque<Arc<SequenceGroup>>, device: &Device) -> Result<PreparedInputs> {
    let chunk_size = self.prefill_chunk_size.unwrap_or(PREFILL_CHUNK_SIZE);
    let num_cached_tokens = seq.deref().get_num_cached_tokens();
    let num_tokens = if chunk_size > 0 {
        std::cmp::min(chunk_size, seq_len - num_cached_tokens)
    } else {
        seq_len - num_cached_tokens
    };
    // Processes only the current chunk
}
```

### Benefits

- **Better Latency**: Decode requests don't wait for long prefills to complete
- **Improved Throughput**: More efficient GPU utilization
- **Memory Management**: Better control over memory usage during prefill

### Usage

```bash
# Set custom chunk size (must be divisible by 1024)
candle-vllm --prefill-chunk-size 4096 --w /path/to/model

# Disable chunked prefilling
candle-vllm --prefill-chunk-size 0 --w /path/to/model
```

---

## Quantization Support

**Status: ✅ Implemented**

Candle-vLLM supports multiple quantization formats to reduce memory usage and accelerate inference.

### Supported Formats

1. **GGUF/GGML Formats**: `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q2k`, `q3k`, `q4k`, `q5k`, `q6k`
2. **Marlin Format**: 4-bit GPTQ/AWQ with optimized CUDA kernels
3. **GPTQ**: 4-bit quantization (128-group, desc_act=False)
4. **AWQ**: 4-bit quantization (after conversion to Marlin format)

### In-Situ Quantization

**Status: ✅ Implemented**

Models can be quantized during loading without pre-conversion:

```bash
# Quantize during loading
candle-vllm --w /path/to/model --isq q4k
```

### Marlin Kernel

**Status: ✅ Implemented**

Marlin provides highly optimized CUDA kernels for 4-bit quantized models:

- **Location**: `kernels/src/marlin/`, `src/openai/models/linear.rs`
- **Requirements**: 4-bit GPTQ/AWQ with specific config (sym=True, groupsize=128, desc_act=False)
- **Performance**: Significantly faster than standard quantized inference

### Code Reference

```rust
// src/openai/models/linear.rs
pub fn qlinear(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
    shards: Shard,
    quant_config: &Option<QuantConfig>,
    // ...
) -> Result<Linear> {
    // Supports Marlin, GPTQ, AWQ formats
}
```

### Conversion Tools

- **GPTQ to Marlin**: `examples/convert_marlin.py`
- **AWQ to Marlin**: `examples/convert_awq_marlin.py`

---

## FP8 KV Cache

**Status: ✅ Implemented (Optional)**

FP8 (8-bit floating point) KV cache reduces memory usage for KV cache storage while maintaining reasonable accuracy.

### Implementation Details

- **Location**: `src/openai/models/layers/attention.rs`, `src/openai/pipelines/pipeline.rs`
- **Activation**: Via `--fp8-kvcache` flag
- **Storage**: Uses `DType::U8` for KV cache when enabled

### How It Works

1. **Memory Reduction**: KV cache stored in FP8 format (8 bits vs 16/32 bits)
2. **Transparent Usage**: Attention layers automatically handle FP8 conversion
3. **Compatibility**: Works with all attention implementations (paged, flash, naive)

### Code Reference

```rust
// src/main.rs
let kv_cache_dtype = if args.fp8_kvcache { DType::U8 } else { dtype };

// src/openai/models/layers/attention.rs
if cfg.fp8_kvcache.unwrap_or(false) {
    // Uses FP8 KV cache
}
```

### Benefits

- **Memory Savings**: ~50% reduction in KV cache memory (vs FP16/BF16)
- **Larger Batches**: Enables larger batch sizes with the same GPU memory
- **Minimal Accuracy Loss**: FP8 provides good accuracy for cache storage

### Usage

```bash
candle-vllm --fp8-kvcache --w /path/to/model
```

---

## Block-Based KV Cache Management

**Status: ✅ Implemented**

Efficient block-based memory management for KV cache with support for swapping and copy-on-write operations.

### Implementation Details

- **Location**: `src/scheduler/block_engine.rs`, `src/scheduler/cache_engine.rs`
- **Key Features**:
  - Block allocation and deallocation
  - CPU-GPU swapping for memory management
  - Copy-on-write (COW) for shared blocks
  - Reference counting for block sharing

### How It Works

1. **Block Allocation**: KV cache divided into fixed-size blocks (typically 16 tokens)
2. **Block Tables**: Each sequence maintains a table mapping logical to physical blocks
3. **Swapping**: Blocks can be swapped to CPU when GPU memory is limited
4. **COW**: Shared blocks are copied when modified to maintain correctness

### Code Reference

```rust
// src/scheduler/block_engine.rs
pub struct BlockEngine {
    num_gpu_blocks: usize,
    gpu_allocator: Allocator<GPUAllocator>,
    cpu_allocator: Allocator<CPUAllocator>,
    pub block_tables: HashMap<SeqID, BlockTable>,
    block_size: usize,
}

pub fn swap_out(&mut self, seq_group: &SequenceGroup) -> HashMap<usize, usize> {
    // Swaps GPU blocks to CPU
}

pub fn swap_in(&mut self, seq_group: &SequenceGroup) -> HashMap<usize, usize> {
    // Swaps CPU blocks back to GPU
}
```

### Benefits

- **Memory Efficiency**: Only allocates what's needed
- **Flexible Scheduling**: Enables preemption and swapping
- **Better Utilization**: Maximizes GPU memory usage

---

## Multi-GPU and Multi-Node Support

**Status: ✅ Implemented**

Candle-vLLM supports distributed inference across multiple GPUs and nodes.

### Multi-GPU Support

**Modes**:
1. **Multi-Process Mode** (Default): Each GPU runs in a separate process
2. **Multi-Threaded Mode**: All GPUs in a single process (for debugging)

**Implementation**:
- **Location**: `src/openai/distributed.rs`, `src/openai/communicator.rs`
- **Communication**: NCCL for GPU communication
- **Tensor Parallelism**: Model weights sharded across GPUs

### Multi-Node Support

**Status: ✅ Implemented (via MPI)**

- **Location**: `src/openai/distributed.rs`
- **Requirement**: MPI (Message Passing Interface)
- **Communication**: MPI + NCCL for inter-node communication

### Code Reference

```rust
// src/openai/distributed.rs
pub struct DistributedModel {
    // Handles multi-GPU/multi-node coordination
}

// Multi-process coordination via NCCL
#[cfg(feature = "nccl")]
pub struct DaemonManager {
    // Manages communication between processes
}
```

### Build Configuration

```bash
# Multi-GPU (single node)
cargo build --release --features cuda,nccl

# Multi-node (with MPI)
sudo apt install libopenmpi-dev openmpi-bin -y
cargo build --release --features cuda,nccl,mpi
```

### Usage

```bash
# Multi-GPU (2 GPUs)
candle-vllm --d 0,1 --w /path/to/model

# Multi-node (via MPI)
mpirun -np 16 -hostfile ./hostfile candle-vllm --d 0,1,2,3,4,5,6,7 --w /path/to/model
```

---

## Summary

Candle-vLLM implements a comprehensive set of vLLM-style optimizations:

| Optimization | Status | Feature Flag | Notes |
|-------------|--------|--------------|-------|
| Continuous Batching | ✅ | Built-in | Core scheduling mechanism |
| Paged Attention | ✅ | Built-in | CUDA & Metal kernels |
| Flash Attention | ✅ | `flash-attn` | Requires CUDA_ARCH >= 800 |
| CUDA Graph | ✅ | `graph` | Reduces CPU overhead |
| Chunked Prefill | ✅ | Built-in | Default 8K chunks |
| Quantization | ✅ | Built-in | GGUF, Marlin, GPTQ, AWQ |
| FP8 KV Cache | ✅ | Built-in | Via `--fp8-kvcache` flag |
| Block Management | ✅ | Built-in | Core memory system |
| Multi-GPU | ✅ | `nccl` | Tensor parallelism |
| Multi-Node | ✅ | `mpi` | Distributed inference |

### Performance Characteristics

Based on the README benchmarks:
- **Single GPU (BF16)**: 65-107 tokens/s (depending on model)
- **Batched (BF16, bs=16)**: 553-831 tokens/s
- **Quantized (Q4K/Marlin)**: 75-115 tokens/s single, 587-968 tokens/s batched

### Key Design Principles

1. **Memory Efficiency**: Paged attention, block management, FP8 cache
2. **Throughput Optimization**: Continuous batching, chunked prefill, CUDA graphs
3. **Flexibility**: Multiple quantization formats, multi-GPU support
4. **Performance**: Optimized kernels (Marlin, Flash Attention) for critical paths

---

## References

- **Original vLLM Paper**: [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- **vLLM Project**: [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
- **Candle-vLLM Repository**: [https://github.com/EricLBuehler/candle-vllm](https://github.com/EricLBuehler/candle-vllm)

---

*Last Updated: Based on codebase analysis of candle-vllm*

