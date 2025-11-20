# Benchmarking Results - November 20, 2025 (1 GPU Comparison)

## Purpose

This experiment compares candle-vllm and python vLLM performance using **1 GPU** to evaluate their relative performance.

## candle-vllm Server Setup (1 GPU)

### Command
```bash
target/release/candle-vllm --p 2000 --d 2 --m Qwen/Qwen3-0.6B
```

### Configuration
- Port: 2000
- Devices: 1 (device 2)
- Model: Qwen/Qwen3-0.6B

## python vLLM Server Setup (1 GPU)

### Command
```bash
CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen3-0.6B \
    --port 2000 \
    --host 0.0.0.0 \
    --tensor-parallel-size 1 \
    --dtype bfloat16
```

### Configuration
- CUDA device: 2
- Port: 2000
- Host: 0.0.0.0
- Tensor parallel size: 1
- Data type: bfloat16
- Model: Qwen/Qwen3-0.6B

## Benchmarking Script

### Command
```bash
python3 examples/benchmark.py --batch 16 --max_tokens 1024 --port 2000
```

### Configuration
- Batch size: 16
- Max tokens per request: 1024
- Port: 2000

## candle-vllm Results (1 GPU)

### Test Configuration
- **Batch size**: 16
- **Max tokens per request**: 1024
- **Total requests**: 16

### Latency Metrics

#### Time to First Token (TTFT)
- **Mean**: 28.361s
- **Median**: 30.063s
- **Min**: 1.607s
- **Max**: 31.018s
- **Std Dev**: 7.154s

#### Total Time per Request
- **Mean**: 30.537s
- **Median**: 30.182s
- **Min**: 29.325s
- **Max**: 36.426s
- **Std Dev**: 1.655s

### Throughput Metrics

- **Total tokens generated**: 15906
- **Total benchmark time**: 36.426s
- **Overall throughput**: 436.67 tokens/second

#### Per-request Throughput
- **Mean**: 32.57 tokens/second
- **Median**: 32.50 tokens/second
- **Min**: 23.22 tokens/second
- **Max**: 38.46 tokens/second

## python vLLM Results (1 GPU)

### Test Configuration
- **Batch size**: 16
- **Max tokens per request**: 1024
- **Total requests**: 16

### Latency Metrics

#### Time to First Token (TTFT)
- **Mean**: 13.501s
- **Median**: 14.278s
- **Min**: 0.902s
- **Max**: 15.225s
- **Std Dev**: 3.404s

#### Total Time per Request
- **Mean**: 14.404s
- **Median**: 14.403s
- **Min**: 13.459s
- **Max**: 15.353s
- **Std Dev**: 0.600s

### Throughput Metrics

- **Total tokens generated**: 16026
- **Total benchmark time**: 15.353s
- **Overall throughput**: 1043.81 tokens/second

#### Per-request Throughput
- **Mean**: 69.60 tokens/second
- **Median**: 70.85 tokens/second
- **Min**: 64.38 tokens/second
- **Max**: 75.23 tokens/second

## Comparison Summary

| Metric | candle-vllm (1 GPU) | python vLLM (1 GPU) | Ratio (vLLM/candle) |
|--------|---------------------|---------------------|---------------------|
| **Overall Throughput** | 436.67 tokens/s | 1043.81 tokens/s | **2.39x faster** |
| **TTFT Mean** | 28.361s | 13.501s | **2.10x faster** |
| **Total Time Mean** | 30.537s | 14.404s | **2.12x faster** |
| **Per-request Throughput Mean** | 32.57 tokens/s | 69.60 tokens/s | **2.14x faster** |

## Analysis

### Performance Comparison

**Performance Winner: python vLLM is significantly faster**

| Metric | python vLLM Advantage |
|--------|----------------------|
| **Overall Throughput** | **2.39x faster** (1043.81 vs 436.67 tokens/s) |
| **TTFT** | **2.10x faster** (13.501s vs 28.361s) |
| **Total Latency** | **2.12x faster** (14.404s vs 30.537s) |
| **Per-request Throughput** | **2.14x faster** (69.60 vs 32.57 tokens/s) |

### Key Observations

1. **python vLLM consistently outperforms candle-vllm by ~2.1-2.4x** across all metrics
2. **Latency consistency**: python vLLM shows much lower standard deviation (0.600s vs 1.655s for total time), indicating more stable performance
3. **TTFT variance**: candle-vllm has high TTFT variance (std dev: 7.154s vs 3.404s), suggesting inconsistent prefill performance
4. **Throughput efficiency**: python vLLM generates slightly more tokens (16026 vs 15906) in less than half the time

### Root Cause Analysis

The performance gap between python vLLM and candle-vllm likely stems from:

1. **Optimized CUDA kernels**: python vLLM uses highly optimized FlashAttention and custom CUDA kernels
2. **Memory management**: Better memory pooling and allocation strategies
3. **Batch processing**: More efficient batch scheduling and prefilling
4. **Kernel fusion**: Reduced kernel launch overhead through fused operations
5. **Maturity**: python vLLM has years of optimization work from the community

### Recommendations

**For candle-vllm:**
1. **Profile performance**: Use CUDA profiling tools to identify bottlenecks in prefill and decode phases
2. **Optimize kernels**: Focus on improving attention mechanisms and matrix operations
3. **Improve memory management**: Implement better memory pooling and allocation strategies
4. **Reduce variance**: Investigate and fix causes of high TTFT variance
5. **Compare kernel implementations**: Analyze differences in attention mechanisms and matrix operations

**For future experiments:**
- Test with larger models to see if performance characteristics scale differently
- Monitor GPU utilization during inference to verify proper device usage
- Compare memory usage patterns between implementations

## Notes

- Both servers were tested on the same GPU hardware
- Monitor GPU utilization during inference using `nvidia-smi dmon -s u -c 1000`
- Ensure CUDA_VISIBLE_DEVICES is set correctly for python vLLM

