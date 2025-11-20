# Benchmarking Results - November 20, 2025

## Server Setup

### Command
```bash
target/release/candle-vllm --p 2000 --d 2 --m deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
```

### Configuration
- Port: 2000
- Devices: 2
- Model: deepseek-ai/DeepSeek-R1-0528-Qwen3-8B

## vLLM Server Setup

### Command
```bash
CUDA_VISIBLE_DEVICES=2,3 vllm serve deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --port 2000 \
    --host 0.0.0.0 \
    --tensor-parallel-size 2 \
    --dtype bfloat16
```

### Configuration
- CUDA devices: 2, 3
- Port: 2000
- Host: 0.0.0.0
- Tensor parallel size: 2
- Data type: bfloat16
- Model: deepseek-ai/DeepSeek-R1-0528-Qwen3-8B

## Benchmarking Script

### Command
```bash
python3 examples/benchmark.py --batch 16 --max_tokens 1024 --port 2000
```

### Configuration
- Batch size: 16
- Max tokens per request: 1024
- Port: 2000

## Results Summary

### Test Configuration
- **Batch size**: 16
- **Max tokens per request**: 1024
- **Total requests**: 16

### Latency Metrics

#### Time to First Token (TTFT)
- **Mean**: 74.563s
- **Median**: 79.362s
- **Min**: 1.602s
- **Max**: 80.339s
- **Std Dev**: 19.465s

#### Total Time per Request
- **Mean**: 83.462s
- **Median**: 79.490s
- **Min**: 78.524s
- **Max**: 143.992s
- **Std Dev**: 16.151s

### Throughput Metrics

- **Total tokens generated**: 14,038
- **Total benchmark time**: 143.992s
- **Overall throughput**: 97.49 tokens/second

#### Per-request Throughput
- **Mean**: 10.72 tokens/second
- **Median**: 10.82 tokens/second
- **Min**: 6.34 tokens/second
- **Max**: 11.75 tokens/second

## vLLM Results

### Test Configuration
- **Batch size**: 16
- **Max tokens per request**: 1024
- **Total requests**: 16

### Latency Metrics

#### Time to First Token (TTFT)
- **Mean**: 40.273s
- **Median**: 42.851s
- **Min**: 0.884s
- **Max**: 43.564s
- **Std Dev**: 10.512s

#### Total Time per Request
- **Mean**: 42.946s
- **Median**: 42.946s
- **Min**: 42.229s
- **Max**: 43.659s
- **Std Dev**: 0.453s

### Throughput Metrics

- **Total tokens generated**: 14,966
- **Total benchmark time**: 43.659s
- **Overall throughput**: 342.80 tokens/second

#### Per-request Throughput
- **Mean**: 21.78 tokens/second
- **Median**: 21.63 tokens/second
- **Min**: 20.22 tokens/second
- **Max**: 24.43 tokens/second

## Notes

This benchmark was conducted comparing candle-vllm and vLLM servers, both using the DeepSeek-R1-0528-Qwen3-8B model with tensor parallelism across 2 devices. The benchmark script sent 16 concurrent requests, each requesting up to 1024 tokens.

- **candle-vllm**: Running with 2 devices on port 2000
- **vLLM**: Running on CUDA devices 2 and 3 with tensor-parallel-size 2, using bfloat16 precision, on port 2000

Both servers were benchmarked using the same configuration (batch size: 16, max tokens: 1024) to enable direct performance comparison.

