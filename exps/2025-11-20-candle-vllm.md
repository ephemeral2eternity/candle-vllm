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

## Notes

This benchmark was conducted using the candle-vllm server with the DeepSeek-R1-0528-Qwen3-8B model running on 2 devices. The benchmark script sent 16 concurrent requests, each requesting up to 1024 tokens.

