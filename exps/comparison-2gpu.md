# Performance Comparison: candle-vllm vs python vLLM (2 GPUs)

## Summary

Comparing candle-vllm and python vLLM performance with 2 GPUs using the DeepSeek-R1-0528-Qwen3-8B model.

## Test Configuration

Both experiments used:
- **Model**: deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
- **Devices**: 2 GPUs (tensor parallelism)
- **Batch size**: 16 concurrent requests
- **Max tokens per request**: 1024
- **Benchmark script**: `python3 examples/benchmark.py --batch 16 --max_tokens 1024 --port 2000`

## Results Comparison

### Latency Metrics

| Metric | candle-vllm (2 GPUs) | python vLLM (2 GPUs) | Ratio (vLLM/candle) |
|--------|---------------------|---------------------|---------------------|
| **TTFT Mean** | 74.563s | 40.273s | **1.85x faster** |
| **TTFT Median** | 79.362s | 42.851s | **1.85x faster** |
| **TTFT Min** | 1.602s | 0.884s | 1.81x faster |
| **TTFT Max** | 80.339s | 43.564s | 1.84x faster |
| **TTFT Std Dev** | 19.465s | 10.512s | More consistent |
| **Total Time Mean** | 83.462s | 42.946s | **1.94x faster** |
| **Total Time Median** | 79.490s | 42.946s | **1.85x faster** |
| **Total Time Min** | 78.524s | 42.229s | 1.86x faster |
| **Total Time Max** | 143.992s | 43.659s | **3.30x faster** |
| **Total Time Std Dev** | 16.151s | 0.453s | **Much more consistent** |

### Throughput Metrics

| Metric | candle-vllm (2 GPUs) | python vLLM (2 GPUs) | Ratio (vLLM/candle) |
|--------|---------------------|---------------------|---------------------|
| **Overall Throughput** | 97.49 tokens/s | 342.80 tokens/s | **3.51x faster** |
| **Per-request Mean** | 10.72 tokens/s | 21.78 tokens/s | **2.03x faster** |
| **Per-request Median** | 10.82 tokens/s | 21.63 tokens/s | **2.00x faster** |
| **Per-request Min** | 6.34 tokens/s | 20.22 tokens/s | 3.19x faster |
| **Per-request Max** | 11.75 tokens/s | 24.43 tokens/s | 2.08x faster |
| **Total Tokens Generated** | 14,038 | 14,966 | Similar |
| **Total Benchmark Time** | 143.992s | 43.659s | **3.30x faster** |

## Key Observations

1. **Throughput**: python vLLM achieves **3.51x higher throughput** (342.80 vs 97.49 tokens/s)
2. **Latency**: python vLLM is **~1.85-1.94x faster** in time-to-first-token and total time
3. **Consistency**: python vLLM shows much more consistent performance (std dev 0.453s vs 16.151s for total time)
4. **Suspicion**: The performance gap suggests candle-vllm may not be effectively utilizing both GPUs, as the speedup is less than expected for 2 GPUs

## Hypothesis

The significant performance gap (especially the 3.51x throughput difference) suggests that:
- candle-vllm might not be properly distributing the workload across 2 GPUs
- There may be a configuration issue with multi-GPU setup in candle-vllm
- The multi-threaded vs multi-process mode might not be working as expected

## Next Steps

To verify if candle-vllm is properly using both GPUs, we should:
1. Run a comparison with **1 GPU** for both systems
2. If 1-GPU performance is similar, it confirms that candle-vllm's 2-GPU setup is not working correctly
3. Check GPU utilization during inference (using `nvidia-smi`)

