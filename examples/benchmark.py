import openai
import asyncio
from openai import Stream
from openai.types.chat import ChatCompletionChunk
from typing import List, Tuple, Optional
import argparse
import time
import statistics
import requests
# Run candle-vllm service: cargo run --release -- --port 2000 --model-id <MODEL_ID> <MODEL_TYPE> --repeat-last-n 64
# MODEL_ID is the huggingface model id or local weight path
# MODEL_TYPE is one of ["llama", "llama3", "mistral", "phi2", "phi3", "qwen2", "qwen3", "gemma", "yi", "stable-lm"]
# Then run this file: python3 examples/benchmark.py --batch 16

openai.api_key = "EMPTY"

openai.base_url = "http://localhost:2000/v1/"

# You may add your custom prompts here
PROMPT_CANDIDATES = ["Explain how to best learn Rust.", 
            "Please talk about deep learning.", 
            "Do you know the capital city of China? Talk the details of you known.", 
            "Who is the best female actor in the world? Explain why.",
            "Let me know how to deal with depression?",
            "How to make money in short time?",
            "What is the future trend of large language model?",
            "The famous tech companies in the world."]

async def chat_completion(model, max_tokens, prompt):
    completion = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens = max_tokens,
        stream=True,
    )
    return completion

async def stream_response(response_idx, stream: Stream[ChatCompletionChunk], request_start_time: float):
    result = ""
    first_token_time = None
    token_count = 0
    
    for o in stream:
        r = o.choices[0].delta.content
        if r != None:
            if first_token_time is None:
                first_token_time = time.time()
            result += r
            # Approximate token count (rough estimate: ~4 chars per token)
            token_count += len(r) // 4
    
    completion_time = time.time()
    time_to_first_token = (first_token_time - request_start_time) if first_token_time else None
    total_time = completion_time - request_start_time
    
    return (response_idx, result, time_to_first_token, total_time, token_count)

def get_model_from_server(port: int) -> Optional[str]:
    """Try to get the model name from the server's /v1/models endpoint."""
    try:
        response = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "data" in data and len(data["data"]) > 0:
                return data["data"][0]["id"]
    except Exception as e:
        print(f"Warning: Could not auto-detect model from server: {e}")
    return None

async def benchmark(batch, max_tokens=1024, port=2000, model: Optional[str] = None):
    openai.base_url = "http://localhost:"+str(port)+"/v1/"

    # Auto-detect model if not provided
    if model is None:
        # Run the synchronous request in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        detected_model = await loop.run_in_executor(None, get_model_from_server, port)
        if detected_model:
            model = detected_model
            print(f"Auto-detected model: {model}")
        else:
            model = "any"  # Fallback for candle-vllm compatibility
            print("Using default model 'any' (candle-vllm compatibility mode)")
    else:
        print(f"Using specified model: {model}")
    # candidate requests
    prompts = []
    for i in range(batch):
        prompts.append(PROMPT_CANDIDATES[i % len(PROMPT_CANDIDATES)])

    # avoid generating very short answers
    for i in range(len(prompts)):
        prompts[i] = prompts[i] + " Respond in more than {} words.".format(int(max_tokens / 10) * 10)

    # Record overall benchmark start time
    benchmark_start_time = time.time()
    
    # send batch chat requests at the same time
    request_start_times = []
    tasks: List[asyncio.Task] = []
    for i in range(len(prompts)):
        request_start_time = time.time()
        request_start_times.append(request_start_time)
        tasks.append(
            asyncio.create_task(
                chat_completion(model, max_tokens, prompts[i]))
        )

    # obtain the correspond stream object for each request
    outputs: List[Stream[ChatCompletionChunk]] = await asyncio.gather(*tasks)

    # tasks for streaming chat responses
    tasks_stream: List[asyncio.Task] = []
    for i in range(len(outputs)):
        tasks_stream.append(
            asyncio.create_task(
                stream_response(i, outputs[i], request_start_times[i]))
        )

    # gathering the response texts and metrics
    results: List[Tuple[int, str, float, float, int]] = await asyncio.gather(*tasks_stream)
    
    benchmark_end_time = time.time()
    total_benchmark_time = benchmark_end_time - benchmark_start_time

    # Calculate statistics
    ttfts = [r[2] for r in results if r[2] is not None]
    total_times = [r[3] for r in results]
    token_counts = [r[4] for r in results]
    
    total_tokens = sum(token_counts)
    throughput = total_tokens / total_benchmark_time if total_benchmark_time > 0 else 0
    
    # Print summary statistics
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"Batch size: {batch}")
    print(f"Max tokens per request: {max_tokens}")
    print(f"Total requests: {len(results)}")
    print(f"\n--- Latency Metrics ---")
    if ttfts:
        print(f"Time to First Token (TTFT):")
        print(f"  Mean: {statistics.mean(ttfts):.3f}s")
        print(f"  Median: {statistics.median(ttfts):.3f}s")
        print(f"  Min: {min(ttfts):.3f}s")
        print(f"  Max: {max(ttfts):.3f}s")
        if len(ttfts) > 1:
            print(f"  Std Dev: {statistics.stdev(ttfts):.3f}s")
    
    print(f"\nTotal Time per Request:")
    print(f"  Mean: {statistics.mean(total_times):.3f}s")
    print(f"  Median: {statistics.median(total_times):.3f}s")
    print(f"  Min: {min(total_times):.3f}s")
    print(f"  Max: {max(total_times):.3f}s")
    if len(total_times) > 1:
        print(f"  Std Dev: {statistics.stdev(total_times):.3f}s")
    
    print(f"\n--- Throughput Metrics ---")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total benchmark time: {total_benchmark_time:.3f}s")
    print(f"Overall throughput: {throughput:.2f} tokens/second")
    
    per_request_throughputs = [tokens / time if time > 0 else 0 for tokens, time in zip(token_counts, total_times)]
    if per_request_throughputs:
        print(f"Per-request throughput:")
        print(f"  Mean: {statistics.mean(per_request_throughputs):.2f} tokens/second")
        print(f"  Median: {statistics.median(per_request_throughputs):.2f} tokens/second")
        print(f"  Min: {min(per_request_throughputs):.2f} tokens/second")
        print(f"  Max: {max(per_request_throughputs):.2f} tokens/second")
    
    print("="*80)
    
    # Optionally print individual responses (commented out for cleaner output)
    # Uncomment the following lines if you want to see individual responses:
    # print("\n--- Individual Responses ---")
    # for idx, output, ttft, total_time, token_count in results:
    #     print(f"\n\n Response {idx}:")
    #     print(f"  TTFT: {ttft:.3f}s" if ttft else "  TTFT: N/A")
    #     print(f"  Total time: {total_time:.3f}s")
    #     print(f"  Tokens: {token_count}")
    #     print(f"  Content: {output[:200]}..." if len(output) > 200 else f"  Content: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Using 'batch' and 'max_tokens' parameters for candle-vllm benchmark.")
    parser.add_argument('--batch', default=16, type=int)
    parser.add_argument('--max_tokens', default=1024, type=int)
    parser.add_argument('--port', default=2000, type=int)
    parser.add_argument('--model', default=None, type=str, help='Model name to use (auto-detected if not specified)')
    args = parser.parse_args()
    asyncio.run(benchmark(args.batch, args.max_tokens, args.port, args.model))