#!/usr/bin/env python3
"""
Qwen3Guard ONNXRuntime Batch Size Benchmark

This benchmark tests Qwen3Guard model via ONNXRuntime with different batch sizes
to measure performance when processing multiple samples in a single inference call.

Usage:
    python3 qwen3_guard_onnxruntime_batch_benchmark.py \
        --model-path /path/to/Qwen3Guard-Gen-0.6B \
        --dataset-path /path/to/Qwen3GuardTest \
        --split thinking \
        --max-samples 100 \
        --batch-sizes 1,2,4,8,16

The model path should be a HuggingFace standard format directory containing:
    - model.onnx (or similar ONNX model file)
    - tokenizer.json, tokenizer_config.json (for AutoTokenizer)
"""

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


@dataclass
class BenchmarkSample:
    text: str
    label: str  # "safe" or "unsafe"
    unsafe_type: Optional[str] = None


@dataclass
class BenchmarkResult:
    batch_size: int
    total_samples: int
    correct_predictions: int
    accuracy: float
    throughput: float  # samples per second
    avg_latency_per_sample: float  # milliseconds per sample
    avg_latency_per_batch: float  # milliseconds per batch
    min_latency_per_batch: float
    max_latency_per_batch: float
    safe_precision: float
    safe_recall: float
    unsafe_precision: float
    unsafe_recall: float
    failed_batches: Optional[list] = None  # List of (batch_idx, num_samples, error_type)
    failed_samples: int = 0
    successful_samples: int = 0


def load_dataset(
    dataset_path: str, split: str, max_samples: Optional[int] = None
) -> List[BenchmarkSample]:
    """Load dataset from JSONL file."""
    jsonl_path = Path(dataset_path) / f"{split}.jsonl"
    
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")
    
    samples = []
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break
            
            sample_data = json.loads(line.strip())
            
            # Extract user message and assistant response
            messages = sample_data.get("message", [])
            if len(messages) >= 2:
                user_msg = next(
                    (m["content"] for m in messages if m["role"] == "user"), ""
                )
                assistant_msg = next(
                    (m["content"] for m in messages if m["role"] == "assistant"), ""
                )
                text = f"{user_msg}\n{assistant_msg}"
            elif len(messages) > 0:
                text = messages[0].get("content", "")
            else:
                continue
            
            samples.append(
                BenchmarkSample(
                    text=text,
                    label=sample_data.get("label", "safe").lower(),
                    unsafe_type=sample_data.get("unsafe_type"),
                )
            )
    
    return samples


def parse_model_output(output: str) -> str:
    """Parse model output to extract the label."""
    # Parse model output to extract the label
    # Qwen3Guard typically outputs in format: <label>Label</label>
    if "<label>" in output:
        start = output.find("<label>")
        end = output.find("</label>", start)
        if end != -1:
            label = output[start + 7 : end].strip().lower()
            return label
    
    # Fallback: check if output contains "unsafe" or "safe"
    lower = output.lower()
    if "unsafe" in lower:
        return "unsafe"
    elif "safe" in lower:
        return "safe"
    else:
        return "unknown"


def calculate_metrics(
    samples: List[BenchmarkSample], predictions: List[str]
) -> Tuple[int, float, float, float, float]:
    """Calculate accuracy and precision/recall metrics."""
    correct = 0
    safe_tp = 0
    safe_fp = 0
    safe_fn = 0
    unsafe_tp = 0
    unsafe_fp = 0
    unsafe_fn = 0
    
    for sample, pred in zip(samples, predictions):
        if pred == sample.label:
            correct += 1
        
        if sample.label == "safe" and pred == "safe":
            safe_tp += 1
        elif sample.label == "safe" and pred == "unsafe":
            safe_fn += 1
        elif sample.label == "unsafe" and pred == "safe":
            unsafe_fn += 1
        elif sample.label == "unsafe" and pred == "unsafe":
            unsafe_tp += 1
        
        if pred == "safe" and sample.label != "safe":
            safe_fp += 1
        if pred == "unsafe" and sample.label != "unsafe":
            unsafe_fp += 1
    
    safe_prec = safe_tp / (safe_tp + safe_fp) if (safe_tp + safe_fp) > 0 else 0.0
    safe_rec = safe_tp / (safe_tp + safe_fn) if (safe_tp + safe_fn) > 0 else 0.0
    unsafe_prec = (
        unsafe_tp / (unsafe_tp + unsafe_fp) if (unsafe_tp + unsafe_fp) > 0 else 0.0
    )
    unsafe_rec = (
        unsafe_tp / (unsafe_tp + unsafe_fn) if (unsafe_tp + unsafe_fn) > 0 else 0.0
    )
    
    return correct, safe_prec, safe_rec, unsafe_prec, unsafe_rec


def find_onnx_model(model_path: str) -> str:
    """Find ONNX model file in the model directory, including subdirectories."""
    model_dir = Path(model_path)
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    # Common ONNX model file names
    possible_names = [
        "model.onnx",
        "decoder_model.onnx",
        "encoder_model.onnx",
        "model_optimized.onnx",
    ]
    
    # First, check for exact matches in root directory
    for name in possible_names:
        onnx_path = model_dir / name
        if onnx_path.exists():
            return str(onnx_path)
    
    # If not found, search for any .onnx file in root directory
    onnx_files = list(model_dir.glob("*.onnx"))
    if onnx_files:
        return str(onnx_files[0])
    
    # Search recursively in subdirectories (for HuggingFace cache structure)
    # HuggingFace cache has structure: models--Qwen--Qwen3Guard-Gen-0.6B/snapshots/<hash>/
    for name in possible_names:
        onnx_files = list(model_dir.rglob(name))
        if onnx_files:
            # Prefer files in snapshots/ subdirectories (HuggingFace cache)
            snapshots_files = [f for f in onnx_files if "snapshots" in str(f)]
            if snapshots_files:
                return str(snapshots_files[0])
            return str(onnx_files[0])
    
    # Search for any .onnx file recursively
    onnx_files = list(model_dir.rglob("*.onnx"))
    if onnx_files:
        # Prefer files in snapshots/ subdirectories (HuggingFace cache)
        snapshots_files = [f for f in onnx_files if "snapshots" in str(f)]
        if snapshots_files:
            return str(snapshots_files[0])
        return str(onnx_files[0])
    
    # If still not found, check if the path itself is an ONNX file
    if model_path.endswith(".onnx") and Path(model_path).exists():
        return model_path
    
    # Provide helpful error message
    raise FileNotFoundError(
        f"Could not find ONNX model file in {model_path}.\n"
        f"Searched for: {', '.join(possible_names)} and any *.onnx files (including subdirectories).\n"
        f"\n"
        f"The model appears to be in HuggingFace format (PyTorch), not ONNX format.\n"
        f"ONNX Runtime requires an ONNX model file.\n"
        f"\n"
        f"To convert your HuggingFace model to ONNX format, run:\n"
        f"  python3 convert_to_onnx.py \\\n"
        f"    --model-path {model_path} \\\n"
        f"    --output-path /path/to/output/model.onnx \\\n"
        f"    --device cpu\n"
        f"\n"
        f"Alternatively, if you already have an ONNX model file, specify its full path:\n"
        f"  --model-path /path/to/model.onnx"
    )


class ONNXRuntimeModel:
    """Wrapper for ONNXRuntime model inference with batching support."""
    
    def __init__(self, model_path: str, tokenizer_path: Optional[str] = None, device: str = "cpu"):
        """
        Initialize ONNXRuntime model and tokenizer.
        
        Args:
            model_path: Path to ONNX model file (.onnx) or HuggingFace model directory
            tokenizer_path: Optional path to tokenizer. If None, uses model_path
            device: 'cpu' or 'cuda'
        """
        # Find ONNX model file
        onnx_file = find_onnx_model(model_path)
        
        # Use model_path as tokenizer path if not specified
        if tokenizer_path is None:
            tokenizer_path = model_path
        
        # Load tokenizer
        print(f"   Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
        
        # Configure ONNXRuntime session with optimizations
        providers = []
        provider_options = []
        
        if device == "cuda":
            # CUDA provider with optimizations
            # gpu_mem_limit: Maximum GPU memory (in bytes) that ONNX Runtime can allocate.
            # Set this to leave some memory for other processes or to prevent OOM errors.
            # Typical values: 80-90% of total GPU memory (check with nvidia-smi)
            # For 0.6B model, even with small batches, reduce this if you have limited GPU memory
            try:
                try:
                    import torch
                    if torch.cuda.is_available():
                        # Get actual GPU memory and use 80% of it
                        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        gpu_mem_limit = int(gpu_memory_gb * 0.95 * 1024 * 1024 * 1024)
                        print(f"   Detected GPU memory: {gpu_memory_gb:.1f}GB, setting limit to {gpu_mem_limit/(1024**3):.1f}GB")
                    else:
                        gpu_mem_limit = 8 * 1024 * 1024 * 1024  # Default 8GB if can't detect
                except ImportError:
                    # PyTorch not available, use conservative default
                    gpu_mem_limit = 8 * 1024 * 1024 * 1024
                    print(f"   PyTorch not available, using default GPU memory limit: {gpu_mem_limit/(1024**3):.1f}GB")
            except Exception:
                # Fallback: use conservative 8GB limit for 0.6B model
                gpu_mem_limit = 8 * 1024 * 1024 * 1024
                print(f"   Using default GPU memory limit: {gpu_mem_limit/(1024**3):.1f}GB")
            
            cuda_provider_options = {
                "device_id": 0,
                "arena_extend_strategy": "kSameAsRequested",  # More memory efficient than kNextPowerOfTwo
                "gpu_mem_limit": gpu_mem_limit,
                "cudnn_conv_algo_search": "HEURISTIC",  # Find optimal algorithm (one-time cost during init)
                "do_copy_in_default_stream": True,
            }
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            provider_options = [cuda_provider_options, {}]
        else:
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]
        
        # Session options with performance optimizations
        sess_options = ort.SessionOptions()
        
        # Thread configuration
        # For CPU: use all available cores
        # For CUDA: fewer threads since GPU does the work
        if device == "cuda":
            sess_options.intra_op_num_threads = 1  # GPU handles parallelism
            sess_options.inter_op_num_threads = 1
        else:
            import os
            num_threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 4))
            sess_options.intra_op_num_threads = num_threads
            sess_options.inter_op_num_threads = num_threads
        
        # Enable graph optimizations
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Enable memory pattern optimization (reuses memory allocations)
        # This is available in most ONNX Runtime versions
        if hasattr(sess_options, 'enable_mem_pattern'):
            sess_options.enable_mem_pattern = True
        
        # Memory arena settings (varies by ONNX Runtime version)
        # In newer versions, these may be controlled via provider options
        if hasattr(sess_options, 'enable_mem_arena'):
            sess_options.enable_mem_arena = True
        
        # Enable CPU memory arena for CPU provider (if available)
        if device == "cpu" and hasattr(sess_options, 'enable_cpu_mem_arena'):
            sess_options.enable_cpu_mem_arena = True
        
        # Disable profiling for better performance
        sess_options.enable_profiling = False
        
        print(f"   Loading ONNX model from {onnx_file}...")
        print(f"   Using providers: {providers}")
        print(f"   Graph optimization: ENABLE_ALL")
        print(f"   Memory optimizations: enabled")
        
        self.session = ort.InferenceSession(
            onnx_file,
            sess_options=sess_options,
            providers=providers,
            provider_options=provider_options if provider_options else None,
        )
        
        # Get input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        print(f"   Model loaded successfully")
        print(f"   Input names: {self.input_names}")
        print(f"   Output names: {self.output_names}")
    
    def format_prompt(self, text: str) -> str:
        """Format text as a Qwen3Guard classification prompt."""
        return (
            "You are a safety classifier. Classify the following content as safe or unsafe.\n\n"
            "Example:\n"
            "Content: \"Hello, how are you?\"\n"
            "Classification: <label>safe</label>\n\n"
            "Now classify this content:\n"
            f"Content: {text}\n"
            "Classification:"
        )
    
    def generate_batch(self, texts: List[str], max_new_tokens: int = 128) -> List[str]:
        """
        Generate text for a batch of inputs using ONNXRuntime.
        
        Args:
            texts: List of input texts to process
            max_new_tokens: Maximum number of tokens to generate
            max_length: Maximum sequence length for tokenization [default: 4096]
        
        Returns:
            List of generated text outputs
        """
        # Format prompts
        formatted_prompts = [self.format_prompt(text) for text in texts]
        
        # Tokenize batch with padding
        # Use 4096 to match vLLM benchmark (README notes some prompts are longer than 2048 tokens)
        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=4096,
        )
        
        # Prepare inputs for ONNX model
        onnx_inputs = {}
        for name in self.input_names:
            if "input_ids" in name.lower():
                onnx_inputs[name] = inputs["input_ids"].astype(np.int64)
            elif "attention_mask" in name.lower():
                onnx_inputs[name] = inputs["attention_mask"].astype(np.int64)
            elif "position_ids" in name.lower():
                # Generate position_ids efficiently using broadcasting
                # position_ids are the cumulative positions for each token
                batch_size, seq_length = inputs["attention_mask"].shape
                # Use broadcasting instead of loop for better performance
                position_ids = np.arange(seq_length, dtype=np.int64)[None, :].repeat(batch_size, axis=0)
                onnx_inputs[name] = position_ids
            elif "token_type_ids" in name.lower() and "token_type_ids" in inputs:
                onnx_inputs[name] = inputs["token_type_ids"].astype(np.int64)
        
        # Verify all required inputs are provided
        missing_inputs = set(self.input_names) - set(onnx_inputs.keys())
        if missing_inputs:
            raise ValueError(
                f"Missing required inputs: {missing_inputs}. "
                f"Provided inputs: {list(onnx_inputs.keys())}. "
                f"Model expects: {self.input_names}"
            )
        
        # Run inference
        outputs = self.session.run(self.output_names, onnx_inputs)
        
        # Get logits (assuming first output is logits)
        logits = outputs[0]
        
        # Decode outputs for each sample in the batch
        generated_texts = []
        for i in range(len(texts)):
            # Simple greedy decoding: get the most likely token at each position
            generated_ids = np.argmax(logits[i], axis=-1)
            
            # Decode the generated tokens
            generated_text = self.tokenizer.decode(
                generated_ids.flatten()[:max_new_tokens], skip_special_tokens=True
            )
            
            # Extract the relevant part (after the prompt)
            if "Classification:" in generated_text:
                generated_text = generated_text.split("Classification:")[-1].strip()
            
            generated_texts.append(generated_text)
        
        return generated_texts


def benchmark_batch_size(
    model_path: str,
    tokenizer_path: Optional[str],
    samples: List[BenchmarkSample],
    batch_size: int,
    device: str = "cpu",
) -> BenchmarkResult:
    """Benchmark ONNXRuntime with a specific batch size."""
    print(f"   Initializing ONNXRuntime model (device: {device}, batch_size: {batch_size})...")
    
    # Initialize model
    model = ONNXRuntimeModel(model_path, tokenizer_path, device)
    
    # Process samples in batches
    predictions = []
    batch_latencies = []
    failed_batches = []
    successful_samples = 0
    failed_samples = 0
    
    overall_start = time.time()
    
    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i + batch_size]
        batch_texts = [s.text for s in batch_samples]
        batch_idx = i // batch_size
        
        # Measure batch inference time
        batch_start = time.time()
        try:
            # generate_batch uses max_length=4096 internally (matches vLLM benchmark)
            batch_outputs = model.generate_batch(batch_texts, max_new_tokens=128)
            batch_elapsed = (time.time() - batch_start) * 1000.0  # Convert to ms
            
            # Parse outputs
            for output in batch_outputs:
                pred = parse_model_output(output)
                predictions.append(pred)
            
            batch_latencies.append(batch_elapsed)
            successful_samples += len(batch_samples)
            
        except Exception as e:
            error_msg = str(e)
            # Check if it's an OOM error
            is_oom = "Available memory" in error_msg or "smaller than requested" in error_msg
            error_type = "OOM (Out of Memory)" if is_oom else "Error"
            
            print(f"   ⚠️  {error_type} processing batch {batch_idx}: {error_msg[:100]}...")
            
            failed_batches.append((batch_idx, len(batch_samples), error_type))
            failed_samples += len(batch_samples)
            
            # Don't add predictions for failed batches - they're invalid
            # We'll handle this in metrics calculation
            for _ in batch_samples:
                predictions.append(None)  # Mark as None to indicate failure
            batch_latencies.append(None)  # Mark as None
    
    total_time = time.time() - overall_start
    
    # Filter out failed batches for latency calculations
    successful_latencies = [l for l in batch_latencies if l is not None]
    
    # Calculate metrics only for successful samples
    successful_predictions = [p for p in predictions if p is not None]
    successful_sample_indices = [i for i, p in enumerate(predictions) if p is not None]
    successful_samples_list = [samples[i] for i in successful_sample_indices]
    
    if len(successful_predictions) == 0:
        print(f"   ❌ ERROR: All batches failed! Cannot compute meaningful metrics.")
        print(f"   Failed batches: {len(failed_batches)}")
        print(f"   This batch size ({batch_size}) is too large for available GPU memory.")
        return BenchmarkResult(
            batch_size=batch_size,
            total_samples=len(samples),
            correct_predictions=0,
            accuracy=0.0,
            throughput=0.0,
            avg_latency_per_sample=0.0,
            avg_latency_per_batch=0.0,
            min_latency_per_batch=0.0,
            max_latency_per_batch=0.0,
            safe_precision=0.0,
            safe_recall=0.0,
            unsafe_precision=0.0,
            unsafe_recall=0.0,
        )
    
    # Calculate throughput based on successful samples only
    throughput = len(successful_predictions) / total_time if total_time > 0 else 0.0
    
    # Calculate latency metrics from successful batches only
    avg_latency_per_batch = sum(successful_latencies) / len(successful_latencies) if successful_latencies else 0.0
    min_latency_per_batch = min(successful_latencies) if successful_latencies else 0.0
    max_latency_per_batch = max(successful_latencies) if successful_latencies else 0.0
    avg_latency_per_sample = avg_latency_per_batch / batch_size if batch_size > 0 and successful_latencies else 0.0
    
    # Calculate accuracy metrics only for successful samples
    correct, safe_prec, safe_rec, unsafe_prec, unsafe_rec = calculate_metrics(
        successful_samples_list, successful_predictions
    )
    accuracy = correct / len(successful_predictions) if successful_predictions else 0.0
    
    # Print summary
    print(f"   Processing complete:")
    print(f"      Successful: {successful_samples}/{len(samples)} samples ({successful_samples/len(samples)*100:.1f}%)")
    if failed_samples > 0:
        print(f"      Failed: {failed_samples}/{len(samples)} samples ({failed_samples/len(samples)*100:.1f}%)")
        print(f"      Failed batches: {len(failed_batches)}")
        print(f"      ⚠️  Metrics below are based on successful samples only!")
    
    result = BenchmarkResult(
        batch_size=batch_size,
        total_samples=len(samples),
        correct_predictions=correct,
        accuracy=accuracy,
        throughput=throughput,
        avg_latency_per_sample=avg_latency_per_sample,
        avg_latency_per_batch=avg_latency_per_batch,
        min_latency_per_batch=min_latency_per_batch,
        max_latency_per_batch=max_latency_per_batch,
        safe_precision=safe_prec,
        safe_recall=safe_rec,
        unsafe_precision=unsafe_prec,
        unsafe_recall=unsafe_rec,
    )
    
    # Store failure info for reporting
    if failed_batches:
        result.failed_batches = failed_batches
        result.failed_samples = failed_samples
        result.successful_samples = successful_samples
    
    return result


def print_result(title: str, result: BenchmarkResult):
    """Print formatted benchmark results."""
    print("\n" + "═" * 80)
    print(title)
    print("═" * 80)
    print("📊 Performance Metrics:")
    print(f"   Batch size:              {result.batch_size}")
    print(f"   Total samples:           {result.total_samples}")
    if hasattr(result, 'failed_samples') and result.failed_samples > 0:
        print(f"   Successful samples:     {result.successful_samples} ({result.successful_samples/result.total_samples*100:.1f}%)")
        print(f"   Failed samples:         {result.failed_samples} ({result.failed_samples/result.total_samples*100:.1f}%)")
        print(f"   ⚠️  Metrics below are based on successful samples only!")
    print(f"   Throughput:              {result.throughput:.2f} samples/s")
    print(f"   Avg latency (per sample): {result.avg_latency_per_sample:.2f} ms")
    print(f"   Avg latency (per batch):  {result.avg_latency_per_batch:.2f} ms")
    print(f"   Min latency (per batch):  {result.min_latency_per_batch:.2f} ms")
    print(f"   Max latency (per batch):  {result.max_latency_per_batch:.2f} ms")
    print()
    print("🎯 Accuracy Metrics:")
    print(f"   Overall accuracy:        {result.accuracy * 100.0:.2f}%")
    if hasattr(result, 'successful_samples') and result.successful_samples > 0:
        print(
            f"   Correct:                 {result.correct_predictions}/{result.successful_samples} (successful samples)"
        )
    else:
        print(
            f"   Correct:                 {result.correct_predictions}/{result.total_samples}"
        )
    print()
    print("📈 Safe Class:")
    print(f"   Precision:              {result.safe_precision * 100.0:.2f}%")
    print(f"   Recall:                 {result.safe_recall * 100.0:.2f}%")
    print()
    print("⚠️  Unsafe Class:")
    print(f"   Precision:              {result.unsafe_precision * 100.0:.2f}%")
    print(f"   Recall:                 {result.unsafe_recall * 100.0:.2f}%")


def print_comparison_table(results: Dict[int, BenchmarkResult]):
    """Print comparison table for all batch sizes."""
    print("\n" + "═" * 120)
    print("COMPREHENSIVE RESULTS TABLE")
    print("═" * 120)
    
    print("\n📊 THROUGHPUT (samples/s)")
    print("─" * 80)
    print(f"{'Batch Size':<15} {'Throughput':<20} {'Speedup vs BS=1':<20}")
    print("─" * 80)
    
    batch_sizes = sorted(results.keys())
    baseline_throughput = results.get(1, results[batch_sizes[0]]).throughput if results else 0.0
    
    for bs in batch_sizes:
        result = results[bs]
        speedup = result.throughput / baseline_throughput if baseline_throughput > 0 else 1.0
        print(f"{bs:<15} {result.throughput:<20.2f} {speedup:<20.2f}x")
    
    print("\n⏱️  LATENCY (ms)")
    print("─" * 80)
    print(f"{'Batch Size':<15} {'Per Sample':<20} {'Per Batch':<20}")
    print("─" * 80)
    
    for bs in batch_sizes:
        result = results[bs]
        print(f"{bs:<15} {result.avg_latency_per_sample:<20.2f} {result.avg_latency_per_batch:<20.2f}")
    
    print("\n🎯 ACCURACY (%)")
    print("─" * 60)
    print(f"{'Batch Size':<15} {'Accuracy':<20}")
    print("─" * 60)
    
    for bs in batch_sizes:
        result = results[bs]
        print(f"{bs:<15} {result.accuracy * 100.0:<20.2f}")
    
    print("\n" + "═" * 120)


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3Guard ONNXRuntime Batch Size Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to HuggingFace model directory (containing model.onnx and tokenizer files) "
        "or direct path to ONNX model file",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Optional path to tokenizer. If not specified, uses --model-path",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to Qwen3GuardTest dataset directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="thinking",
        help='Dataset split (thinking, thinking_loc, response_loc) [default: thinking]',
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to use [default: all]",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8,16",
        help="Comma-separated batch sizes to test [default: 1,2,4,8,16]",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for inference [default: cpu]",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Maximum sequence length for tokenization [default: 4096, matches vLLM benchmark]",
    )
    
    args = parser.parse_args()
    
    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    
    print("\n" + "═" * 80)
    print("Qwen3Guard ONNXRuntime Batch Size Benchmark")
    print("Testing ONNXRuntime Inference with Different Batch Sizes")
    print("═" * 80)
    
    print(f"Model path: {args.model_path}")
    if args.tokenizer_path:
        print(f"Tokenizer path: {args.tokenizer_path}")
    else:
        print(f"Tokenizer path: {args.model_path} (using model path)")
    print(f"Device: {args.device}")
    print(f"Batch sizes to test: {batch_sizes}")
    
    # Load dataset
    samples = load_dataset(args.dataset_path, args.split, args.max_samples)
    
    if not samples:
        print("Error: No samples loaded from dataset")
        return 1
    
    print("\n📂 Dataset loaded:")
    print(f"   Samples: {len(samples)}")
    unsafe_count = sum(1 for s in samples if s.label == "unsafe")
    safe_count = len(samples) - unsafe_count
    print(
        f"   Safe: {safe_count} ({safe_count / len(samples) * 100.0:.1f}%)"
    )
    print(
        f"   Unsafe: {unsafe_count} ({unsafe_count / len(samples) * 100.0:.1f}%)"
    )
    
    # Run benchmarks for each batch size
    all_results = {}
    
    for batch_size in batch_sizes:
        print("\n" + "═" * 80)
        print(f"Testing Batch Size: {batch_size}")
        print("═" * 80)
        
        result = benchmark_batch_size(
            args.model_path,
            args.tokenizer_path,
            samples,
            batch_size,
            args.device,
        )
        print_result(f"ONNXRuntime Batch Size {batch_size} Results", result)
        
        all_results[batch_size] = result
        
        # Small delay between tests
        time.sleep(1)
    
    # Print comparison table
    print_comparison_table(all_results)
    
    print("\n✅ BENCHMARK COMPLETE\n")
    
    return 0


if __name__ == "__main__":
    exit(main())

