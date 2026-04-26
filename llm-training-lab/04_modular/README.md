# Phase 5: Modular / MAX — From Training to Production Serving

> This is a conceptual guide, not runnable code. MAX requires a separate install and is primarily Linux/cloud-based.

---

## What is Modular MAX?

**Modular MAX** is a high-performance inference engine for deploying trained models in production. It is NOT a training framework — it sits downstream of everything we built in Phases 1-3.

Think of it this way:
- **PyTorch** (what we used) = the kitchen where you cook the meal
- **MAX** = the restaurant that serves the meal to thousands of customers efficiently

MAX takes a trained model (like our LoRA-adapted medical model) and serves it with optimized inference — lower latency, higher throughput, and better hardware utilization than running `model.generate()` in Python.

---

## The Training → Serving Pipeline

```
Phase 1-3 (This Repo)              Phase 5 (Production)
━━━━━━━━━━━━━━━━━━━━━              ━━━━━━━━━━━━━━━━━━━━
Train model (PyTorch)         →    Export model
Save weights (.safetensors)   →    Load into MAX Engine
                                   Serve via HTTP API
                                   Handle concurrent requests
                                   Optimize for target hardware
```

---

## How Our LoRA Adapter Could Be Served via MAX

### Step 1: Merge the LoRA Adapter into the Base Model

In our training (Phase 3), we saved the adapter separately:
```
outputs/sft_model/       ← Base model (270 MB)
outputs/lora_adapter/    ← LoRA adapter (3.5 MB)
```

For serving, we'd merge them into a single model:
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base + adapter
base = AutoModelForCausalLM.from_pretrained("outputs/sft_model")
model = PeftModel.from_pretrained(base, "outputs/lora_adapter")

# Merge adapter into base weights (no adapter overhead at inference)
merged = model.merge_and_unload()

# Save the merged model
merged.save_pretrained("outputs/merged_medical_model")
```

After merging, the adapter's A×B matrices are folded into the original weights. The model is the same size as the base (270 MB) but now permanently includes the medical knowledge. No runtime adapter loading needed.

### Step 2: Export to MAX-Compatible Format

MAX supports models in ONNX, HuggingFace Transformers, and GGUF formats:
```bash
# Option A: Use HuggingFace format directly
# MAX can load from the saved HuggingFace directory

# Option B: Convert to GGUF for maximum efficiency
# (GGUF is a quantized format optimized for inference)
python -m llama_cpp.convert outputs/merged_medical_model --outtype q4_k_m
```

### Step 3: Serve via MAX Container

```bash
# Pull the MAX serving container
docker pull modular/max-serving:latest

# Serve the model
docker run -p 8000:8000 \
  -v ./outputs/merged_medical_model:/model \
  modular/max-serving:latest \
  --model-path /model \
  --max-batch-size 32
```

The model would then be accessible via a standard HTTP API:
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "### Medical Question:\nWhat is hypertension?\n\n### Answer:\n",
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

---

## Benchmark Comparison: HuggingFace vs MAX

For a model like SmolLM2-135M, the throughput comparison would look roughly like:

| Metric | HuggingFace `model.generate()` | MAX Engine |
|--------|-------------------------------|------------|
| Latency (single request) | ~200-500ms | ~50-150ms |
| Throughput (tokens/sec) | ~50-100 | ~500-2000 |
| Concurrent requests | 1 (sequential) | 32+ (batched) |
| Memory efficiency | High (full Python overhead) | Low (optimized runtime) |
| Startup time | 5-10s (Python import chain) | <1s (compiled) |

*Note: Exact numbers depend on hardware, model size, and sequence length. These are illustrative.*

The key insight: Python/PyTorch is designed for **training** (flexibility, autograd, easy debugging). MAX is designed for **serving** (throughput, latency, hardware utilization). Using PyTorch for serving is like using a food processor to serve dinner — it works, but it's not what it was designed for.

---

## Why MAX is Fast: Mojo and MLIR

MAX's performance advantage comes from two technologies:

### Mojo Language
Mojo is a programming language built by Modular that combines Python's usability with systems-level performance:
- **Python-compatible syntax** — looks like Python, runs like C++
- **Compile-time type system** — catches errors before runtime, enables aggressive optimization
- **Zero-cost abstractions** — high-level code compiles to the same machine code as hand-tuned C
- **Hardware-aware** — directly maps operations to SIMD, GPU kernels, and accelerator instructions

### MLIR Backend
MLIR (Multi-Level Intermediate Representation) is the compiler infrastructure behind MAX:
- **Graph-level optimization** — fuses operations, eliminates redundant computation
- **Hardware-specific codegen** — generates optimal code for each target (CPU, GPU, accelerator)
- **Kernel fusion** — combines multiple small operations into single efficient kernels
- **Memory planning** — optimizes data layout and movement to minimize cache misses

Together, Mojo + MLIR allow MAX to:
1. Take a PyTorch model graph
2. Compile it into hardware-optimized machine code
3. Serve it with minimal Python overhead
4. Batch and schedule requests efficiently

---

## When to Use What

| Scenario | Tool | Why |
|----------|------|-----|
| Training / fine-tuning | PyTorch | Flexibility, autograd, ecosystem |
| Experimentation | PyTorch + MLflow | Easy iteration, metric tracking |
| Development inference | HuggingFace `model.generate()` | Simple, quick, good for debugging |
| Production serving | MAX / vLLM / TGI | Throughput, latency, batching |
| Edge deployment | GGUF + llama.cpp / MAX | Small footprint, CPU-friendly |

For this training lab, PyTorch is the right choice. But if you were deploying the medical model to serve real users, you'd want MAX or a similar serving engine.

---

## Further Reading

- [Modular MAX documentation](https://docs.modular.com/max/)
- [Mojo language](https://docs.modular.com/mojo/)
- [MLIR project](https://mlir.llvm.org/)
- [vLLM](https://github.com/vllm-project/vllm) — another popular open-source LLM serving engine
- [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) — HuggingFace's serving solution
