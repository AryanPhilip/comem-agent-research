# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CoMEM-Agent is a research system for GUI web automation with scalable continuous memory. The key claim: encoding trajectories into 8 fixed continuous embeddings (via Q-Former + LoRA) outperforms text-based memory and scales monotonically with memory size.

The repo has two independent sub-projects:
- **`CoMEM-Agent-Inference/`** — evaluation/inference stack (runs experiments)
- **`CoMEM-Agent-train/`** — training stack for the Q-Former memory encoder

## Setup

```bash
# Recommended: using uv
uv venv && source .venv/bin/activate
uv pip install -e .
playwright install

# Alternative: conda
conda create -n gui-agent python=3.10 && conda activate gui-agent
pip install -r requirements.txt && playwright install
```

`flash-attn` requires CUDA toolkit. Inference runs require a vLLM server running separately for local models (Qwen2.5-VL, UI-TARS).

## Running Evaluations

All evaluation commands are run from the repo root; the shell script `cd`s into `CoMEM-Agent-Inference/` automatically.

```bash
# Baseline (no memory)
./CoMEM-Agent-Inference/run_baseline.sh --eval_type mmina --domain shopping --model qwen2.5-vl

# With text-based memory
./CoMEM-Agent-Inference/run_baseline.sh --eval_type mmina --domain shopping --model qwen2.5-vl --use_memory

# With continuous memory (CoMEM)
./CoMEM-Agent-Inference/run_baseline.sh --eval_type mmina --domain shopping --model qwen2.5-vl --use_continuous_memory

# Collect training data during a run
./CoMEM-Agent-Inference/run_baseline.sh --eval_type mmina --domain shopping --model qwen2.5-vl --collect_training_data
```

Supported `--eval_type` values: `mmina`, `mind2web`, `webvoyager`

Supported `--domain` values:
- mmina: `shopping`, `wikipedia`
- mind2web: `test_website`, `test_domain_Info`, `test_domain_Service`
- webvoyager: `test`, `Amazon`, `ArXiv`, `Booking`, `GitHub`, `Google_Map`, etc.

Results are saved to `results/<eval_type>/<domain>/<model>/<datetime>/` including `metrics_summary.json` and `metrics_summary.md`.

## Architecture

### Inference Stack (`CoMEM-Agent-Inference/`)

**Entry point**: `run.py` → calls `config()` (argument parser) → loads grounding model → constructs agent → runs the appropriate `TestRunner`.

**Agent** (`agent/agent.py`): `FunctionCallAgent` implements ReAct. On each step it builds a multimodal prompt (screenshot + history + memory), calls the VLM, parses the JSON action response, and dispatches to a tool. Tools are defined in `tools/` (GUI actions, web search, content analysis).

**Memory** (`memory/experience_memory.py`): The `Memory` class loads trajectory JSONL files from `training_data/`, builds CLIP embeddings, and indexes them with FAISS. On each new task, it retrieves the top-k similar trajectories and injects their action sequences as in-context examples. Two modes: text-only (CLIP text encoder) and multimodal (CLIP text+image concatenated embeddings).

**Continuous Memory path**: When `--use_continuous_memory` is set, `args.model` is forced to `'agent-qformer'` and `llm_config.py` loads `DirectTransformersModel` instead of a vLLM wrapper. This model calls the Q-Former compressor to encode retrieved trajectories into 8-token embeddings that are prepended to the VLM's input embedding sequence.

**Data flow for continuous memory**:
1. `Memory.retrieve_similar_conversations()` → returns JSONL file paths
2. `Memory.construct_experience_memory()` → parses actions + extracts screenshots per step
3. `knowledge_processor_vlm()` (`data_preparation/prepare_inference_data_memory.py`) → runs Q-Former on each trajectory to produce 8 continuous embeddings
4. Embeddings are prepended to the LLM input via `monkey_patch_forward.py`

### Training Stack (`CoMEM-Agent-train/`)

**Key files**:
- `src_agent/training/train.py` — entry point; freezes base VLM, enables grad only on `knowledge_processor` (Q-Former)
- `src_agent/training/qformer.py` — Q-Former with shared or independent PerceiverAttention layers; `num_queries=8` by default
- `src_agent/training/qwenVL_compressor.py` — subclasses `Qwen2_5_VLForConditionalGeneration`, adds `knowledge_processor` (Q-Former) and a frozen `model_inf` copy
- `src_agent/training/monkey_patch_forward.py` — patches Qwen's forward pass to accept prepended continuous embeddings
- `src_agent/training/data.py` — dataset loading; expects JSONL trajectories with `messages`, `response`, `similar_trajectories`, `recent_trajectory` fields

Training only updates ~1.2% of parameters (LoRA on Q-Former). Base VLM is always frozen.

### Adding a New Model

Edit `CoMEM-Agent-Inference/agent/llm_config.py`:
- Add to `model_name_map` (short name → HuggingFace ID)
- Add to `model_server_map` (short name → vLLM server URL)

## Key Caveats

- **Hardcoded model filter** in `memory/experience_memory.py:141`: only `qwen2.5-vl-32b` trajectories are loaded from the training data directory. Change this when using other trajectory sources.
- **`use_memory` and `use_continuous_memory` are `type=bool`** in argparse (not `action='store_true'`), so they must be passed as `--use_memory True`, not just `--use_memory`.
- Training data is saved to `training_data/<eval_type>/<domain>/<model>/<datetime>/` during runs with `--collect_training_data`.
- The FAISS index is built once at startup and is not updated during inference. Pass `--faiss_index_path` to reuse a pre-built index.
- `--bank_size` caps the number of memory entries used (truncates the FAISS index to the first N entries).
