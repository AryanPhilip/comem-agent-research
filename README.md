# CoMEM-Agent Research

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-2563eb.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-16a34a.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/dataset-HuggingFace-f59e0b.svg)](https://huggingface.co/datasets/WenyiWU0111/CoMEM-agent-memory-trajectories)

</div>

<p align="center">
  <img src="CoMEM-Agent-Inference/media/agent_comem_combined.drawio (1).png" alt="CoMEM-Agent overview" width="100%">
</p>

CoMEM-Agent is a research codebase for **long-horizon GUI agents with scalable continuous memory**. The core idea is simple: instead of stuffing prior trajectories back into the model as long text summaries, compress each trajectory into a small fixed set of continuous embeddings and inject those directly into the VLM pipeline.

That shift matters because GUI tasks are dominated by state, layout, and visual context. As trajectories get longer, text history becomes noisy and expensive. Continuous memory preserves more of the useful signal while keeping the prompt compact.

## Why This Project Matters

- **Better memory representation**: previous GUI trajectories are encoded into **8 fixed continuous tokens** instead of long text blocks.
- **Parameter-efficient training**: only about **1.2% of parameters** are tuned using **LoRA on a Q-Former**.
- **Scalable data flywheel**: the pipeline discovers environments, synthesizes tasks, rolls out agents, and verifies outcomes automatically.
- **Large training corpus**: roughly **188,451 trajectories** collected for about **$1,972**.
- **Strong benchmark coverage**: evaluated across **MMInA**, **Mind2Web**, and **WebVoyager**.

## What Is In This Repo

This repo contains two coordinated stacks:

- `CoMEM-Agent-Inference/`
  End-to-end evaluation, agent runtime, browser environment, retrieval, continuous-memory injection, and benchmark runners.
- `CoMEM-Agent-train/`
  Training code for the Q-Former memory encoder and the modified Qwen-based continuous-memory model.

This version also includes engineering improvements for experimentation:

- explicit `memory_mode` support: `none`, `text`, `continuous`, `hybrid`
- dynamic memory refresh policies
- verifier and reflection hooks for long-horizon recovery
- structured metrics summaries for runs
- richer trajectory metadata for retrieval and ablations
- lazy training-data loading with benchmark/domain/horizon filters

## Core Idea

### Text memory breaks first

For GUI agents, long textual histories tend to:

- inflate context length
- lose precise visual grounding
- mix useful state with irrelevant narration
- degrade as more retrieved examples are added

### Continuous memory changes the tradeoff

CoMEM compresses each retrieved trajectory into a fixed-size latent representation and prepends it to the model input. In practice, this gives a cleaner retrieval-and-control loop:

1. retrieve similar trajectories
2. compress them into continuous memory tokens
3. inject them into the VLM
4. execute the next step with better long-horizon state awareness

## Repo Layout

```text
CoMEM-Agent/
├── CoMEM-Agent-Inference/
│   ├── agent/                   # ReAct agent, model wrappers, planner/verifier hooks
│   ├── browser_env/             # Playwright-based browser environment
│   ├── config/                  # CLI and runtime config
│   ├── data_preparation/        # On-the-fly data and memory prep
│   ├── memory/                  # Retrieval, indexing, condensation, continuous preprocessing
│   ├── MMInA_evaluation/        # MMInA runner
│   ├── Mind2Web_evaluation/     # Mind2Web runner
│   ├── webvoyager_evaluation/   # WebVoyager runner
│   ├── utils/                   # Metrics, reliability, data collection
│   └── run.py                   # Main evaluation entrypoint
├── CoMEM-Agent-train/
│   └── src_agent/training/      # Q-Former / Qwen training stack
├── tests/                       # Lightweight logic tests for memory/runtime wiring
├── requirements.txt
└── pyproject.toml
```

## Installation

### Using `uv`

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
playwright install
```

### Using `pip` / conda

```bash
conda create -n gui-agent python=3.10
conda activate gui-agent
pip install -r requirements.txt
playwright install
```

Notes:

- `flash-attn` may require a compatible CUDA toolchain.
- local open-source inference expects running model servers for supported VLMs.
- Playwright browsers must be installed before evaluation.

## Quick Start

### 1. Baseline run

```bash
python CoMEM-Agent-Inference/run.py \
  --evaluation_type mmina \
  --domain shopping \
  --model qwen2.5-vl \
  --memory_mode none
```

### 2. Text memory

```bash
python CoMEM-Agent-Inference/run.py \
  --evaluation_type mmina \
  --domain shopping \
  --model qwen2.5-vl \
  --memory_mode text
```

### 3. Continuous memory

```bash
python CoMEM-Agent-Inference/run.py \
  --evaluation_type mmina \
  --domain shopping \
  --memory_mode continuous \
  --memory_token_budget 8
```

### 4. Hybrid memory with verifier-backed recovery

```bash
python CoMEM-Agent-Inference/run.py \
  --evaluation_type mmina \
  --domain shopping \
  --memory_mode hybrid \
  --memory_refresh verifier \
  --enable_verifier \
  --enable_reflection_memory
```

Every run writes structured outputs under `results/...`, including:

- `metrics_summary.json`
- `metrics_summary.md`

Those summaries now track:

- task success rate
- average steps and duration
- repeated-action rate
- verifier intervention rate
- retrieval hit rate
- performance by horizon bucket

## Benchmarks

| Benchmark | Scope | Example domains |
|---|---|---|
| `mmina` | multimodal multi-hop web tasks | `shopping`, `wikipedia` |
| `mind2web` | cross-site web navigation | `test_website`, `test_domain_Info`, `test_domain_Service` |
| `webvoyager` | broader multi-domain web tasks | `Amazon`, `ArXiv`, `Booking`, `GitHub`, `Google_Map`, more |

## Memory and Control Features

The current runtime supports a more research-friendly control loop than a plain retrieved-example baseline.

### Memory modes

- `none`
  No retrieved memory.
- `text`
  Retrieved trajectories are injected as textual exemplars.
- `continuous`
  Retrieved trajectories are encoded and passed as continuous memory.
- `hybrid`
  Uses both textual memory and continuous memory together.

### Memory refresh policies

- `task_start`
  Retrieve once per task.
- `page_change`
  Refresh retrieval when the page state changes.
- `verifier`
  Refresh retrieval when the verifier detects drift or failure patterns.

### Verifier and reflection hooks

The runtime includes lightweight planning and recovery support for long-horizon tasks:

- planner-generated subgoals
- verifier checks for loops, missing elements, and wrong-page drift
- reflection memory to inject the most relevant corrective guidance

## Training Continuous Memory

The training stack lives in `CoMEM-Agent-train/` and is centered on a Q-Former-based compressor for GUI trajectories.

Key details:

- base VLM remains frozen
- LoRA is applied to the Q-Former path
- training data can be loaded lazily with benchmark/domain/success/horizon filters
- continuous-memory token budget is configurable for ablations

## Dataset and Checkpoints

### Dataset

Hugging Face dataset:

- [WenyiWU0111/CoMEM-agent-memory-trajectories](https://huggingface.co/datasets/WenyiWU0111/CoMEM-agent-memory-trajectories)

### Pretrained checkpoints

| Model | Base VLM | Checkpoint |
|---|---|---|
| Qwen2.5-VL + CoMEM | Qwen2.5-VL-7B-Instruct | [WenyiWU0111/lora_qformer_test_V4-700_merged](https://huggingface.co/WenyiWU0111/lora_qformer_test_V4-700_merged) |
| UI-TARS + CoMEM | UI-TARS-V1.5-7B | [WenyiWU0111/lora_qformer_uitars_test_V1-400_merged](https://huggingface.co/WenyiWU0111/lora_qformer_uitars_test_V1-400_merged) |

## Supported Models

Open-source VLMs:

- Qwen2.5-VL-7B
- Qwen2.5-VL-32B
- Qwen2-VL-7B
- UI-TARS-V1.5-7B
- CogAgent-9B
- WebSight-7B

Commercial APIs:

- GPT-4o
- Claude-family models
- Gemini-family models

To add a new model, update the model registry in `CoMEM-Agent-Inference/agent/llm_config.py`.

## Project Positioning

This repo is best read as a **research engineering project** at the intersection of:

- multimodal agents
- retrieval and memory systems
- long-horizon control
- GUI/web automation
- parameter-efficient adaptation

If you are evaluating this project as portfolio work, the strongest technical pieces are:

- continuous memory injection into the VLM stack
- large-scale trajectory collection and verification pipeline
- benchmark-driven evaluation across multiple GUI-agent tasks
- engineering extensions for hybrid memory, verifier-backed recovery, and structured metrics


## License

MIT
