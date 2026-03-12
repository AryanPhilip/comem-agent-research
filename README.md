# Auto-Scaling Continuous Memory For GUI Agent

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2510.09038-b31b1b.svg)](https://arxiv.org/abs/2510.09038)
[![arXiv](https://img.shields.io/badge/Website-CoMEMAgent-c8b6ff.svg)](https://wenyiwu0111.github.io/CoMEM-Agent-project-page/)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-GUI--Agent--Trajectories-yellow)](https://huggingface.co/datasets/WenyiWU0111/CoMEM-agent-memory-trajectories)
</div>

<p align="center">
  <img src="CoMEM-Agent-Inference/media/agent_comem_combined.drawio (1).png" alt="GUI-Agent Overview" width="100%">
</p>

This is the official code repository for the paper: [Auto-Scaling Continuous Memory For GUI Agent]().
## 📖 Introduction

We study how to endow GUI agents with **scalable continuous memory** that helps generalize across unfamiliar interfaces. Prior GUI agents compress past trajectories into text tokens, which balloons context length and misses decisive visual cues (*e.g.*, exact widget size and position). 

We propose a **continuous memory** that encodes each GUI trajectory into a fixed-length sequence of continuous embeddings using the VLM itself as an encoder; these embeddings are plugged directly into the backbone's input layer, sharply reducing context cost while preserving fine-grained visual information. As memory size and retrieval depth increase, performance improves monotonically, unlike text memories that degrade with long prompts.

### Key Features

- 🎯 **Fixed-length Continuous Memory**: Encode GUI trajectories into compact embeddings (8 continuous tokens)
- 🚀 **Efficient Fine-tuning**: Train only 1.2% of model parameters using LoRA on Q-Former
- 📈 **Scalable Performance**: Monotonic improvement with more memory, unlike text-based approaches
- 🔄 **Auto-scaling Data Flywheel**: Discover environments → Synthesize tasks → Roll out trajectories → Verify success
- 💰 **Cost-effective**: Collect 188,451 trajectories for ~$1,972
- 🏆 **SOTA Performance**: Qwen-2.5-VL-7B + continuous memory matches GPT-4o and Claude-4

### Data Flywheel

Our auto-scaling pipeline automatically grows the memory corpus:
1. **Discover**: Find new environments via search
2. **Synthesize**: Generate tasks with open-source VLMs
3. **Roll out**: Execute trajectories with the agent
4. **Verify**: Validate success with the same VLM

## 🚀 Quick Start

### Installation

#### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer written in Rust. Install it first:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or using pip: pip install uv
# Or using homebrew: brew install uv
```

Then set up the project:

```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .

# Install Playwright browsers
playwright install
```

#### Using pip (Alternative)

```bash
# Create environment
conda create -n gui-agent python=3.10
conda activate gui-agent

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install
```

**Note:** For `flash-attn`, you may need CUDA toolkit and build tools installed. See [flash-attn installation guide](https://github.com/Dao-AILab/flash-attention) for details.

## 📊 Benchmarks

We evaluate on multiple real-world GUI benchmarks:

### MMInA

- **Shopping** (200 tasks): E-commerce interactions
- **Wikipedia** (308 tasks): Information seeking

### Mind2Web

Cross-website task execution with:
- `test_website`: General websites
- `test_domain_Info`: Information domains
- `test_domain_Service`: Service domains

### WebVoyager

Multi-domain web navigation across 15+ domains including:
- E-commerce (Amazon, Apple)
- Information (ArXiv, Wikipedia, BBC News)
- Services (Booking, GitHub, Google Maps)
- And more...

## 🎮 Running Experiments

### Command Line Interface

Use the main script with flexible options:

```bash
CoMEM-Agent-Inference/run_baseline.sh \
    --eval_type mmina \
    --domain shopping \
    --model qwen2.5-vl \
    --max_steps 15 \
    --use_memory 
```

Every evaluation run now emits structured metrics summaries in the specified `--result_dir` (see `metrics_summary.json` and `metrics_summary.md`) so you can track success rates, average steps, and runtime at a glance.

### Available Options

```bash
Options:
  --eval_type TYPE          Benchmark for Evaluation (mmina, mind2web, webvoyager)
  --domain DOMAIN           Domain for evaluation
  --model MODEL             Model to use
  --max_steps N             Maximum steps per task (default: 15)
  --result_dir DIR          Results directory (default: results)
  --use_memory              Enable memory
  --use_continuous_memory   Enable continuous memory
  --checkpoint_path         Used only when use_continuous_memory is True
  --collect_training_data   Collect trajectory data for memory
  --help, -h                Show help message
```

### Example Scripts

We provide ready-to-use example scripts in the `CoMEM-Agent-Inference/examples/` directory:

#### Baseline Evaluation

```bash
# MMInA Shopping with Qwen2.5-VL
CoMEM-Agent-Inference/examples/run_mmina_shopping.sh

# MMInA Wikipedia
CoMEM-Agent-Inference/examples/run_mmina_wikipedia.sh

# Mind2Web Evaluation
CoMEM-Agent-Inference/examples/run_mind2web.sh

# WebVoyager Evaluation
CoMEM-Agent-Inference/examples/run_webvoyager.sh
```

#### With Memory

```bash
# Text-based memory
CoMEM-Agent-Inference/examples/run_with_text_memory.sh

# Continuous memory (CoMEM)
CoMEM-Agent-Inference/examples/run_with_continuous_memory.sh
```

See [`CoMEM-Agent-Inference/examples/README.md`](CoMEM-Agent-Inference/examples/README.md) for detailed documentation and more examples.

## 📦 Dataset

We release our auto-collected trajectory dataset on HuggingFace:

**[GUI-Agent-Trajectories](https://huggingface.co/datasets/WenyiWU0111/CoMEM-agent-memory-trajectories)**

This dataset contains **188,451** GUI interaction trajectories collected through our auto-scaling data flywheel across diverse websites and tasks:

- **Multi-domain Coverage**: E-commerce, information seeking, booking, social media, and more
- **Rich Annotations**: Task descriptions, Website url, Screenshots, model responses, and actions at each step
- **Cost-effective**: Collected **188,451** trajectories for **$1,972** using our automated pipeline
- **Self-synthesized**: Tasks generated by open-source VLMs, verified automatically

## 🤗 Pre-trained Checkpoints

We release our continuous memory checkpoints on HuggingFace:

| Model | Base VLM | HuggingFace Link |
|-------|----------|------------------|
| **Qwen2.5-VL + CoMEM** | Qwen2.5-VL-7B-Instruct |  [WenyiWU0111/lora_qformer_test_V4-700_merged](https://huggingface.co/WenyiWU0111/lora_qformer_test_V4-700_merged) |
| **UI-TARS + CoMEM** | UI-TARS-V1.5-7B |  [WenyiWU0111/lora_qformer_uitars_test_V1-400_merged](https://huggingface.co/WenyiWU0111/lora_qformer_uitars_test_V1-400_merged) |


## 🔧 Supported Models

### Open-Source VLMs

- **Qwen2.5-VL-7B** / **Qwen2.5-VL-32B**: SOTA vision-language models
- **Qwen2-VL-7B**: Previous generation Qwen VL
- **UI-TARS-V1.5-7B**: Specialized for GUI understanding
- **CogAgent-9B**: Multi-modal agent model
- **WebSight-7B**: Web-specific VLM

### Commercial APIs

- **GPT-4o**: OpenAI's multimodal model
- **Claude-3.5-Sonnet / Claude-4**: Anthropic's model
- **Gemini-2.5-Pro**: Google's latest model

### Adding New Models

To add a new model, edit `CoMEM-Agent-Inference/agent/llm_config.py`:

```python
# In create_direct_vllm_model function
model_name_map = {
    'your-model': 'HuggingFace/model-name',
    ...
}

model_server_map = {
    'your-model': 'http://localhost:PORT/v1',
    ...
}
```

## 📂 CoMEM-Agent-Inference Structure

```
GUI-Agent/
├── actions/              # Action creation and parsing
├── agent/                # Core agent implementation with ReAct
├── browser_env/          # Playwright-based browser environment
├── config/               # Configuration and argument parsing
├── data_preparation/     # Data preparation scripts
├── examples/             # Example scripts for running experiments
├── memory/               # Experience memory system (FAISS indexing)
├── memory_evolution/     # Data flywheel for memory expansion
├── Mind2Web_evaluation/  # Mind2Web benchmark evaluation
├── MMInA_evaluation/     # MMInA benchmark evaluation
├── mmina/                # MMInA dataset
├── tools/                # Function calling tools (GUI, search, analysis)
├── utils/                # Shared utilities and helpers
├── webvoyager_evaluation/# WebVoyager benchmark evaluation
├── run_baseline.sh       # Main evaluation script
└── run.py                # Python entry point
```

Each directory contains a detailed `README.md` with component documentation.

## 🎓 Training Continuous Memory

For training your own continuous memory models, please refer to [this folder]((CoMEM-Agent-train)) and our training repository:

**[CoMEM Training Repository](https://github.com/WenyiWU0111/CoMEM)**

The repository includes:
- Training scripts for Q-Former memory encoder
- Data synthesis pipeline
- Memory retrieval and indexing
- Evaluation protocols

Key training details:
- **Parameters**: Only 1.2% of the model (LoRA on Q-Former)
- **Memory Size**: 8 continuous embeddings per trajectory
- **Base Model**: Frozen during training and inference

## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{wu2025comemagent,
  title={Auto-Scaling Continuous Memory For GUI Agent},
  author={Wenyi Wu, Kun Zhou, Ruoxin Yuan, Vivian Yu, Stephen Wang, Zhiting Hu, Biwei Huang},
  journal={arXiv preprint arXiv:2510.09038},
  year={2025}
}
```

## 📧 Contact

For questions or collaboration opportunities, please reach out through Email: wew058@ucsd.edu

---
