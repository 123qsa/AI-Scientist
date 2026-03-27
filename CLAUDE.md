# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The AI Scientist is a fully automated scientific discovery system that uses LLMs to generate research ideas, conduct experiments, write papers in LaTeX, and perform peer review. It operates on research templates (e.g., nanoGPT, 2D Diffusion, Grokking) and produces complete academic papers.

**Key Papers/Resources:**
- Main paper: https://arxiv.org/abs/2408.06292
- Blog post: https://sakana.ai/ai-scientist/

## Environment Setup

**Requirements:** Linux with NVIDIA GPUs and CUDA. CPU-only execution is not feasible.

```bash
# Create conda environment
conda create -n ai_scientist python=3.11
conda activate ai_scientist

# Install LaTeX (required for paper generation)
sudo apt-get install texlive-full

# Install Python dependencies
pip install -r requirements.txt
```

**API Keys (set as environment variables):**
- `OPENAI_API_KEY` - for GPT-4o, o1, o3 models
- `ANTHROPIC_API_KEY` - for Claude models
- `DEEPSEEK_API_KEY` - for DeepSeek models
- `GEMINI_API_KEY` - for Google Gemini models
- `KIMI_API_KEY` - for Kimi models via API Key (optional if using OAuth)
- `S2_API_KEY` - Semantic Scholar (optional, for literature search)
- `OPENALEX_MAIL_ADDRESS` - for OpenAlex alternative (no API key needed)

### Kimi OAuth Login (推荐)

如果你订阅了 Kimi 会员，可以使用 OAuth 登录而无需 API Key：

```bash
# 1. 安装 Kimi CLI
pip install kimi-cli

# 2. 登录 Kimi 账号（浏览器 OAuth 授权）
kimi login

# 3. 验证登录状态
kimi --version

# 4. 验证 OAuth 配置（可选但推荐）
python verify_kimi_oauth.py

# 5. 运行 AI-Scientist（会自动使用 OAuth token）
python launch_scientist.py --model "kimi-k2.5" --experiment nanoGPT_lite --num-ideas 2
```

OAuth token 存储在 `~/.kimi/credentials/kimi-code.json`，AI-Scientist 会自动读取并通过 kimi CLI 子进程调用 API。

如果遇到 OAuth 问题，运行 `python verify_kimi_oauth.py` 可以诊断配置问题。

## Common Commands

### Template Setup (Required Before Running)

Each template requires a baseline run (`run_0`) for comparison:

```bash
# NanoGPT template
cd templates/nanoGPT
python experiment.py --out_dir run_0
python plot.py

# 2D Diffusion template (requires NPEET installation first)
cd templates/2d_diffusion
python experiment.py --out_dir run_0
python plot.py

# Grokking template
cd templates/grokking
python experiment.py --out_dir run_0
python plot.py
```

### Running AI Scientist

```bash
# Generate papers (sequential)
python launch_scientist.py --model "claude-3-5-sonnet-20241022" --experiment nanoGPT_lite --num-ideas 2

# Use Kimi K2.5
python launch_scientist.py --model "kimi-k2.5" --experiment nanoGPT_lite --num-ideas 2

# Parallel execution (one per GPU)
python launch_scientist.py --model "gpt-4o-2024-05-13" --experiment nanoGPT_lite --num-ideas 4 --parallel 4

# Skip idea generation (use existing ideas.json)
python launch_scientist.py --model "claude-3-5-sonnet-20241022" --experiment nanoGPT_lite --skip-idea-generation

# Skip novelty check
python launch_scientist.py --model "claude-3-5-sonnet-20241022" --experiment nanoGPT_lite --skip-novelty-check

# Use OpenAlex instead of Semantic Scholar
python launch_scientist.py --model "claude-3-5-sonnet-20241022" --experiment nanoGPT_lite --engine openalex

# Enable paper improvement based on reviews
python launch_scientist.py --model "claude-3-5-sonnet-20241022" --experiment nanoGPT_lite --improvement
```

### Reviewing Papers

```python
from ai_scientist.perform_review import load_paper, perform_review
import openai

client = openai.OpenAI()
paper_text = load_paper("report.pdf")
review = perform_review(
    paper_text,
    model="gpt-4o-2024-05-13",
    client=client,
    num_reflections=5,
    num_fs_examples=1,
    num_reviews_ensemble=5,
    temperature=0.1,
)
```

## Architecture

### Core Modules (`ai_scientist/`)

**`llm.py`**: LLM client abstraction and API handling
- `create_client(model)` - Returns appropriate client for the model
- `get_response_from_llm()` - Single response generation
- `get_batch_responses_from_llm()` - For ensemble reviews
- `AVAILABLE_LLMS` - List of supported models
- Supports: OpenAI, Anthropic (direct/Bedrock/Vertex), DeepSeek, OpenRouter, Gemini, Kimi (Moonshot)

**`generate_ideas.py`**: Research idea generation and novelty checking
- `generate_ideas()` - Generates ideas using seed_ideas.json as examples
- `check_idea_novelty()` - Uses Semantic Scholar or OpenAlex to check literature
- `search_for_papers()` - Literature search API wrapper

**`perform_experiments.py`**: Code execution and experimentation
- `perform_experiments()` - Main experiment loop using Aider for code editing
- `run_experiment()` - Executes `python experiment.py --out_dir=run_N`
- `run_plotting()` - Executes `python plot.py`
- MAX_ITERS = 4 (max code iterations), MAX_RUNS = 5 (max experiment runs)

**`perform_writeup.py`**: LaTeX paper generation
- `perform_writeup()` - Generates paper section by section
- `generate_latex()` - Compiles LaTeX with error correction
- Uses `per_section_tips` dictionary for guidance on each section
- Performs citation search and insertion via Semantic Scholar/OpenAlex

**`perform_review.py`**: Paper review generation
- `perform_review()` - Generates structured review with scores
- `load_paper()` - Extracts text from PDF (tries pymupdf4llm, pymupdf, pypdf)
- Supports ensemble reviewing with meta-review aggregation
- Few-shot examples in `fewshot_examples/` directory

### Template Structure

Each template in `templates/` is a self-contained research domain:

```
templates/<name>/
├── experiment.py          # Main experiment code (must accept --out_dir)
├── plot.py               # Visualization generation
├── prompt.json           # System prompt and task description
├── seed_ideas.json       # Example ideas for few-shot generation
├── latex/
│   ├── template.tex      # LaTeX paper template
│   ├── iclr2024_conference.bst  # Bibliography style
│   └── *.sty             # LaTeX style files
└── run_0/                # Baseline results (user must create)
    └── final_info.json   # Results summary
```

**Official templates:** nanoGPT, 2d_diffusion, grokking
**Community templates:** MACE, earthquake-prediction, mobilenetV3, probes, seir, sketch_rnn, tensorf

### Main Entry Point (`launch_scientist.py`)

Orchestrates the full pipeline:
1. Generates/checks ideas
2. Spawns workers (sequential or parallel across GPUs)
3. Each worker:
   - Copies template to `results/<experiment>/<timestamp>_<idea_name>/`
   - Runs experiments via Aider
   - Generates LaTeX writeup
   - Performs review
   - Optionally improves based on review

Results are saved to `results/<experiment>/` with timestamped folders.

## Key Implementation Details

### Aider Integration

The system uses [Aider](https://aider.chat/) for LLM-driven code editing:
- Edit format: "diff" (SEARCH/REPLACE blocks)
- Git disabled (`use_git=False`)
- Operates on `experiment.py`, `plot.py`, and notes

### Experiment Flow

1. Baseline results loaded from `run_0/final_info.json`
2. Coder modifies `experiment.py` to implement idea
3. `python experiment.py --out_dir=run_N` executed (N from 1 to MAX_RUNS)
4. Results parsed from `run_N/final_info.json`
5. Process repeats until experiments complete or MAX_ITERS reached
6. `plot.py` modified to generate visualizations
7. `notes.txt` updated with experiment descriptions

### LaTeX Generation

- Template must have `references.bib` inside `filecontents` environment
- Citation lookups use Semantic Scholar API by default
- `chktex` used for linting with automatic error correction
- PDF compilation: pdflatex → bibtex → pdflatex → pdflatex

### Parallelization

- Uses Python `multiprocessing`
- Each worker assigned a specific GPU via `CUDA_VISIBLE_DEVICES`
- Staggered startup (150s delay between workers)

## Important Notes

- **Always run baseline** (`run_0`) before launching experiments - hardware-dependent timing comparisons require it
- **Containerization recommended** - The system executes LLM-generated code; use Docker for safety
- **Cost**: ~$15 per paper with Claude Sonnet 3.5 (DeepSeek is more cost-effective)
- **Success rate varies** by model and template complexity
- Reviews are best done with GPT-4o; other models may have positivity bias
- Never commit API keys to the repository
