# vLLM Bench: Real-World LLM Inference Benchmarking for vLLM

## Prereqs

* A running vLLM instance accessible via HTTP.
* Python 3.10+

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
# List available benchmarks
python main.py --list

# Run benchmarks against a vLLM endpoint
python main.py [--endpoint URL] [--model MODEL] [--data-dir DIR] benchmark1 [benchmark2 ...]
```

### Options

* `--endpoint URL` — vLLM endpoint (default: `http://127.0.0.1:8080`)
* `--model MODEL` — Model name (auto-detected from endpoint if omitted)
* `--data-dir DIR` — Dataset cache directory (default: `./data`)

### Examples

```bash
# Run with defaults (localhost:8080, auto-detect model)
python main.py narrativeqa humaneval

# Specify endpoint and model
python main.py --endpoint http://127.0.0.1:8080 --model facebook/opt-125m alpaca triviaqa

# Custom data directory
python main.py --data-dir /tmp/datasets narrativeqa
```

## Available Benchmarks

| Benchmark | Description |
|---|---|
| `alpaca` | Instruction following |
| `humaneval` | Python code generation |
| `kvprobe` | KV cache efficiency test |
| `leval` | Long context evaluation |
| `longbench_gov` | Government report summarization |
| `longbench_qmsum` | Meeting summarization |
| `loogle` | Long document summarization |
| `narrativeqa` | Story-based reading comprehension |
| `sharegpt` | Multi-turn conversations |
| `triviaqa` | Open-domain trivia QA |
| `wikitext` | Language modeling |

## Files

```
.
├── main.py              # Benchmark runner (CLI entry point)
├── benchmarks/          # Benchmark task implementations
├── dataloaders/         # Dataset loading utilities
├── src/                 # Core benchmark base classes
└── tasks/               # Task definitions
```

## Authors

* Alexander "Sasha" Joukov (alexander.joukov@stonybrook.edu)
* Amir Zadeh (anajafizadeh@cs.stonybrook.edu)

File Systems & Storage Lab @ Stony Brook University, 2026
