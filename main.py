#!/usr/bin/env python3

"""
vLLM Benchmark Coordinator

Usage:
  ./main.py --list
  ./main.py [--endpoint URL] [--model MODEL] [--data-dir DIR] benchmark1 [benchmark2 ...]

Examples:
  ./main.py --list
  ./main.py narrativeqa humaneval
  ./main.py --endpoint http://127.0.0.1:8080 --model facebook/opt-125m alpaca triviaqa

@authors:
    - Alexander "Sasha" Joukov (alexander.joukov@stonybrook.edu)
    - Amir Zadeh (anajafizadeh@cs.stonybrook.edu)

@year: 2026
@by: File Systems & Storage Lab @ Stony Brook University
"""

import argparse
import os
import sys

import requests

from benchmarks import REGISTRY, list_all

REQUEST_TIMEOUT = 600  # seconds
DEFAULT_ENDPOINT = "http://127.0.0.1:8080"
DEFAULT_DATA_DIR = "./data"


def detect_model(endpoint: str) -> str:
    """Auto-detect the served model from the /v1/models endpoint."""
    r = requests.get(f"{endpoint.rstrip('/')}/v1/models", timeout=10)
    r.raise_for_status()
    models = [m.get("id") for m in r.json().get("data", []) if m.get("id")]
    if not models:
        print("Error: No models found at endpoint.", file=sys.stderr)
        sys.exit(1)
    if len(models) > 1:
        print(f"Multiple models found: {models}. Using first: {models[0]}", file=sys.stderr)
    return models[0]


def assert_server_up(endpoint: str, timeout_s: float = 5.0):
    r = requests.get(f"{endpoint.rstrip('/')}/health", timeout=timeout_s)
    r.raise_for_status()


def run_benchmark(name, benchmark, endpoint):
    """
    Run a single benchmark: iterate its entries, send HTTP requests,
    and print the status code for each.
    """
    print(f"\n=== Benchmark: {name} ===")
    n = 0
    ok = 0
    fail = 0

    for result in benchmark.run():
        uri = result["uri"]
        payload = result["payload"]

        # Skip entries with empty prompts (filtered by build_input)
        if not payload.get("prompt") and not payload.get("messages"):
            continue

        url = f"{endpoint.rstrip('/')}/v1{uri}"
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT
            )
            status = response.status_code
            if status < 400:
                ok += 1
            else:
                fail += 1
            n += 1
            print(f"  [{status}] {name} #{n}")

        except requests.exceptions.Timeout:
            fail += 1
            n += 1
            print(f"  [TIMEOUT] {name} #{n}")

        except Exception as e:
            fail += 1
            n += 1
            print(f"  [ERROR] {name} #{n}: {e}")

    print(f"--- {name}: {n} requests, {ok} ok, {fail} failed ---")
    return n, ok, fail


def main():
    ap = argparse.ArgumentParser(
        description="vLLM Benchmark Runner — connects to a running vLLM instance"
    )
    ap.add_argument(
        "--list", action="store_true", help="List available benchmarks and exit"
    )
    ap.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"vLLM endpoint URL (default: {DEFAULT_ENDPOINT})",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Model name (auto-detected from endpoint if omitted)",
    )
    ap.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Dataset cache directory (default: {DEFAULT_DATA_DIR})",
    )
    ap.add_argument(
        "benchmarks",
        nargs="*",
        help="Benchmark names to run",
    )

    args = ap.parse_args()

    if args.list:
        print("Available benchmarks:")
        for name in list_all():
            print(f"  {name}")
        return

    if not args.benchmarks:
        ap.print_usage()
        print("Error: specify at least one benchmark (or use --list).", file=sys.stderr)
        sys.exit(2)

    # Validate benchmark names
    for name in args.benchmarks:
        if name not in REGISTRY:
            print(f"Error: Unknown benchmark '{name}'.", file=sys.stderr)
            print(f"Available: {list_all()}", file=sys.stderr)
            sys.exit(2)

    endpoint = args.endpoint
    data_dir = os.environ.get("HF_HOME") or args.data_dir

    # Check server
    print(f"Checking server at {endpoint} ...")
    try:
        assert_server_up(endpoint)
    except Exception as e:
        print(f"Error: Cannot reach vLLM at {endpoint}: {e}", file=sys.stderr)
        sys.exit(1)
    print("Server is up.")

    # Detect or use provided model
    model = args.model or detect_model(endpoint)
    print(f"Model: {model}")

    os.makedirs(data_dir, exist_ok=True)

    # Run benchmarks sequentially
    total_n = 0
    total_ok = 0
    total_fail = 0

    for name in args.benchmarks:
        bench_cls = REGISTRY[name]
        benchmark = bench_cls.create(model=model, cache_dir=data_dir)
        n, ok, fail = run_benchmark(name, benchmark, endpoint)
        total_n += n
        total_ok += ok
        total_fail += fail

    print(f"\n=== All done: {total_n} total requests, {total_ok} ok, {total_fail} failed ===")

    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
