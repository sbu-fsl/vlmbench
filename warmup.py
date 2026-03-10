import requests
import threading
import time
from typing import Optional

TARGET_UTILIZATION = 0.95
PROMPT_RATIO = 1.0 / 3.0
REQUEST_INTERVAL_S = 1.0
DEFAULT_REQUEST_TIMEOUT_S = 10.0
MIN_PROMPT_TOKENS = 16
MIN_GEN_TOKENS = 16


def _split_tokens_from_max_len(max_model_len: int) -> tuple[int, int]:
    usable = max(1, max_model_len - 2)
    prompt_tokens = max(MIN_PROMPT_TOKENS, int(usable * PROMPT_RATIO))
    gen_tokens = max(MIN_GEN_TOKENS, usable - prompt_tokens)

    if prompt_tokens + gen_tokens > usable:
        gen_tokens = max(MIN_GEN_TOKENS, usable - prompt_tokens)
        if prompt_tokens + gen_tokens > usable:
            prompt_tokens = max(MIN_PROMPT_TOKENS, usable - gen_tokens)

    return prompt_tokens, gen_tokens


def _get_kv_usage(metrics_url: str, timeout_s: float) -> float:
    r = requests.get(metrics_url, timeout=timeout_s)
    r.raise_for_status()
    for line in r.text.splitlines():
        if "vllm:kv_cache_usage_perc" in line:
            return float(line.split()[-1])
    return 0.0


def _keep_request_alive(
    completions_url: str,
    model: str,
    prompt_tokens: int,
    gen_tokens: int,
    request_timeout_s: float,
) -> None:
    prompt = "warmup " * prompt_tokens

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": gen_tokens,
        "temperature": 0,
        "stream": True,
    }

    with requests.post(
        completions_url,
        json=payload,
        stream=True,
        timeout=request_timeout_s,
    ) as r:
        r.raise_for_status()
        for _ in r.iter_lines():
            pass


def run_warmup_plugin(
    endpoint: str,
    model: str,
    max_model_len: int,
    total_kv_tokens: int,
    target_utilization: float = TARGET_UTILIZATION,
    request_interval_s: float = REQUEST_INTERVAL_S,
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
) -> None:
    completions_url = f"{endpoint.rstrip('/')}/v1/completions"
    metrics_url = f"{endpoint.rstrip('/')}/metrics"
    prompt_tokens, gen_tokens = _split_tokens_from_max_len(max_model_len)

    target_kv_tokens = int(total_kv_tokens * target_utilization)

    print("=== Warmup plugin ===")
    print("Total KV tokens:", total_kv_tokens)
    print("Target utilization:", target_utilization)
    print("Target KV tokens:", target_kv_tokens)
    print("Max model length:", max_model_len)
    print("Prompt tokens:", prompt_tokens)
    print("Generation tokens:", gen_tokens)

    threads = []

    while True:
        usage = _get_kv_usage(metrics_url, timeout_s=request_timeout_s)
        print("KV usage:", usage)

        if usage >= target_utilization:
            print("Target KV utilization reached")
            break

        t = threading.Thread(
            target=_keep_request_alive,
            args=(
                completions_url,
                model,
                prompt_tokens,
                gen_tokens,
                request_timeout_s,
            ),
            daemon=True,
        )
        t.start()
        threads.append(t)

        time.sleep(request_interval_s)

    print("Warmup complete")


def warmup(
    endpoint: str,
    model: str,
    max_model_len: int,
    total_kv_tokens: int,
    target_utilization: Optional[float] = None,
) -> None:
    run_warmup_plugin(
        endpoint=endpoint,
        model=model,
        max_model_len=max_model_len,
        total_kv_tokens=total_kv_tokens,
        target_utilization=(
            TARGET_UTILIZATION if target_utilization is None else target_utilization
        ),
    )
