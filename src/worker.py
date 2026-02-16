import json
import time
from typing import Any, Dict, Optional

import requests


class Worker:
    def __init__(self, request_timeout: int):
        self._rto = request_timeout

        # Counters
        self._total = 0
        self._success = 0
        self._http_error = 0
        self._timeout = 0
        self._exception = 0

        # Bytes
        self._total_request_bytes = 0
        self._total_response_bytes = 0

        # Latency stats
        self._latencies = []

    def stats(self) -> Dict[str, Any]:
        return {
            "total_requests": self._total,
            "success": self._success,
            "http_error": self._http_error,
            "timeout": self._timeout,
            "exception": self._exception,
            "avg_latency_ms": self._avg_latency(),
            "p95_latency_ms": self._percentile(95),
            "total_request_bytes": self._total_request_bytes,
            "total_response_bytes": self._total_response_bytes,
        }

    def process(
        self,
        name: str,
        url: str,
        headers: Dict[str, str],
        payload: Any,
    ) -> Optional[Dict[str, Any]]:

        self._total += 1

        request_body = json.dumps(payload)
        request_size = len(request_body.encode("utf-8"))
        self._total_request_bytes += request_size

        start = time.perf_counter()

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self._rto,
            )

            latency = (time.perf_counter() - start) * 1000
            self._latencies.append(latency)

            status = response.status_code
            response_size = len(response.content)
            self._total_response_bytes += response_size

            if status < 400:
                self._success += 1
            else:
                self._http_error += 1

            llm_meta = self._extract_llm_metadata(response)

            print(
                f"[{status}] {name} "
                f"latency={latency:.2f}ms "
                f"req={request_size}B "
                f"resp={response_size}B"
            )

            return {
                "status": status,
                "latency_ms": latency,
                "request_bytes": request_size,
                "response_bytes": response_size,
                "llm_meta": llm_meta,
            }

        except requests.exceptions.Timeout:
            self._timeout += 1
            print(f"[TIMEOUT] {name}")
            return None

        except Exception as e:
            self._exception += 1
            print(f"[ERROR] {name}: {e}")
            return None

    def _avg_latency(self) -> float:
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    def _percentile(self, p: int) -> float:
        if not self._latencies:
            return 0.0
        sorted_lat = sorted(self._latencies)
        k = int(len(sorted_lat) * p / 100)
        return sorted_lat[min(k, len(sorted_lat) - 1)]

    def _extract_llm_metadata(self, response) -> Dict[str, Any]:
        """
        Extract LLM-specific metadata if available.
        Works for OpenAI-compatible APIs.
        """
        try:
            data = response.json()

            return {
                "model": data.get("model"),
                "prompt_tokens": data.get("usage", {}).get("prompt_tokens"),
                "completion_tokens": data.get("usage", {}).get("completion_tokens"),
                "total_tokens": data.get("usage", {}).get("total_tokens"),
                "finish_reason": (
                    data.get("choices", [{}])[0].get("finish_reason")
                    if data.get("choices")
                    else None
                ),
            }

        except Exception:
            return {}
