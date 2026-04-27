import json
import queue
import threading
import time
from typing import Any, Dict, Optional

import requests

from src.prometheus import MetricsSnapshot, fetch_snapshot
from src.runner.stats import RunnerStats


class Runner(threading.Thread):
    """Runner thread processes jobs from a queue, send requests, and records statistics."""

    def __init__(
        self,
        runner_id: int,
        endpoint: str,
        stats: RunnerStats,
        request_timeout: int,
        enable_metrics: bool = False,
        verbose: bool = False,
    ):
        """Initialize a Runner thread."""

        super().__init__(name=f"runner-{runner_id}", daemon=True)

        self._runner_id = f"runner-{runner_id}"
        self._endpoint = endpoint
        self._rto = request_timeout
        self._stats = stats
        self._enable_metrics = enable_metrics
        self._verbose = verbose
        self._jobs = queue.Queue[Optional[Dict[str, Any]]]()
        self._stop_event = threading.Event()

    def id(self) -> str:
        """Get the unique identifier of this runner."""

        return self._runner_id

    def stop(self) -> None:
        """Stop the thread by setting the stop event."""

        self._stop_event.set()

    def queue_job(self, job: Dict[str, Any]) -> None:
        """Add a job to the queue for processing."""

        self._jobs.put(job)

    def run(self) -> None:
        """Main loop of the runner thread.

        It continuously processes jobs from the queue until a `None` job is encountered, which signals the runner to stop.
        """

        while True:
            # if a stop signal is received, exit the loop
            if self._stop_event.is_set():
                return

            job = self._jobs.get()
            try:
                if job is None:
                    return

                self._process(
                    index=job["index"],
                    name=job["name"],
                    url=job["url"],
                    headers=job["headers"],
                    payload=job["payload"],
                )
            except Exception as e:
                print(e)
            finally:
                self._jobs.task_done()

    def _process(
        self,
        index: str,
        name: str,
        url: str,
        headers: Dict[str, str],
        payload: Any,
    ) -> None:
        """Sending a request to the specified URL with the given headers and payload, and recording the relevant statistics.

        Parameters
        ----------
        index : str
            The position of the current request among the total number of requests.
        name : str
            A name for the request, used for logging purposes.
        url : str
            The URL to which the request will be sent.
        headers : Dict[str, str]
            A dictionary of HTTP headers to include in the request.
        payload : Any
            The body of the request, which will be JSON-encoded before sending.

        Raises
        ------
        RuntimeError
            If an unexpected error happens when sending the HTTP request.
        """

        # calculate request size in bytes
        request_body = json.dumps(payload)

        # define statistics variables
        http_status: int = 0
        http_req_bytes: int = len(request_body.encode("utf-8"))
        http_res_bytes: int = 0
        http_latency: float = 0
        prompt_tokens: int = 0
        total_tokens: int = 0
        completion_tokens: int = 0
        start_datetime: str = ""
        end_datetime: str = ""
        
        metrics_snapshot: MetricsSnapshot = None

        if self._verbose:
            # dump the request body to a file for debugging purposes
            print(f"Request body for [{self.id()}] [{index}] {name}:\n{request_body}")

        try:
            # start the timer for latency measurement
            start_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            start = time.perf_counter()

            # send the request
            response: requests.Response
            if self._rto:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self._rto,
                )
            else:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                )

            # calculate latency in milliseconds
            http_latency = (time.perf_counter() - start) * 1000
            end_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            # if metrics collection is enabled, take a snapshot of metrics after receiving the response
            if self._enable_metrics:
                # wait a short moment to ensure that the metrics are updated after processing the request
                time.sleep(1)

                # take a metrics snapshot
                tout = self._rto if self._rto else 10  # use request timeout or a default value for metrics fetch timeout
                metrics_snapshot = fetch_snapshot(
                    base_url=self._endpoint, timeout=tout
                )

            http_status = response.status_code
            http_res_bytes = len(response.content)

            # record success or error based on status code
            if http_status == 200:
                self._stats.record_success(http_latency, http_req_bytes, http_res_bytes)

                # extract token counts from the response if available
                try:
                    response_json = response.json()
                    if "usage" in response_json:
                        usage = response_json["usage"]
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        total_tokens = usage.get("total_tokens", 0)
                except Exception:
                    pass  # if token counts are not available or response is not JSON, ignore the error
            else:
                self._stats.record_error(http_latency, http_req_bytes, http_res_bytes)

        except requests.exceptions.Timeout:
            http_status = 408
            http_latency = (
                self._rto * 1000
            )  # set latency to the request timeout value in milliseconds
            
            end_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            # on timeout record a timeout, but do post metrics poll if enabled
            self._stats.record_timeout(http_req_bytes)

            # if metrics collection is enabled, take a snapshot of metrics after the timeout
            if self._enable_metrics:
                # wait a short moment to ensure that the metrics are updated after processing the request
                time.sleep(1)

                # take a metrics snapshot
                tout = self._rto if self._rto else 10  # use request timeout or a default value for metrics fetch timeout
                metrics_snapshot = fetch_snapshot(
                    base_url=self._endpoint, timeout=tout
                )

        except Exception as e:
            raise RuntimeError(f"Error while processing an entry: {e}")

        # calculate and print the differences in metrics values before and after the request
        metrics_str = ""
        if self._enable_metrics:
            values = metrics_snapshot.as_dict()
            metrics_str = "\n".join(
                f"vllm:{metric} = {value:.4f}" for metric, value in values.items()
            )

        # log the process
        print(
            f"[{self.id()}] [{index}] [{http_status}] {name} "
            f"latency={http_latency:.2f}ms "
            f"req={http_req_bytes}B "
            f"resp={http_res_bytes}B "
            f"\nstart={start_datetime} , end={end_datetime}",
            f"\nprompt_tokens={prompt_tokens} , completion_tokens={completion_tokens} , total_tokens={total_tokens}",
            f"\n{metrics_str}",
        )
