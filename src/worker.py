import requests


class Worker:
    def __init__(self, request_timeout: int):
        self._n = 0
        self._ok = 0
        self._failed = 0
        self._rto = request_timeout

    def stats(self) -> tuple[int, int, int]:
        return (self._n, self._ok, self._failed)

    def process(self, name: str, url: str, headers: dict, payload: any) -> int:
        self._n += 1
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=self._rto
            )
            status = response.status_code
            if status < 400:
                self._ok += 1
            else:
                self._failed += 1
            print(f"  [{status}] {name} #{self._n}")

        except requests.exceptions.Timeout:
            self._fail += 1
            print(f"  [TIMEOUT] {name} #{self._n}")

        except Exception as e:
            self._fail += 1
            print(f"  [ERROR] {name} #{self._n}: {e}")
