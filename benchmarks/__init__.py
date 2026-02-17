"""Benchmark registry."""

from .alpaca import AlpacaBenchmark, LocalAlpacaBenchmark
from .humaneval import HumanEvalBenchmark
from .kvprobe import KVProbeBenchmark
from .leval import LEvalBenchmark
from .longbench_gov import LongBenchGovBenchmark
from .longbench_qmsum import LocalLongBenchQMSumBenchmark, LongBenchQMSumBenchmark
from .loogle import LooGLEBenchmark
from .narrativeqa import LocalNarrativeQABenchmark, NarrativeQABenchmark
from .sharegpt import LocalShareGPTBenchmark, ShareGPTBenchmark
from .triviaqa import TriviaQABenchmark
from .wikitext import WikitextBenchmark
from .wmt16 import WMT16Benchmark

REGISTRY = {
    "alpaca": AlpacaBenchmark,
    "local_alpaca": LocalAlpacaBenchmark,
    "triviaqa": TriviaQABenchmark,
    "narrativeqa": NarrativeQABenchmark,
    "local_narrativeqa": LocalNarrativeQABenchmark,
    "wikitext": WikitextBenchmark,
    "humaneval": HumanEvalBenchmark,
    "longbench_gov": LongBenchGovBenchmark,
    "longbench_qmsum": LongBenchQMSumBenchmark,
    "local_longbench_qmsum": LocalLongBenchQMSumBenchmark,
    "leval": LEvalBenchmark,
    "kvprobe": KVProbeBenchmark,
    "sharegpt": ShareGPTBenchmark,
    "local_sharegpt": LocalShareGPTBenchmark,
    "loogle": LooGLEBenchmark,
    "wmt16": WMT16Benchmark,
}


def list_all():
    """Return list of all registered benchmark names."""
    return sorted(REGISTRY.keys())
