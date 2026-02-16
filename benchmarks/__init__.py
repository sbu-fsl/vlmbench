"""Benchmark registry."""

from .alpaca import AlpacaBenchmark
from .humaneval import HumanEvalBenchmark
from .kvprobe import KVProbeBenchmark
from .leval import LEvalBenchmark
from .longbench_gov import LongBenchGovBenchmark
from .longbench_qmsum import LongBenchQMSumBenchmark
from .loogle import LooGLEBenchmark
from .narrativeqa import NarrativeQABenchmark
from .sharegpt import ShareGPTBenchmark
from .triviaqa import TriviaQABenchmark
from .wikitext import WikitextBenchmark

REGISTRY = {
    "alpaca": AlpacaBenchmark,
    "triviaqa": TriviaQABenchmark,
    "narrativeqa": NarrativeQABenchmark,
    "wikitext": WikitextBenchmark,
    "humaneval": HumanEvalBenchmark,
    "longbench_gov": LongBenchGovBenchmark,
    "longbench_qmsum": LongBenchQMSumBenchmark,
    "leval": LEvalBenchmark,
    "kvprobe": KVProbeBenchmark,
    "sharegpt": ShareGPTBenchmark,
    "loogle": LooGLEBenchmark,
}


def list_all():
    """Return list of all registered benchmark names."""
    return sorted(REGISTRY.keys())
