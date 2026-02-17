"""Benchmark registry."""

from .alpaca import AlpacaBenchmark, LocalAlpacaBenchmark
from .humaneval import HumanEvalBenchmark
from .kvprobe import KVProbeBenchmark
from .leval import LEvalBenchmark
from .longbench_gov import LongBenchGovBenchmark
from .longbench_qmsum import LongBenchQMSumBenchmark, LocalLongBenchQMSumBenchmark
from .loogle import LooGLEBenchmark
from .narrativeqa import NarrativeQABenchmark, LocalNarrativeQABenchmark
from .sharegpt import ShareGPTBenchmark, LocalShareGPTBenchmark
from .triviaqa import TriviaQABenchmark
from .wikitext import WikitextBenchmark

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
}


def list_all():
    """Return list of all registered benchmark names."""
    return sorted(REGISTRY.keys())
