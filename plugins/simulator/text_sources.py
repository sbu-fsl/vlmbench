from __future__ import annotations

import enum
import json
import os
import random
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


class TaskType(str, enum.Enum):
    """Supported prompt task types."""

    SUMMARIZE = "summarize"
    QA = "qa"
    CHAT = "chat"
    EXPLAIN = "explain"
    CONTINUE = "continue"


_SUFFIX_TEMPLATES: dict[TaskType, list[str]] = {
    TaskType.SUMMARIZE: [
        "Summarize the above passage in three to five sentences.",
        "Write a concise summary of the text above, highlighting the key points.",
        "Provide a brief overview of the main ideas from the passage above.",
        "In your own words, summarize what the text above is about.",
        "Condense the above passage into a single short paragraph.",
        "Capture the essential message of the passage above in a few sentences.",
        "State the central ideas from the text above without unnecessary detail.",
        "Produce a short summary that preserves the most important information.",
        "Summarize the passage above for someone who has not read it.",
        "Write a compact explanation of the main content presented above.",
        "Extract the most important insights from the passage above.",
        "Reduce the above text to its core meaning in concise form.",
    ],
    TaskType.EXPLAIN: [
        "Explain the key concepts from the passage above as if speaking to a curious beginner.",
        "What are the most important ideas in the text above? Explain each one clearly.",
        "Describe the main points from the text above in plain, simple language.",
        "Break down the core ideas of the passage above and explain them step by step.",
        "Clarify the meaning of the passage above in an accessible way.",
        "Explain the underlying concepts in the text above with practical examples.",
        "Rewrite the ideas above so they are easier for a newcomer to understand.",
        "Teach the content of the passage above as if introducing it in a classroom.",
        "Explain why the ideas in the passage above matter.",
        "Interpret the text above and describe its important implications.",
        "Walk through the main concepts from the passage above carefully.",
        "Explain the subject of the passage above using straightforward language.",
    ],
    TaskType.CHAT: [
        (
            "You are a helpful assistant. A user has just read the passage above and wants "
            "to discuss it. Engage them in an informative conversation about the topic."
        ),
        (
            "Act as a knowledgeable tutor. The student has just read the passage above. "
            "Ask them a thought-provoking question about it, then provide a brief answer."
        ),
        (
            "Based on the passage above, write a short dialogue between a curious student "
            "and an expert. The expert should clarify the main ideas in the text."
        ),
        (
            "Simulate a conversation where a reader asks follow-up questions about the "
            "passage above and an assistant answers clearly."
        ),
        (
            "Create a short educational exchange about the text above between two people "
            "trying to understand its main message."
        ),
        (
            "Imagine someone is confused after reading the passage above. Write a brief "
            "assistant response that helps them understand it."
        ),
        ("Write a natural question-and-answer exchange inspired by the passage above."),
    ],
    TaskType.CONTINUE: [
        "Continue the text above in the same writing style for at least two more paragraphs.",
        "Write the next paragraph that would naturally follow the passage above.",
        "Extend the above passage with additional relevant facts and detail.",
        "Add a concluding section to the passage above that ties the ideas together.",
        "Develop the ideas above further while preserving tone and structure.",
        "Continue writing as if the original author were expanding the topic.",
        "Add another section that logically follows the discussion above.",
        "Write a continuation that deepens the explanation already given.",
        "Expand the passage above by introducing a related point or example.",
        "Carry the text forward naturally without changing its style.",
    ],
    TaskType.QA: [
        "Based on the passage above, what is the main topic being discussed? Explain in detail.",
        "What conclusions can be drawn from the text above? Support your answer with evidence from the passage.",
        "Identify and explain three key facts presented in the passage above.",
        "What is the central argument or message of the passage above?",
        "Answer a question that naturally arises from the text above and justify your response.",
        "What important evidence does the passage above provide?",
        "Which ideas in the text above are most critical for understanding it?",
        "What can a reader infer from the passage above?",
        "Describe one major takeaway from the passage above and explain why it matters.",
        "What problem or concept is the passage above trying to address?",
    ],
}


@dataclass
class PromptPair:
    """A (prefix, suffix) pair ready for the simulator."""

    prefix: str
    """The passage text — used as the shared KV-cache prefix."""

    suffix: str
    """The task instruction — appended as the unique request suffix."""

    task: TaskType
    """The task type this pair exercises."""

    source_name: str
    """Which backend produced the passage."""

    @property
    def full_prompt(self) -> str:
        """prefix + blank line + suffix."""

        return f"{self.prefix}\n\n{self.suffix}"


class TextSource(ABC):
    """Abstract text backend."""

    name: str = "base"

    @abstractmethod
    def fetch_passage(self, min_chars: int = 500, max_chars: int = 3000) -> str:
        """Return a passage of meaningful English text."""
        pass

    def fetch_qa_pair(self, max_chars: int = 3000) -> tuple[str, str]:
        """Return ``(context, question)``.  Default implementation uses fetch_passage."""

        passage = self.fetch_passage(max_chars=max_chars)
        question = "What is the main topic discussed in this passage?"

        return passage, question


class WikitextSource(TextSource):
    """Draws passages from the *wikitext-103* HuggingFace dataset."""

    name = "wikitext"

    def __init__(
        self,
        split: str = "train",
        cache_dir: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        import datasets as hfds

        self._rng = random.Random(seed)

        dataset_path = None
        if cache_dir:
            dataset_path = os.path.join(cache_dir, f"wikitext-103-raw-v1-{split}")

        # try loading from local disk cache
        if dataset_path and os.path.exists(dataset_path):
            try:
                self._ds = hfds.load_from_disk(dataset_path)
            except Exception:
                self._ds = self._load_and_cache(hfds, split, cache_dir, dataset_path)
        else:
            # download and cache
            self._ds = self._load_and_cache(hfds, split, cache_dir, dataset_path)

        self._size = len(self._ds)

    def _load_and_cache(self, hfds, split, cache_dir, dataset_path):
        ds = hfds.load_dataset(
            "wikitext",
            "wikitext-103-raw-v1",
            split=split,
            cache_dir=cache_dir,
        )

        # Save explicitly for future runs
        if dataset_path:
            try:
                ds.save_to_disk(dataset_path)
            except Exception:
                pass

        return ds

    def fetch_passage(self, min_chars: int = 500, max_chars: int = 3000) -> str:
        """Concatenate consecutive wikitext rows until we have at least *min_chars*
        characters, then trim to *max_chars*.
        """

        start = self._rng.randint(0, self._size - 1)
        parts: list[str] = []
        total = 0

        for offset in range(2000):
            idx = (start + offset) % self._size
            text = self._ds[idx]["text"].strip()

            if not text or text.startswith(" ="):
                continue

            parts.append(text)
            total += len(text) + 1
            if total >= min_chars:
                break

        passage = " ".join(parts)[:max_chars].strip()

        return passage if len(passage) >= min_chars else _FALLBACK_TEXT


class SQuADSource(TextSource):
    """Draws (context, question) pairs from *SQuAD v1.1* via HuggingFace datasets."""

    name = "squad"

    def __init__(
        self,
        split: str = "train",
        cache_dir: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        import datasets as hfds

        self._ds = hfds.load_dataset(
            "rajpurkar/squad",
            split=split,
            cache_dir=cache_dir,
        )

        self._rng = random.Random(seed)
        self._size = len(self._ds)

    def fetch_passage(self, min_chars: int = 500, max_chars: int = 3000) -> str:
        context, _ = self.fetch_qa_pair(max_chars=max_chars)
        return context

    def fetch_qa_pair(self, max_chars: int = 3000) -> tuple[str, str]:  # type: ignore[override]
        idx = self._rng.randint(0, self._size - 1)
        row = self._ds[idx]

        return row["context"][:max_chars], row["question"]


class WikipediaSource(TextSource):
    """Fetches real Wikipedia article text on-the-fly."""

    name = "wikipedia"

    def __init__(self, lang: str = "en", seed: Optional[int] = None) -> None:
        try:
            import wikipedia as _wp
        except ImportError as exc:
            raise ImportError(
                "WikipediaSource requires the `wikipedia` package. "
                "Install it with:  pip install wikipedia"
            ) from exc

        self._wp = _wp
        self._wp.set_lang(lang)
        self._rng = random.Random(seed)
        self._seed = seed
        self._page_cache: dict[str, str] = {}
        self._prefetched_passages: list[str] = []

    def _page_text(self, title: str, max_chars: int) -> str:
        cached = self._page_cache.get(title)
        if cached is not None:
            return cached[:max_chars]

        page = self._wp.page(title, auto_suggest=False, preload=False)
        # Strip section headings that start with == for cleaner prose chunks.
        lines = [line for line in page.content.splitlines() if not line.startswith("=")]
        text = " ".join(lines).strip()
        self._page_cache[title] = text
        return text[:max_chars]

    def _load_prefetch_snapshot(
        self,
        snapshot_path: str,
        min_chars: int,
        max_chars: int,
        max_count: int,
    ) -> bool:
        if not os.path.exists(snapshot_path):
            return False

        try:
            with open(snapshot_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception:
            return False

        if not isinstance(data, dict):
            return False

        raw_passages = data.get("passages")
        if not isinstance(raw_passages, list):
            return False

        loaded: list[str] = []
        for item in raw_passages:
            if not isinstance(item, str):
                continue
            text = " ".join(item.split())[:max_chars].strip()
            if len(text) >= min_chars:
                loaded.append(text)
            if len(loaded) >= max_count:
                break

        self._prefetched_passages = loaded
        return len(self._prefetched_passages) > 0

    def _save_prefetch_snapshot(self, snapshot_path: str) -> None:
        payload = {
            "version": 1,
            "seed": self._seed,
            "passages": self._prefetched_passages,
        }

        parent = os.path.dirname(snapshot_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(snapshot_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False)

    def prefetch_passages(
        self,
        count: int,
        min_chars: int = 500,
        max_chars: int = 3000,
        snapshot_path: Optional[str] = None,
    ) -> int:
        """Populate an in-memory deterministic passage pool.

        If *snapshot_path* exists, passages are loaded from disk for strict
        cross-run reproducibility. Otherwise, passages are fetched and optionally
        saved to *snapshot_path*.
        """

        count = max(0, int(count))
        if count == 0:
            self._prefetched_passages = []
            return 0

        if snapshot_path and self._load_prefetch_snapshot(
            snapshot_path=snapshot_path,
            min_chars=min_chars,
            max_chars=max_chars,
            max_count=count,
        ):
            return len(self._prefetched_passages)

        passages: list[str] = []
        attempts = 0
        max_attempts = max(50, count * 12)

        while len(passages) < count and attempts < max_attempts:
            attempts += 1
            try:
                title = self._rng.choice(_WIKIPEDIA_TOPICS)
                text = self._page_text(title=title, max_chars=max_chars)
            except self._wp.exceptions.DisambiguationError as exc:
                if not exc.options:
                    continue
                try:
                    option = sorted(exc.options)[self._rng.randint(0, len(exc.options) - 1)]
                    text = self._page_text(title=option, max_chars=max_chars)
                except Exception:
                    continue
            except Exception:
                continue

            text = " ".join(text.split()).strip()
            if len(text) >= min_chars:
                passages.append(text)

        # Fill shortages deterministically from fallback text to keep count stable.
        while len(passages) < count:
            rotation = len(passages) * 113
            doubled = f"{_FALLBACK_TEXT} {_FALLBACK_TEXT}"
            start = rotation % len(_FALLBACK_TEXT)
            filler = doubled[start : start + max_chars].strip()
            if len(filler) < min_chars:
                filler = (_FALLBACK_TEXT * 2)[:max_chars].strip()
            passages.append(filler)

        self._prefetched_passages = passages
        if snapshot_path:
            try:
                self._save_prefetch_snapshot(snapshot_path)
            except Exception:
                pass

        return len(self._prefetched_passages)

    def fetch_passage(self, min_chars: int = 500, max_chars: int = 3000) -> str:
        if self._prefetched_passages:
            for _ in range(10):
                text = self._rng.choice(self._prefetched_passages)[:max_chars].strip()
                if len(text) >= min_chars:
                    return text

        for _ in range(10):
            try:
                title = self._rng.choice(_WIKIPEDIA_TOPICS)
                text = self._page_text(title=title, max_chars=max_chars)
                if len(text) >= min_chars:
                    return text

            except self._wp.exceptions.DisambiguationError as exc:
                if not exc.options:
                    continue
                # Resolve ambiguity deterministically by taking sorted first.
                resolved_title = sorted(exc.options)[0]
                try:
                    text = self._page_text(title=resolved_title, max_chars=max_chars)
                    if len(text) >= min_chars:
                        return text
                except Exception:
                    continue

            except Exception:
                continue

        return _FALLBACK_TEXT

    def fetch_qa_pair(self, max_chars: int = 3000) -> tuple[str, str]:
        passage = self.fetch_passage(max_chars=max_chars)
        question = "What are the key facts presented in this passage?"

        return passage, question


_DEFAULT_TASKS = [
    TaskType.SUMMARIZE,
    TaskType.EXPLAIN,
    TaskType.CHAT,
    TaskType.CONTINUE,
]


def pick_task_and_suffix(
    source: TextSource,
    task: Optional[TaskType] = None,
    rng: Optional[random.Random] = None,
    qa_max_chars: int = 3000,
) -> tuple[TaskType, str]:
    """Pick a task and suffix using only the provided RNG.

    This helper is intended for deterministic suffix generation in simulator
    paths that do not need to build a full (prefix, suffix) prompt pair.
    """

    if rng is None:
        rng = random.Random()

    selected_task = task
    if selected_task is None:
        selected_task = (
            TaskType.QA if isinstance(source, SQuADSource) else rng.choice(_DEFAULT_TASKS)
        )

    if selected_task == TaskType.QA and isinstance(source, SQuADSource):
        _, question = source.fetch_qa_pair(max_chars=qa_max_chars)
        return (
            selected_task,
            f"Based on the passage above, answer the following question:\n{question}",
        )

    return selected_task, rng.choice(_SUFFIX_TEMPLATES[selected_task])


def build_prompt_pair(
    source: TextSource,
    task: Optional[TaskType] = None,
    min_prefix_chars: int = 400,
    max_prefix_chars: int = 3000,
    rng: Optional[random.Random] = None,
) -> PromptPair:
    """Build a PromptPair from source.

    Parameters
    ----------
    source:
        Any :class:`TextSource` instance.
    task:
        Task type.  If ``None``, a random task is chosen (``QA`` is preferred
        automatically for :class:`SQuADSource`).
    min_prefix_chars / max_prefix_chars:
        Character range for the retrieved passage.
    rng:
        Optional :class:`random.Random` for reproducible suffix selection.
    """

    if rng is None:
        rng = random.Random()

    # default task selection
    if task is None:
        task = (
            TaskType.QA
            if isinstance(source, SQuADSource)
            else rng.choice(_DEFAULT_TASKS)
        )

    # fetch passage (and question if QA + SQuAD)
    if task == TaskType.QA and isinstance(source, SQuADSource):
        context, question = source.fetch_qa_pair(max_chars=max_prefix_chars)
        suffix = (
            f"Based on the passage above, answer the following question:\n{question}"
        )

        return PromptPair(
            prefix=context, suffix=suffix, task=task, source_name=source.name
        )

    passage = source.fetch_passage(
        min_chars=min_prefix_chars, max_chars=max_prefix_chars
    )
    suffix = rng.choice(_SUFFIX_TEMPLATES[task])

    return PromptPair(prefix=passage, suffix=suffix, task=task, source_name=source.name)


def make_source(
    source_type: str = "wikitext",
    cache_dir: Optional[str] = None,
    seed: Optional[int] = None,
) -> TextSource:
    """Instantiate a :class:`TextSource` by name.

    Parameters
    ----------
    source_type:
        ``"wikitext"``   :class:`WikitextSource` (default, offline-friendly)\n
        ``"squad"``      :class:`SQuADSource` (real QA pairs)\n
        ``"wikipedia"``  :class:`WikipediaSource` (live; needs ``pip install wikipedia``)
    cache_dir:
        HuggingFace dataset cache directory (ignored for ``"wikipedia"``).
    seed:
        Random seed passed to the source.
    """

    key = source_type.lower()
    if key == "wikitext":
        return WikitextSource(cache_dir=cache_dir, seed=seed)
    if key in ("squad", "squadv1"):
        return SQuADSource(cache_dir=cache_dir, seed=seed)
    if key == "wikipedia":
        return WikipediaSource(seed=seed)
    raise ValueError(
        f"Unknown source_type {source_type!r}. "
        "Valid choices: 'wikitext', 'squad', 'wikipedia'."
    )


_WIKIPEDIA_TOPICS = [
    "history",
    "science",
    "mathematics",
    "literature",
    "philosophy",
    "geography",
    "biology",
    "physics",
    "economics",
    "technology",
    "art",
    "music",
    "architecture",
    "astronomy",
    "linguistics",
    "medicine",
    "engineering",
    "politics",
    "sociology",
    "ecology",
    "climate",
    "evolution",
    "democracy",
    "industrial revolution",
    "quantum mechanics",
    "Renaissance",
    "ancient Rome",
    "language",
]

_FALLBACK_TEXT = textwrap.dedent("""\
    The development of modern science has been one of the most transformative
    processes in human history. Beginning with the Scientific Revolution of the
    sixteenth and seventeenth centuries, thinkers such as Galileo Galilei,
    Johannes Kepler, and Isaac Newton established systematic methods for
    investigating natural phenomena through observation, experimentation, and
    mathematical reasoning. This approach gradually supplanted earlier
    explanations rooted in tradition or authority, opening the door to
    discoveries that reshaped humanity's understanding of the cosmos, life,
    and matter itself. Over the subsequent centuries, the natural sciences
    expanded dramatically, branching into specialised disciplines including
    chemistry, biology, geology, and eventually physics in its modern quantum
    and relativistic forms. Each new field brought fresh conceptual tools and
    experimental techniques, accelerating the rate at which knowledge accumulated.
    Today, science underpins virtually every aspect of contemporary life, from
    the medicines that treat disease to the digital infrastructure that connects
    billions of people across the globe.
""").strip()
