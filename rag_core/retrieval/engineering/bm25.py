"""A small, dependency-free BM25 retriever for code and technical prose."""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Callable, Iterable, Sequence

from .models import EngineeringSearchResult


Tokenizer = Callable[[str], list[str]]

_RAW_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:[.:/\\-][A-Za-z0-9_]+)*"
    r"|\d+(?:\.\d+)*"
    r"|[\u3400-\u4dbf\u4e00-\u9fff]+"
)
_CAMEL_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+")
_CJK_RE = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]+$")


def engineering_tokenize(text: str) -> list[str]:
    """Tokenize identifiers, paths and Chinese prose without extra packages.

    Exact identifiers are retained and ASCII components are also emitted, so
    both ``QueryEngine.submit_message`` and ``submit_message`` can match. CJK
    runs retain the full run plus character bigrams for short Chinese queries.
    """

    tokens: list[str] = []
    for match in _RAW_TOKEN_RE.finditer(text):
        raw = match.group(0)
        if _CJK_RE.match(raw):
            tokens.append(raw)
            if len(raw) > 1:
                tokens.extend(raw[i : i + 2] for i in range(len(raw) - 1))
            continue

        lowered = raw.casefold()
        tokens.append(lowered)
        components = [part for part in re.split(r"[.:/\\-]+", raw) if part]
        for component in components:
            component_lower = component.casefold()
            if component_lower != lowered:
                tokens.append(component_lower)
            camel_parts = [part.casefold() for part in _CAMEL_RE.findall(component)]
            if len(camel_parts) > 1:
                tokens.extend(camel_parts)
    return tokens


class BM25Retriever:
    """Rank :class:`EngineeringSearchResult` objects using Okapi BM25."""

    def __init__(
        self,
        documents: Iterable[EngineeringSearchResult],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Tokenizer = engineering_tokenize,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.documents = tuple(documents)
        self.k1 = float(k1)
        self.b = float(b)
        self.tokenizer = tokenizer
        self._term_frequencies: list[Counter[str]] = []
        self._document_lengths: list[int] = []
        document_frequency: Counter[str] = Counter()

        for document in self.documents:
            tokens = tokenizer(document.content)
            frequencies = Counter(tokens)
            self._term_frequencies.append(frequencies)
            self._document_lengths.append(len(tokens))
            document_frequency.update(frequencies.keys())

        self._average_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths
            else 0.0
        )
        count = len(self.documents)
        self._idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        corpus: str | None = None,
        authority: str | None = None,
    ) -> list[EngineeringSearchResult]:
        """Return positive-scoring results in descending BM25 order."""

        if top_k <= 0 or not self.documents:
            return []
        query_terms = self.tokenizer(query)
        if not query_terms:
            return []

        query_frequency = Counter(query_terms)
        ranked: list[tuple[float, int, EngineeringSearchResult]] = []
        for index, document in enumerate(self.documents):
            if corpus is not None and document.corpus != corpus:
                continue
            if authority is not None and document.authority != authority:
                continue

            length = self._document_lengths[index]
            normalization = 1.0 - self.b
            if self._average_length:
                normalization += self.b * length / self._average_length

            score = 0.0
            frequencies = self._term_frequencies[index]
            for term, query_count in query_frequency.items():
                term_frequency = frequencies.get(term, 0)
                if not term_frequency:
                    continue
                denominator = term_frequency + self.k1 * normalization
                score += (
                    self._idf.get(term, 0.0)
                    * (term_frequency * (self.k1 + 1.0) / denominator)
                    * query_count
                )

            if score > 0:
                ranked.append((score, index, document))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [
            document.updated(score=score, retriever="bm25")
            for score, _, document in ranked[:top_k]
        ]

    @classmethod
    def from_texts(
        cls,
        texts: Sequence[str],
        *,
        sources: Sequence[str] | None = None,
        corpus: str = "internal",
        authority: str = "unknown",
        **kwargs: object,
    ) -> "BM25Retriever":
        """Convenience constructor for small in-memory corpora."""

        if sources is not None and len(sources) != len(texts):
            raise ValueError("sources and texts must have the same length")
        documents = [
            EngineeringSearchResult(
                content=text,
                source=sources[index] if sources is not None else f"document-{index}",
                corpus=corpus,
                authority=authority,
            )
            for index, text in enumerate(texts)
        ]
        return cls(documents, **kwargs)
