# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Dict, Any, Optional

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_precision,
        context_recall,
    )
except Exception:
    Dataset = None  # type: ignore
    evaluate = None  # type: ignore
    answer_relevancy = None  # type: ignore
    faithfulness = None  # type: ignore
    context_precision = None  # type: ignore
    context_recall = None  # type: ignore


class RAGASEvaluator:
    def __init__(
        self,
        llm: Optional[Any] = None,
        embeddings: Optional[Any] = None,
        metrics: Optional[List[Any]] = None,
    ) -> None:
        if evaluate is None or Dataset is None:
            raise ImportError("ragas and datasets are required. Install with: pip install ragas datasets")
        self.llm = llm
        self.embeddings = embeddings
        self.metrics = metrics or [
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
        ]

    def evaluate(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        samples must contain:
          - question: str
          - contexts: List[str]
          - answer: str
          - ground_truth: str
        """
        dataset = Dataset.from_list(samples)
        result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings,
            show_progress=True,
        )
        return dict(result)
