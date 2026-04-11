from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class HuggingFaceNLIClient:
    """
    Love-OS NLI Backend for Zero-Time ∞/∞ Detection.
    Core module for detecting logical dissipation and divergence.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.label_map = {0: "contradiction", 1: "entailment", 2: "neutral"}
        self.contradiction_idx = 0
        self.entailment_idx = 1
        self.neutral_idx = 2

    def batch_predict(self, pairs: List[Tuple[str, str]]) -> List[Dict]:
        """Calculates raw probabilities for logical states."""
        if not pairs:
            return []

        inputs = self.tokenizer(
            pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

        return [
            {
                "contradiction": float(p[0]),
                "entailment": float(p[1]),
                "neutral": float(p[2]),
                "max_label": self.label_map[int(np.argmax(p))],
            }
            for p in probs
        ]

    def divergence_score(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Calculates ∞/∞ divergence score.
        Measures logical 'friction' weighted against 'synchronization'.
        """
        eps = 1e-6
        preds = self.batch_predict(pairs)
        return [
            float(p["contradiction"] / (p["entailment"] + 0.2 * p["neutral"] + eps))
            for p in preds
        ]

    def diagnose(
        self,
        pairs: List[Tuple[str, str]],
        contradiction_threshold: float = 0.65,
        divergence_threshold: float = 1.5,
    ) -> List[Dict]:
        """Performs a structural diagnosis of logical alignment."""
        preds = self.batch_predict(pairs)
        divergences = self.divergence_score(pairs)

        return [
            {
                "contradiction": p["contradiction"],
                "entailment": p["entailment"],
                "neutral": p["neutral"],
                "divergence": div,
                "state": self._state_label(
                    p["contradiction"], p["entailment"], div,
                    contradiction_threshold, divergence_threshold
                ),
            }
            for p, div in zip(preds, divergences)
        ]

    def _state_label(self, contradiction: float, entailment: float, divergence: float,
                     c_th: float, d_th: float) -> str:
        """Determines the systemic state based on logical entropy."""
        if contradiction >= c_th:
            return "CONTRADICTION"      # Critical logical collapse -> Trigger Zero-Ritual
        if divergence >= d_th:
            return "DIVERGENT"          # ∞/∞ State: Information leakage/instability
        if entailment > contradiction + 0.15:
            return "CONSISTENT"         # High Sync (Love State): Zero dissipation
        return "UNDECIDED"
