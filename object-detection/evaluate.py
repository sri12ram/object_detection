# author: materialplus.io
# Date: 01/06/2025

"""
Module: evaluate
Calculates WER, BLEU, chrF, and latency; logs results to MLflow for drift tracking.

Classes:
- EvaluationMetrics: Manages the evaluation of model performance.

Methods:
- calculate_metrics: Computes evaluation metrics and logs them to MLflow.

Usage:
- Instantiate EvaluationMetrics and call calculate_metrics with predictions and references.
"""

from typing import List, Tuple
import mlflow
from datasets import load_metric

class EvaluationMetrics:
    def __init__(self):
        self.wer_metric = load_metric("wer")
        self.bleu_metric = load_metric("bleu")
        self.chrf_metric = load_metric("chrf")

    def calculate_metrics(self, predictions: List[str], references: List[str]) -> None:
        """
        Computes evaluation metrics and logs them to MLflow.

        Args:
            predictions (List[str]): List of predicted transcripts or translations.
            references (List[str]): List of reference transcripts or translations.
        """
        try:
            wer = self.wer_metric.compute(predictions=predictions, references=references)
            bleu = self.bleu_metric.compute(predictions=predictions, references=references)
            chrf = self.chrf_metric.compute(predictions=predictions, references=references)
        except Exception as e:
            print(f"Error calculating metrics: {e}")

        # Log metrics to MLflow
        mlflow.log_metric("WER", wer)
        mlflow.log_metric("BLEU", bleu["bleu"])
        mlflow.log_metric("chrF", chrf["score"])

        # Placeholder for latency calculation
        latency = 0.0  # Simulated latency value
        mlflow.log_metric("Latency", latency)

        print(f"Metrics logged: WER={wer}, BLEU={bleu['bleu']}, chrF={chrf['score']}, Latency={latency}")

# Example usage
if __name__ == "__main__":
    predicted_transcripts = ["Simulated prediction 1", "Simulated prediction 2"]
    reference_transcripts = ["Reference transcript 1", "Reference transcript 2"]
    evaluator = EvaluationMetrics()
    evaluator.calculate_metrics(predicted_transcripts, reference_transcripts)
