# author: materialplus.io
# Date: 01/06/2025

"""
Module: active_learning
Ingests human-corrected SRTs nightly and triggers incremental retraining of ASR and MT models.

Classes:
- ActiveLearningPipeline: Manages the active learning process for model improvement.

Methods:
- ingest_and_retrain: Ingests corrected SRTs and retrains models incrementally.

Usage:
- Instantiate ActiveLearningPipeline and call ingest_and_retrain with SRT file paths.
"""

from typing import List
import os

class ActiveLearningPipeline:
    def __init__(self, asr_model_name: str, mt_model_name: str):
        self.asr_model_name = asr_model_name
        self.mt_model_name = mt_model_name

    def ingest_and_retrain(self, srt_files: List[str]) -> None:
        """
        Ingests corrected SRTs and retrains models incrementally.

        Args:
            srt_files (List[str]): List of paths to human-corrected SRT files.
        """
        try:
            for srt_file in srt_files:
                print(f"Ingesting and retraining with {srt_file}")
        except Exception as e:
            print(f"Error during ingestion and retraining: {e}")

        # Simulate retraining process
        print(f"Retraining ASR model: {self.asr_model_name}")
        print(f"Retraining MT model: {self.mt_model_name}")

# Example usage
if __name__ == "__main__":
    corrected_srt_paths = ["path/to/corrected1.srt", "path/to/corrected2.srt"]
    active_learning_pipeline = ActiveLearningPipeline("Whisper-v3", "SeamlessM4T-v2-large")
    active_learning_pipeline.ingest_and_retrain(corrected_srt_paths)
