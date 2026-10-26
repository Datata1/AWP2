"""Multi-task neural networks for crop and growth-stage classification."""

from awp2.models.multi_task.lstm import MultiTaskLSTMClassifier
from awp2.models.multi_task.mlp import MultiTaskMLPClassifier

__all__ = ["MultiTaskLSTMClassifier", "MultiTaskMLPClassifier"]
