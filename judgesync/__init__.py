"""JudgeSync: Align LLM judges to human preferences on Azure."""

__version__ = "0.1.0"
__author__ = "James Asher"
__license__ = "MIT"
__url__ = "https://github.com/jasher4994/judgesync"

from .alignment import AlignmentTracker
from .comparison import ComparisonResults, JudgeComparison, JudgeConfig
from .data_loader import DataLoader
from .judge import Judge
from .metrics import AlignmentMetrics
from .types import AlignmentResults, EvaluationItem, ScoreRange

__all__ = [
    "__version__",
    "__author__",
    # Types
    "ScoreRange",
    "EvaluationItem",
    "AlignmentResults",
    # Main classes
    "DataLoader",
    "Judge",
    "AlignmentMetrics",
    "AlignmentTracker",
    # Comparison functionality
    "JudgeComparison",
    "JudgeConfig",
    "ComparisonResults",
]
