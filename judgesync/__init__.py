"""JudgeSync: Align LLM judges to human preferences."""

__version__ = "0.1.0"
__author__ = "James Asher"

# Import all main classes and types
from .alignment import AlignmentTracker
from .comparison import ComparisonResults, JudgeComparison, JudgeConfig
from .data_loader import DataLoader
from .judge import Judge
from .metrics import AlignmentMetrics
from .types import AlignmentResults, EvaluationItem, ScoreRange

# Define what's available when someone does 'from judgesync import *'
__all__ = [
    # Version info
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
