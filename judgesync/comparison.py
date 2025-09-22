"""Multiple judge comparison functionality."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from .judge import Judge
from .metrics import AlignmentMetrics
from .types import AlignmentResults, EvaluationItem, ScoreRange

logger = logging.getLogger(__name__)


@dataclass
class JudgeConfig:
    """Configuration for a judge in comparison."""

    name: str
    system_prompt: str
    deployment_name: Optional[str] = None
    temperature: float = 0.2

    def __hash__(self):
        """Make hashable for use as dict key."""
        return hash(
            (self.name, self.system_prompt, self.deployment_name, self.temperature)
        )


@dataclass
class ComparisonResults:
    """Results from comparing multiple judges."""

    judge_results: Dict[str, AlignmentResults]
    rankings: pd.DataFrame
    best_judge: str
    detailed_scores: Optional[pd.DataFrame] = None

    def __str__(self) -> str:
        """Pretty print comparison results."""
        output = ["=" * 60]
        output.append("JUDGE COMPARISON RESULTS")
        output.append("=" * 60)

        output.append("\nRankings by Kappa Score:")
        for idx, row in self.rankings.iterrows():
            output.append(
                f"  {idx + 1}. {row['judge']}: "
                f"κ={row['kappa']:.3f}, "
                f"Agreement={row['agreement']:.1%}, "
                f"Correlation={row['correlation']:.3f}"
            )

        output.append(f"\n🏆 Best Judge: {self.best_judge}")
        output.append("=" * 60)

        return "\n".join(output)

    def to_dataframe(self) -> pd.DataFrame:
        """Export comparison results as a DataFrame."""
        return self.rankings


class JudgeComparison:
    """Compare multiple judges to find the best alignment with human scores."""

    def __init__(
        self,
        score_range: ScoreRange = ScoreRange.FIVE_POINT,
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize the judge comparison.

        Args:
            score_range: The scoring range to use.
            azure_endpoint: Azure OpenAI endpoint.
            api_key: Azure OpenAI API key.
        """
        self.score_range = score_range
        self.azure_endpoint = azure_endpoint
        self.api_key = api_key

        self.judges: Dict[str, Judge] = {}
        self.configs: Dict[str, JudgeConfig] = {}
        self.metrics = AlignmentMetrics(score_range=score_range)

    def add_judge(
        self,
        name: str,
        system_prompt: str,
        deployment_name: Optional[str] = None,
        temperature: float = 0.2,
    ) -> None:
        """Add a judge configuration to compare.

        Args:
            name: Unique name for this judge configuration.
            system_prompt: The system prompt for the judge.
            deployment_name: Optional Azure deployment name (e.g., "gpt-4", "gpt-3.5").
            temperature: Temperature setting for the judge.

        Example:
            >>> comparison = JudgeComparison()
            >>> comparison.add_judge("strict", "Be very strict in your evaluation.")
            >>> comparison.add_judge("lenient", "Be generous in your evaluation.")
        """
        if name in self.judges:
            raise ValueError(f"Judge '{name}' already exists")

        config = JudgeConfig(
            name=name,
            system_prompt=system_prompt,
            deployment_name=deployment_name,
            temperature=temperature,
        )

        self.configs[name] = config

        self.judges[name] = Judge(
            system_prompt=system_prompt,
            score_range=self.score_range,
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            deployment_name=deployment_name,
            temperature=temperature,
        )

    def add_judge_from_instance(self, name: str, judge: Judge) -> None:
        """Add an existing judge instance.

        Args:
            name: Unique name for this judge.
            judge: The Judge instance to add.
        """
        if name in self.judges:
            raise ValueError(f"Judge '{name}' already exists")

        self.judges[name] = judge
        self.configs[name] = JudgeConfig(
            name=name,
            system_prompt=judge.system_prompt,
            deployment_name=judge.deployment_name,
            temperature=judge.temperature,
        )

    def run_comparison(
        self,
        items: List[EvaluationItem],
        use_async: bool = True,
        batch_size: int = 10,
        calculate_correlation: bool = True,
    ) -> ComparisonResults:
        """Run all judges on the same items and compare results.

        Args:
            items: List of evaluation items with human scores.
            use_async: Whether to use async batch processing.
            batch_size: Batch size for async processing.
            calculate_correlation: Whether to calculate correlation metrics.

        Returns:
            ComparisonResults with rankings and detailed metrics.

        Example:
            >>> items = [EvaluationItem(...) for _ in range(100)]
            >>> results = comparison.run_comparison(items)
            >>> print(results)
        """
        if not self.judges:
            raise ValueError("No judges added for comparison")

        items_with_scores = [item for item in items if item.human_score is not None]
        if not items_with_scores:
            raise ValueError("No items have human scores")

        logger.info(
            f"Comparing {len(self.judges)} judges on {len(items_with_scores)} items..."
        )

        results = {}
        all_scores = {}

        for name, judge in self.judges.items():
            logger.info(f"Running judge '{name}'...")

            judge_items = [
                EvaluationItem(
                    question=item.question,
                    response=item.response,
                    human_score=item.human_score,
                    metadata=item.metadata,
                )
                for item in items_with_scores
            ]

            if use_async:
                scored_items = judge.score_items_async(
                    judge_items, batch_size=batch_size, show_progress=False
                )
            else:
                scored_items = judge.score_items(judge_items, use_async=False)

            alignment_results = self.metrics.calculate(scored_items)
            results[name] = alignment_results

            all_scores[name] = [item.judge_score for item in scored_items]

            if calculate_correlation:
                try:
                    correlation = self.metrics.calculate_correlation(scored_items)
                    results[name].correlation = correlation
                except Exception as e:
                    logger.warning(f"Could not calculate correlation for {name}: {e}")
                    results[name].correlation = 0.0

        rankings_data = []
        for name, result in results.items():
            rankings_data.append(
                {
                    "judge": name,
                    "kappa": result.kappa_score,
                    "agreement": result.agreement_rate,
                    "correlation": getattr(result, "correlation", 0.0),
                    "sample_size": result.sample_size,
                }
            )

        rankings = pd.DataFrame(rankings_data)
        rankings = rankings.sort_values("kappa", ascending=False).reset_index(drop=True)

        best_judge = rankings.iloc[0]["judge"]

        detailed_scores = pd.DataFrame(all_scores)
        detailed_scores["human_score"] = [
            item.human_score for item in items_with_scores
        ]

        cols = ["human_score"] + [
            col for col in detailed_scores.columns if col != "human_score"
        ]
        detailed_scores = detailed_scores[cols]

        return ComparisonResults(
            judge_results=results,
            rankings=rankings,
            best_judge=best_judge,
            detailed_scores=detailed_scores,
        )

    def plot_comparison(
        self,
        results: ComparisonResults,
        save_path: Optional[str] = None,
        dpi: int = 100,
        show: bool = True,
    ) -> None:
        """Create visualizations comparing judges.

        Args:
            results: The comparison results to visualize.
            save_path: Optional path to save the figure (e.g., "comparison.png").
            dpi: Resolution for saved figure.
            show: Whether to display the figure.
        """
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # 1. Bar chart of kappa scores
            ax = axes[0, 0]
            judges = results.rankings["judge"].tolist()
            kappas = results.rankings["kappa"].tolist()
            bars = ax.bar(judges, kappas)
            bars[0].set_color("green")  # Highlight best
            ax.set_ylabel("Kappa Score")
            ax.set_title("Cohen's Kappa by Judge")
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=45)

            # 2. Agreement rates
            ax = axes[0, 1]
            agreements = results.rankings["agreement"].tolist()
            bars = ax.bar(judges, agreements)
            bars[0].set_color("green")
            ax.set_ylabel("Agreement Rate")
            ax.set_title("Agreement Rate by Judge")
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=45)

            # 3. Score distribution comparison
            ax = axes[0, 2]
            if results.detailed_scores is not None:
                # Plot distribution for human scores and best judge
                best_judge = results.best_judge
                human_scores = results.detailed_scores["human_score"]
                judge_scores = results.detailed_scores[best_judge]

                # Create histogram
                bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
                ax.hist(
                    [human_scores, judge_scores],
                    bins=bins,
                    label=["Human", best_judge],
                    alpha=0.7,
                    edgecolor="black",
                )
                ax.set_xlabel("Score")
                ax.set_ylabel("Frequency")
                ax.set_title(f"Score Distribution: Human vs {best_judge}")
                ax.legend()
                ax.set_xticks([1, 2, 3, 4, 5])

            # 4. Scatter plot of human vs judge scores (best judge)
            ax = axes[1, 0]
            if results.detailed_scores is not None:
                best_judge = results.best_judge
                ax.scatter(
                    results.detailed_scores["human_score"],
                    results.detailed_scores[best_judge],
                    alpha=0.5,
                )
                ax.plot([0, 5], [0, 5], "r--", alpha=0.5)  # Perfect agreement line
                ax.set_xlabel("Human Score")
                ax.set_ylabel(f"{best_judge} Score")
                ax.set_title(f"Best Judge ({best_judge}) vs Human Scores")

            # 5. Correlation comparison
            ax = axes[1, 1]
            correlations = results.rankings["correlation"].tolist()
            bars = ax.bar(judges, correlations)
            bars[0].set_color("green")
            ax.set_ylabel("Correlation")
            ax.set_title("Pearson Correlation by Judge")
            ax.set_ylim(-1, 1)
            ax.tick_params(axis="x", rotation=45)

            # 6. Distribution comparison for all judges
            ax = axes[1, 2]
            if results.detailed_scores is not None:
                judge_cols = [
                    col
                    for col in results.detailed_scores.columns
                    if col != "human_score"
                ]
                data_to_plot = [
                    results.detailed_scores[col].dropna()
                    for col in ["human_score"] + judge_cols
                ]
                labels = ["Human"] + judge_cols

                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                bp["boxes"][0].set_facecolor("lightblue")
                best_idx = judge_cols.index(results.best_judge) + 1
                bp["boxes"][best_idx].set_facecolor("lightgreen")

                ax.set_ylabel("Score")
                ax.set_title("Score Distribution Comparison (All Judges)")
                ax.tick_params(axis="x", rotation=45)

            plt.suptitle(
                f"Judge Comparison Results - Best: {results.best_judge}",
                fontsize=14,
                fontweight="bold",
            )

            plt.tight_layout(rect=[0, 0.03, 1, 0.97])

            if save_path:
                plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
                logger.info(f"Figure saved to: {save_path}")

            if show:
                plt.show()
            else:
                plt.close()

        except ImportError:
            logger.error(
                "Matplotlib not installed. Install with: pip install matplotlib"
            )

    def get_disagreement_items(
        self, results: ComparisonResults, threshold: float = 1.0
    ) -> pd.DataFrame:
        """Find items where judges disagree the most.

        Args:
            results: The comparison results.
            threshold: Minimum standard deviation to consider as disagreement.

        Returns:
            DataFrame of items with high disagreement.
        """
        if results.detailed_scores is None:
            return pd.DataFrame()

        judge_cols = [
            col for col in results.detailed_scores.columns if col != "human_score"
        ]

        df = results.detailed_scores.copy()

        judge_std = df[judge_cols].std(axis=1)

        high_disagreement = df[judge_std >= threshold].copy()

        return high_disagreement
