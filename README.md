
# JudgeSync 🧑‍⚖️

JudgeSync is a lightweight Python package for calibrating LLM judges to align with human evaluations. It helps minimize bias and improve reliability in LLM-as-a-judge workflows by comparing different judge configurations and finding the best alignment with human scores.

## Why JudgeSync?

LLM judges are powerful but exhibit biases:
- **Verbosity bias**: Favoring longer responses even with errors
- **Position bias**: Preferring the first option presented
- **Self-bias**: Favoring outputs from similar models
- **Leniency/Strictness**: Inconsistent scoring patterns

JudgeSync helps you find the optimal judge configuration (prompt, model, temperature) that best aligns with human judgments.

## Installation

```bash
pip install judgesync
```


## Quick Start

```python
from judgesync import AlignmentTracker, ScoreRange

# Load your evaluation data with human scores
tracker = AlignmentTracker(score_range=ScoreRange.FIVE_POINT)
tracker.load_human_scores_from_csv("evaluation_data.csv")

# Ensure Azure OpenAI credentials are configured (see section below)
prompt_comparison = tracker.create_comparison()

prompt_comparison.add_judge(
    name="strict",
    system_prompt="You are a strict evaluator. Only give high scores to exceptional responses.",
)

prompt_comparison.add_judge(
    name="balanced",
    system_prompt="You are a balanced evaluator. Consider both strengths and weaknesses fairly.",
)

prompt_comparison.add_judge(
    name="detailed_rubric",
    system_prompt="""Rate responses on a 1-5 scale:\n5: Comprehensive, accurate, well-structured\n4: Good accuracy, minor gaps\n3: Adequate, addresses main points\n2: Partially correct, significant gaps\n1: Incorrect or irrelevant""",
)

# Run comparison and find the best judge
results = prompt_comparison.run_comparison(tracker.data_loader.items, use_async=True)
print(results)

# Visualize results (requires matplotlib: pip install matplotlib)
prompt_comparison.plot_comparison(results, save_path="judge_comparison.png")
```

## Key Metrics

### Cohen's Kappa (κ)
Measures agreement between human and LLM judge, accounting for chance agreement:
- **κ > 0.7**: Production-ready alignment
- **κ = 0.4-0.7**: Good alignment, may need fine-tuning
- **κ < 0.4**: Poor alignment, needs improvement

### Agreement Rate
Percentage of exact score matches between human and judge.

## Features

### 🎯 Score Alignment
- Support for multiple scoring scales (binary, 5-point, 10-point, percentage)
- Automatic score range validation
- Statistical metrics (Cohen's Kappa, correlation, agreement rate)

### 🔬 Judge Comparison
- Test multiple prompts/models simultaneously
- Async batch processing for efficiency
- Identify optimal judge configuration

### 📊 Visualization
- Performance comparison charts
- Score distribution analysis
- Disagreement identification

### 🎛️ Model Configuration
- Compare different models (GPT-4, GPT-3.5, etc.)
- Test temperature settings
- Experiment with custom prompts

## Advanced Usage


### Compare Different Models

```python
configs = [
    JudgeConfig(
        name="gpt-4-cold",
        system_prompt="Rate the response quality.",
        deployment_name="gpt-4",
        temperature=0.0,
    ),
    JudgeConfig(
        name="gpt-4-warm",
        system_prompt="Rate the response quality.",
        deployment_name="gpt-4",
        temperature=0.7,
    ),
]
comparison = JudgeComparison(configs, items)
results = comparison.run_comparison()
```


### Analyze Disagreements

```python
# Find items where judges disagree significantly
disagreements = comparison.get_disagreement_items(results, threshold=1.0)
print(f"Found {len(disagreements)} items with high disagreement")
```

### Custom Azure OpenAI Configuration
(Set these before using `create_comparison().`)

```python
# Option 1: Environment variables (.env file)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4


# Option 2: Direct configuration
judge = Judge(
    system_prompt="Your prompt here",
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_key="your-api-key",
    deployment_name="gpt-4"
)
```

## CSV Format


Your evaluation data should have these columns:
- `question`: The input/prompt
- `response`: The response to evaluate
- `human_score`: The human-assigned score

```csv
question,response,human_score
What is the capital of France?,"Paris is the capital of France.",5
Explain photosynthesis.,"Plants make food from sunlight.",3
```


## Visualization Examples

JudgeSync generates comprehensive comparison charts showing:
- Cohen's Kappa scores by judge
- Agreement rates
- Score distributions
- Correlation analysis

![Judge Comparison Example](examples/prompt_comparison.png)

## Best Practices

1. **Start with diverse test data**: Include responses across all score ranges
2. **Test multiple prompts**: Even small wording changes can impact alignment
3. **Consider temperature**: Lower temperatures (0.0-0.3) often provide more consistent scoring
4. **Validate on held-out data**: Ensure your calibrated judge generalizes well
5. **Monitor for biases**: Check if judges favor certain response styles

## Requirements

- Python 3.9+
- Azure OpenAI API access
- pandas
- numpy
- scikit-learn
- python-dotenv

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, coding standards, and our preferred workflow.

## License

MIT License - See LICENSE file for details.

## Citation

If you use JudgeSync in your research, please cite:

```bibtex
@software{judgesync2025,
  title = {JudgeSync: Calibrating LLM Judges with Human Feedback},
  author = {Asher, James},
  year = {2025},
  url = {https://github.com/jasher4994/judgesync}
}
```

## Acknowledgments

Inspired by research on LLM judge calibration and the challenges observed in production LLM evaluation systems.
