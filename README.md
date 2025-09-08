# JudgeSync

A Python package for aligning LLM judges to human preferences. JudgeSync helps you statistically measure and improve the alignment between automated LLM evaluations and human judgments using metrics like Cohen's kappa.

## Features

- 📊 **Statistical Alignment Metrics**: Calculate Cohen's kappa, agreement rates, and correlation coefficients
- 🤖 **Azure OpenAI Integration**: Built-in support for using Azure OpenAI models as judges
- 📁 **Flexible Data Loading**: Load human scores from CSV files or add them programmatically
- 🎯 **Multiple Scoring Ranges**: Support for binary, Likert scale, percentage, and custom scoring systems
- 📈 **Alignment Tracking**: Track improvement across different prompts and configurations
- 💾 **Prompt Export**: Export successful prompts for production use

## Installation

```bash
pip install judgesync
```

For development:

```bash
git clone https://github.com/yourusername/judgesync.git
cd judgesync
pip install -e ".[dev]"
```

## Quick Start

```python
from judgesync import AlignmentTracker, ScoreRange

# Initialize tracker with a 5-point scale
tracker = AlignmentTracker(score_range=ScoreRange.FIVE_POINT)

# Load human-labeled data
tracker.load_human_scores_from_csv(
    "data.csv",
    question_col="question",
    response_col="response", 
    score_col="human_score"
)

# Configure the LLM judge
system_prompt = """You are an expert evaluator. Rate responses on a scale of 1-5:
5 = Excellent
4 = Good  
3 = Adequate
2 = Poor
1 = Very poor"""

tracker.set_judge(system_prompt)

# Run alignment test
results = tracker.run_alignment_test()

# View results
print(f"Kappa Score: {results.kappa_score:.3f}")
print(f"Agreement Rate: {results.agreement_rate:.2%}")

# Export the prompt
tracker.export_prompt("optimized_prompt.txt")
```

## Configuration

### Azure OpenAI Setup

Create a `.env` file in your project root:

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
```

Or pass credentials directly:

```python
tracker.set_judge(
    system_prompt="...",
    azure_endpoint="...",
    api_key="...",
    deployment_name="..."
)
```

### Scoring Ranges

JudgeSync supports multiple scoring ranges:

```python
from judgesync import ScoreRange

ScoreRange.BINARY       # 0-1 (Pass/Fail)
ScoreRange.FIVE_POINT   # 1-5 (Likert scale)
ScoreRange.PERCENTAGE   # 0-100
ScoreRange.TEN_POINT    # 1-10
```

## Data Format

### CSV Format

Your CSV should have columns for questions, responses, and human scores:

```csv
question,response,human_score
"What is 2+2?","4",5
"Capital of France?","Paris",5
"Explain gravity","It makes things fall",3
```

### Programmatic Input

```python
tracker.add_evaluation_item(
    question="What is the capital of France?",
    response="Paris is the capital of France.",
    human_score=5,
    metadata={"category": "geography"}
)
```

## Metrics

JudgeSync calculates several alignment metrics:

- **Cohen's Kappa**: Inter-rater reliability (-1 to 1, higher is better)
- **Agreement Rate**: Percentage of scores within tolerance
- **Correlation**: Pearson or Spearman correlation coefficients
- **Confusion Matrix**: For analyzing disagreement patterns

## Advanced Usage

### Testing Multiple Prompts

```python
prompts = [
    "Be a strict evaluator...",
    "Be a lenient evaluator...",
    "Focus on accuracy..."
]

for prompt in prompts:
    tracker.set_judge(prompt)
    results = tracker.run_alignment_test()
    print(f"Prompt: {prompt[:30]}... -> Kappa: {results.kappa_score:.3f}")

# Get the best performing prompt
best = tracker.get_best_prompt()
```

### Custom Metrics

```python
from judgesync import AlignmentMetrics

metrics = AlignmentMetrics(score_range=ScoreRange.PERCENTAGE)
correlation = metrics.calculate_correlation(items, method="spearman")
confusion_matrix = metrics.get_confusion_matrix(items)
```

## Requirements

- Python 3.8+
- Azure OpenAI API access (for judge functionality)

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- openai >= 1.0.0
- azure-identity >= 1.14.0
- python-dotenv >= 0.19.0

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint code
ruff check .

# Format code
ruff format .
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

James Asher

## Version

0.1.0
